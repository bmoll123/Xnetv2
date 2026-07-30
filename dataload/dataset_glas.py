import os
import glob
import random

import numpy as np
import cv2
import pywt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset

# Channel statistics for the GlaS RGB images (config/dataset_config/dataset_cfg.py -> 'GlaS').
GLAS_MEAN = [0.787803, 0.512017, 0.784938]
GLAS_STD = [0.428206, 0.507778, 0.426366]


def _read_id_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _all_train_ids(data_root):
    list_path = os.path.join(data_root, 'train_slices.list')
    if os.path.exists(list_path):
        return _read_id_list(list_path)
    img_dir = os.path.join(data_root, 'train', 'images')
    return sorted(os.path.splitext(fn)[0] for fn in os.listdir(img_dir))


def resolve_split(data_root, portion, split_seed):
    """Resolve (labeled_ids, unlabeled_ids, val_ids) for the requested --portion.

    'unsegSplit10'/'unsegSplit20' reuse the fixed partitions shipped under
    <data_root>/partitions/glas_{10,20}. '10%'/'20%' keep the same fixed 12-image
    val split (identical between glas_10 and glas_20) but draw the labeled subset
    (7 or 14 images) via a dedicated RNG seeded by --split_seed, independent of the
    global --seed used for training reproducibility.
    """
    partitions_dir = os.path.join(data_root, 'partitions')

    if portion in ('unsegSplit10', 'unsegSplit20'):
        pct = '10' if portion == 'unsegSplit10' else '20'
        pdir = os.path.join(partitions_dir, 'glas_' + pct)
        labeled = _read_id_list(os.path.join(pdir, 'labeled.txt'))
        unlabeled = _read_id_list(os.path.join(pdir, 'unlabeled.txt'))
        val = _read_id_list(os.path.join(pdir, 'val.txt'))
        return labeled, unlabeled, val

    if portion in ('10%', '20%'):
        pct = '10' if portion == '10%' else '20'
        val = _read_id_list(os.path.join(partitions_dir, 'glas_' + pct, 'val.txt'))
        all_ids = _all_train_ids(data_root)
        pool = sorted(set(all_ids) - set(val))
        k = 7 if portion == '10%' else 14
        rng = random.Random(split_seed)
        labeled = sorted(rng.sample(pool, k))
        unlabeled = sorted(set(pool) - set(labeled))
        return labeled, unlabeled, val

    raise ValueError('unknown portion {}'.format(portion))


def _find_image_path(img_dir, stem):
    matches = glob.glob(os.path.join(img_dir, stem + '.*'))
    if not matches:
        raise FileNotFoundError('no image found for {} in {}'.format(stem, img_dir))
    return matches[0]


def _norm01_255(x):
    lo = np.amin(x, (0, 1))
    hi = np.amax(x, (0, 1))
    span = np.where(hi - lo == 0, 1, hi - lo)
    return (x - lo) / span * 255


def wavelet_LH(img_arr, wavelet_type, alpha_range, beta_range):
    """Reproduce XNetv2's image-level LF/HF fusion (dataload/dataset_2d.py)."""
    LL, (LH, HL, HH) = pywt.dwt2(img_arr, wavelet_type, axes=(0, 1))

    LL = _norm01_255(LL)
    LH = _norm01_255(LH)
    HL = _norm01_255(HL)
    HH = _norm01_255(HH)

    H_ = HL + LH + HH
    H_ = _norm01_255(H_)

    a = random.uniform(alpha_range[0], alpha_range[1])
    L = _norm01_255(LL + a * H_)

    b = random.uniform(beta_range[0], beta_range[1])
    H = _norm01_255(H_ + b * LL)

    return L.astype(np.float32), H.astype(np.float32)


class GlasSegDataset(Dataset):
    """GLAS dataset reading directly from the raw <data_root>/{train,test} layout.

    mode='train_aug': flip + transpose + rotate90, then pad-to-at-least and random
    crop to crop_size x crop_size (all applied identically to image/L/H/mask).
    mode='eval': no crop, just pad up to a multiple of pad_divisor so the fully
    convolutional network (4 downsamplings) can run on the native resolution.
    Because pywt.dwt2 halves L/H spatial size, every sample is first passed through
    an A.Resize back to the image's own native (h, w) so image/L/H/mask stay aligned
    for the rest of the pipeline.
    """

    SPLIT_DIRS = {
        'train': 'train',
        'testA': os.path.join('test', 'testA'),
        'testB': os.path.join('test', 'testB'),
    }

    def __init__(self, data_root, ids, split, mode, wavelet_type='haar',
                 alpha=(0.0, 0.4), beta=(0.5, 0.8), crop_size=512, pad_divisor=32,
                 target_len=None, resample_seed=0):
        assert split in self.SPLIT_DIRS, split
        assert mode in ('train_aug', 'eval'), mode
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.wavelet_type = wavelet_type
        self.alpha = alpha
        self.beta = beta
        self.crop_size = crop_size
        self.pad_divisor = pad_divisor

        split_dir = os.path.join(data_root, self.SPLIT_DIRS[split])
        self.img_dir = os.path.join(split_dir, 'images')
        self.mask_dir = os.path.join(split_dir, 'labels')

        ids = list(ids)
        if target_len is not None and len(ids) > 0 and target_len != len(ids):
            base_len = len(ids)
            quotient, remainder = divmod(target_len, base_len)
            g = torch.Generator().manual_seed(resample_seed)
            rand_indices = torch.randperm(base_len, generator=g).tolist()[:remainder]
            ids = ids * quotient + [ids[i] for i in rand_indices]
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def _build_transform(self, h0, w0):
        if self.mode == 'train_aug':
            return A.Compose([
                A.Resize(h0, w0, p=1),
                A.Flip(p=0.75),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=1),
                A.PadIfNeeded(min_height=self.crop_size, min_width=self.crop_size,
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                              position='top_left'),
                A.RandomCrop(height=self.crop_size, width=self.crop_size),
            ], additional_targets={'L': 'image', 'H': 'image'})
        return A.Compose([
            A.Resize(h0, w0, p=1),
            A.PadIfNeeded(min_height=None, min_width=None,
                          pad_height_divisor=self.pad_divisor, pad_width_divisor=self.pad_divisor,
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                          position='top_left'),
        ], additional_targets={'L': 'image', 'H': 'image'})

    def __getitem__(self, index):
        stem = self.ids[index]
        img_path = _find_image_path(self.img_dir, stem)
        img = np.array(Image.open(img_path).convert('RGB'))
        h0, w0 = img.shape[:2]

        mask_path = os.path.join(self.mask_dir, stem + '.png')
        has_mask = os.path.exists(mask_path)
        if has_mask:
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros((h0, w0), dtype=np.uint8)

        L, H = wavelet_LH(img, self.wavelet_type, self.alpha, self.beta)

        transform = self._build_transform(h0, w0)
        augmented = transform(image=img, mask=mask, L=L, H=H)
        img_t, L_t, H_t, mask_t = augmented['image'], augmented['L'], augmented['H'], augmented['mask']

        raw_image = img_t.copy()

        normalize = A.Compose([
            A.Normalize(GLAS_MEAN, GLAS_STD),
            ToTensorV2(),
        ], additional_targets={'L': 'image', 'H': 'image'})
        normed = normalize(image=img_t, mask=mask_t, L=L_t, H=H_t)

        return {
            'image': normed['image'],
            'L': normed['L'],
            'H': normed['H'],
            'mask': normed['mask'].long(),
            'ID': stem,
            'orig_h': h0,
            'orig_w': w0,
            'raw_image': raw_image,
            'has_mask': has_mask,
        }
