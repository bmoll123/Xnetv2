import numpy as np
from PIL import Image


def _mask_to_rgb(mask_bin):
    out = np.zeros((mask_bin.shape[0], mask_bin.shape[1], 3), dtype=np.uint8)
    out[mask_bin.astype(bool)] = (255, 255, 255)
    return out


def _mask_to_rgb_conf(mask_bin, keep_bool):
    """Like _mask_to_rgb, but where keep_bool is False (excluded by
    --confidence_threshold, i.e. neither class reached the threshold) the pixel
    renders gray instead of black/white, regardless of which class argmax favored.
    keep_bool=None means no confidence filtering was applied -> plain black/white.
    """
    out = _mask_to_rgb(mask_bin)
    if keep_bool is not None:
        out[~keep_bool.astype(bool)] = (128, 128, 128)
    return out


def _hpack(cells, gap=4, bg=255):
    h, w = cells[0].shape[:2]
    n = len(cells)
    canvas = np.full((h, w * n + gap * (n - 1), 3), bg, dtype=np.uint8)
    for i, c in enumerate(cells):
        x0 = i * (w + gap)
        canvas[:, x0:x0 + w] = c
    return canvas


def save_triplet(image_rgb_uint8, gt_bin, pred_bin, save_path):
    """Test visualization, left to right: raw image | GT | prediction."""
    canvas = _hpack([image_rgb_uint8, _mask_to_rgb(gt_bin), _mask_to_rgb(pred_bin)])
    Image.fromarray(canvas).save(save_path)


def save_pseudo_grid(image_rgb_uint8, gt_bin, m, l, h, save_path, gap=4, bg=255):
    """Pseudo-label visualization for one unlabeled image / epoch:
    top row:    raw image | GT (train/labels)
    bottom row: M-branch pseudo label | L-branch pseudo label | H-branch pseudo label

    m/l/h are each (bin, keep) tuples. keep is the per-pixel confidence-filter
    mask for that branch (or None to skip filtering); excluded pixels render gray
    -- black/white otherwise (background/foreground).
    """
    m_bin, m_keep = m
    l_bin, l_keep = l
    h_bin, h_keep = h

    h_img, w_img = image_rgb_uint8.shape[:2]
    top = _hpack([image_rgb_uint8, _mask_to_rgb(gt_bin)], gap=gap, bg=bg)
    bottom = _hpack([
        _mask_to_rgb_conf(m_bin, m_keep),
        _mask_to_rgb_conf(l_bin, l_keep),
        _mask_to_rgb_conf(h_bin, h_keep),
    ], gap=gap, bg=bg)

    width = max(top.shape[1], bottom.shape[1])
    canvas = np.full((h_img * 2 + gap, width, 3), bg, dtype=np.uint8)
    canvas[:h_img, :top.shape[1]] = top
    canvas[h_img + gap:, :bottom.shape[1]] = bottom

    Image.fromarray(canvas).save(save_path)
