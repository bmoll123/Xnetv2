import numpy as np
from scipy import ndimage

_STRUCT8 = np.ones((3, 3), dtype=int)


def threshold_sweep(scores, trues, interval=0.02):
    """Pixel-level mIoU/mDice/Acc, sweeping the foreground threshold like the
    upstream evaluate() in config/eval_config/eval.py, plus pixel accuracy at the
    threshold that maximizes IoU.
    scores/trues: 1-D numpy arrays (flattened foreground probability / binary GT).
    """
    thresholds = np.arange(0, 0.9, interval)
    jaccard = np.zeros(len(thresholds))
    dice = np.zeros(len(thresholds))
    acc = np.zeros(len(thresholds))
    n = trues.size

    for i, t in enumerate(thresholds):
        pred = (scores > t).astype(np.int8)
        sum_area = pred + trues
        tp = float(np.sum(sum_area == 2))
        union = np.sum(sum_area == 1)
        tn = float(np.sum(sum_area == 0))
        jaccard[i] = tp / (union + tp) if (union + tp) > 0 else 0.0
        dice[i] = 2 * tp / (union + 2 * tp) if (union + 2 * tp) > 0 else 0.0
        acc[i] = (tp + tn) / n

    idx = int(np.argmax(jaccard))
    return float(thresholds[idx]), float(jaccard[idx]), float(dice[idx]), float(acc[idx])


def _one_direction_dice(src_lbl, n_src, dst_lbl):
    if n_src == 0:
        return 0.0
    total_area = float(np.sum(src_lbl > 0))
    if total_area == 0:
        return 0.0

    acc = 0.0
    for i in range(1, n_src + 1):
        Si = src_lbl == i
        area_i = float(Si.sum())
        overlap_labels = dst_lbl[Si]
        overlap_labels = overlap_labels[overlap_labels > 0]
        if overlap_labels.size == 0:
            dice_i = 0.0
        else:
            counts = np.bincount(overlap_labels)
            best = int(np.argmax(counts))
            Gi = dst_lbl == best
            inter = float(np.sum(Si & Gi))
            denom = area_i + float(Gi.sum())
            dice_i = 2 * inter / denom if denom > 0 else 0.0
        acc += (area_i / total_area) * dice_i
    return acc


def object_dice(pred_bin, gt_bin):
    """GlaS-challenge object-level Dice (Sirinukunwattana et al., 2017):
    Dice_obj = 0.5 * [ sum_i (|S_i|/sum|S|) * Dice(S_i, G_match) +
                        sum_j (|G_j|/sum|G|) * Dice(G_j, S_match) ]
    using 8-connected components and best-overlap matching in each direction.
    """
    pred_lbl, n_pred = ndimage.label(pred_bin, structure=_STRUCT8)
    gt_lbl, n_gt = ndimage.label(gt_bin, structure=_STRUCT8)

    sum1 = _one_direction_dice(pred_lbl, n_pred, gt_lbl)
    sum2 = _one_direction_dice(gt_lbl, n_gt, pred_lbl)

    if n_pred == 0 and n_gt == 0:
        return 1.0
    return 0.5 * (sum1 + sum2)
