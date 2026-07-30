"""
Semi-supervised XNetv2 training on the raw GLAS dataset layout
(<data_root>/train/{images,labels}, <data_root>/test/{testA,testB}/{images,labels},
<data_root>/partitions/glas_{10,20}/{labeled,unlabeled,val}.txt).

Single GPU (no DDP -- this machine only has one). See `python train_glas_semi.py -h`
for the full argument list.
"""

import argparse
import json
import os
import random
import sys
import time
from warnings import simplefilter

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.warmup_config.warmup import GradualWarmupScheduler
from dataload.dataset_glas import GlasSegDataset, resolve_split
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from tools.glas_metrics import object_dice, threshold_sweep
from tools.glas_viz import save_pseudo_grid, save_triplet

simplefilter(action="ignore", category=FutureWarning)

EVAL_ALPHA = (0.2, 0.2)
EVAL_BETA = (0.65, 0.65)


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #


def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Tee:
    """Duplicate stdout to a persistent log file (append-only, never truncated)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def build_exp_name(args):
    parts = [
        args.network,
        "portion=" + args.portion,
    ]
    if args.portion in ("10%", "20%"):
        parts.append("ss=" + str(args.split_seed))
    parts += [
        "seed=" + str(args.seed),
        "l=" + str(args.lr),
        "e=" + str(args.num_epochs),
        "s=" + str(args.step_size),
        "g=" + str(args.gamma),
        "b=" + str(args.batch_size),
        "uw=" + str(args.unsup_weight),
        "so=" + str(args.sup_only_epochs),
        "ct=" + str(args.confidence_threshold),
        "w=" + str(args.warm_up_duration),
        "loss=" + str(args.loss),
        "crop=" + str(args.crop_size),
    ]
    return "-".join(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/cvml-3/yy/Dataset/Glas")
    parser.add_argument(
        "--save_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        help="override the auto-generated experiment folder name",
    )

    parser.add_argument(
        "--portion",
        default="unsegSplit10",
        choices=["10%", "20%", "unsegSplit10", "unsegSplit20"],
    )
    parser.add_argument(
        "--split_seed",
        default=0,
        type=int,
        help="which files are labeled, only used when portion is 10%% or 20%%",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="global reproducibility seed"
    )

    parser.add_argument(
        "--val_interval",
        default=1,
        type=int,
        help="run the full validation suite every N epochs (1 = every epoch)",
    )
    parser.add_argument(
        "--val_only",
        action="store_true",
        help="skip training, only run validation/testing with --ckpt",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint to resume training from, or to evaluate with --val_only",
    )

    parser.add_argument("-n", "--network", default="XNetv2")
    parser.add_argument("-b", "--batch_size", default=2, type=int)
    parser.add_argument("-e", "--num_epochs", default=200, type=int)
    parser.add_argument("-s", "--step_size", default=50, type=int)
    parser.add_argument("-l", "--lr", default=0.01, type=float)
    parser.add_argument("-g", "--gamma", default=0.5, type=float)
    parser.add_argument("-u", "--unsup_weight", default=0.5, type=float)
    parser.add_argument(
        "--sup_only_epochs",
        default=20,
        type=int,
        help="for the first N epochs, train on labeled data only (no unsup forward/"
             "backward at all); unsup consistency training starts at epoch N+1",
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.5,
        type=float,
        help="cross-branch pseudo labels are only kept where the providing branch's "
             "softmax confidence >= this value (0.5 = no filtering, since argmax "
             "confidence is always >= 0.5 for 2 classes); pixels below threshold are "
             "excluded from the unsup loss via DiceLoss's ignore_index=-1",
    )
    parser.add_argument("--loss", default="dice")
    parser.add_argument("-w", "--warm_up_duration", default=20, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument(
        "--wd",
        default=-5,
        type=float,
        help="weight decay power (weight_decay = 5 * 10**wd)",
    )
    parser.add_argument(
        "--wavelet_type", default="haar", help="haar, db2, bior1.5, coif1, dmey"
    )
    parser.add_argument("--alpha", default=[0.0, 0.4], nargs=2, type=float)
    parser.add_argument("--beta", default=[0.5, 0.8], nargs=2, type=float)

    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--pad_divisor", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--gpu", default=0, type=int)

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def build_datasets(args):
    labeled_ids, unlabeled_ids, val_ids = resolve_split(
        args.data_root, args.portion, args.split_seed
    )

    ds_train_unsup = GlasSegDataset(
        args.data_root,
        unlabeled_ids,
        split="train",
        mode="train_aug",
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        crop_size=args.crop_size,
    )
    ds_train_sup = GlasSegDataset(
        args.data_root,
        labeled_ids,
        split="train",
        mode="train_aug",
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        crop_size=args.crop_size,
        target_len=len(ds_train_unsup),
        resample_seed=args.seed,
    )

    def eval_ds(ids, split):
        return GlasSegDataset(
            args.data_root,
            ids,
            split=split,
            mode="eval",
            wavelet_type=args.wavelet_type,
            alpha=EVAL_ALPHA,
            beta=EVAL_BETA,
            pad_divisor=args.pad_divisor,
        )

    ds_val = eval_ds(val_ids, "train")
    ds_train_labeled_eval = eval_ds(labeled_ids, "train")
    ds_train_unlabeled_eval = eval_ds(unlabeled_ids, "train")

    testA_ids = sorted(
        os.path.splitext(fn)[0]
        for fn in os.listdir(os.path.join(args.data_root, "test", "testA", "images"))
    )
    testB_ids = sorted(
        os.path.splitext(fn)[0]
        for fn in os.listdir(os.path.join(args.data_root, "test", "testB", "images"))
    )
    ds_testA = eval_ds(testA_ids, "testA")
    ds_testB = eval_ds(testB_ids, "testB")

    return {
        "labeled_ids": labeled_ids,
        "unlabeled_ids": unlabeled_ids,
        "val_ids": val_ids,
        "testA_ids": testA_ids,
        "testB_ids": testB_ids,
        "train_sup": ds_train_sup,
        "train_unsup": ds_train_unsup,
        "val": ds_val,
        "train_labeled_eval": ds_train_labeled_eval,
        "train_unlabeled_eval": ds_train_unlabeled_eval,
        "testA": ds_testA,
        "testB": ds_testB,
    }


def build_loaders(datasets, args):
    loaders = {
        "train_sup": DataLoader(
            datasets["train_sup"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "train_unsup": DataLoader(
            datasets["train_unsup"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
    }
    for key in ("val", "train_labeled_eval", "train_unlabeled_eval", "testA", "testB"):
        loaders[key] = DataLoader(
            datasets[key],
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    return loaders


# --------------------------------------------------------------------------- #
# Inference / metrics
# --------------------------------------------------------------------------- #


def confident_pseudo_label(logits, threshold):
    """Hard argmax pseudo label, but pixels where the providing branch's softmax
    confidence is below `threshold` are set to -1 (DiceLoss's ignore_index), so
    they contribute nothing to the cross-branch consistency loss. threshold=0.5 is
    equivalent to plain argmax (a 2-class softmax max is always >= 0.5).
    Returns (label, keep) where keep is the per-pixel boolean confidence mask
    (callers reduce it to a scalar fraction, or use it as-is for visualization).
    """
    probs = torch.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)
    idx = idx.long()
    keep = conf >= threshold
    idx = torch.where(keep, idx, torch.full_like(idx, -1))
    return idx, keep


def run_inference(model, loader, criterion, device, confidence_threshold=None):
    """Forward-only pass over an eval loader (batch_size=1). Also captures the L/H
    branch predictions (argmax) alongside the main branch, since the pseudo-label
    grid visualization needs all three and the forward pass already computes them.
    When confidence_threshold is given, also captures each branch's per-pixel
    confidence-filter keep-mask (keep_M/keep_L/keep_H), for visualizing which
    pixels --confidence_threshold would exclude from the unsup loss.
    """
    model.eval()
    preds_store = []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            L = batch["L"].to(device)
            H = batch["H"].to(device)
            mask = batch["mask"].to(device)
            oh = int(batch["orig_h"][0])
            ow = int(batch["orig_w"][0])

            pred_main, pred_L, pred_H = model(img, L, H)
            pred_main = pred_main[:, :, :oh, :ow]
            pred_L = pred_L[:, :, :oh, :ow]
            pred_H = pred_H[:, :, :oh, :ow]
            mask_c = mask[:, :oh, :ow]

            loss = criterion(pred_main, mask_c)
            loss_sum += loss.item()
            n += 1

            probs = torch.softmax(pred_main, dim=1)[:, 1]
            score = probs[0].detach().cpu().numpy()
            true = mask_c[0].detach().cpu().numpy().astype(np.uint8)
            raw_img = batch["raw_image"][0].numpy()[:oh, :ow]
            l_bin = (
                torch.max(pred_L, dim=1)[1][0].detach().cpu().numpy().astype(np.uint8)
            )
            h_bin = (
                torch.max(pred_H, dim=1)[1][0].detach().cpu().numpy().astype(np.uint8)
            )

            entry = {
                "ID": batch["ID"][0],
                "score": score,
                "true": true,
                "raw_image": raw_img,
                "pred_L_bin": l_bin,
                "pred_H_bin": h_bin,
            }

            if confidence_threshold is not None:
                _, keep_m = confident_pseudo_label(pred_main, confidence_threshold)
                _, keep_l = confident_pseudo_label(pred_L, confidence_threshold)
                _, keep_h = confident_pseudo_label(pred_H, confidence_threshold)
                entry["keep_M"] = keep_m[0].detach().cpu().numpy()
                entry["keep_L"] = keep_l[0].detach().cpu().numpy()
                entry["keep_H"] = keep_h[0].detach().cpu().numpy()

            preds_store.append(entry)
    return preds_store, loss_sum, n


def compute_metrics(preds_store, loss_sum, n):
    scores_cat = np.concatenate([p["score"].reshape(-1) for p in preds_store])
    trues_cat = np.concatenate([p["true"].reshape(-1) for p in preds_store])
    thr, miou, mdice, acc = threshold_sweep(scores_cat, trues_cat)

    obj_scores = []
    for p in preds_store:
        pred_bin = (p["score"] > thr).astype(np.uint8)
        p["pred_bin"] = pred_bin
        obj_scores.append(object_dice(pred_bin, p["true"]))
    dice_obj = float(np.mean(obj_scores)) if obj_scores else 0.0

    return {
        "Threshold": thr,
        "mIoU": miou,
        "mDice": mdice,
        "Acc": acc,
        "Loss": loss_sum / max(n, 1),
        "Dice_obj": dice_obj,
    }


def evaluate_split(model, loader, criterion, device, confidence_threshold=None):
    preds_store, loss_sum, n = run_inference(
        model, loader, criterion, device, confidence_threshold=confidence_threshold
    )
    return compute_metrics(preds_store, loss_sum, n), preds_store


def format_metrics(tag, m):
    return "| {:<45s} mIoU={:.4f} mDice={:.4f} Acc={:.4f} Loss={:.4f} Dice_obj={:.4f} (thr={:.2f})".format(
        tag, m["mIoU"], m["mDice"], m["Acc"], m["Loss"], m["Dice_obj"], m["Threshold"]
    )


def log_metrics_tb(writer, tag, m, step):
    writer.add_scalar("{}/mIoU".format(tag), m["mIoU"], step)
    writer.add_scalar("{}/mDice".format(tag), m["mDice"], step)
    writer.add_scalar("{}/Acc".format(tag), m["Acc"], step)
    writer.add_scalar("{}/Loss".format(tag), m["Loss"], step)
    writer.add_scalar("{}/Dice_obj".format(tag), m["Dice_obj"], step)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def _warmup_scheduler_state(scheduler_warmup):
    """GradualWarmupScheduler.state_dict() would embed the live `after_scheduler`
    object, which itself holds a reference to the optimizer -- pickling that nests
    a duplicate copy of the whole optimizer (all parameter/momentum tensors) inside
    the checkpoint, roughly doubling file size and making load_state_dict() attach
    a stale, disconnected optimizer to the scheduler. Save/restore only the plain,
    tensor-free scheduler fields instead.
    """
    return {
        "after_scheduler": scheduler_warmup.after_scheduler.state_dict(),
        "last_epoch": scheduler_warmup.last_epoch,
        "finished": scheduler_warmup.finished,
        "multiplier": scheduler_warmup.multiplier,
        "total_epoch": scheduler_warmup.total_epoch,
    }


def _load_warmup_scheduler_state(scheduler_warmup, state):
    scheduler_warmup.after_scheduler.load_state_dict(state["after_scheduler"])
    scheduler_warmup.last_epoch = state["last_epoch"]
    scheduler_warmup.finished = state["finished"]


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler_warmup,
    epoch,
    best_val_metric,
    best_test_metric,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": _warmup_scheduler_state(scheduler_warmup),
            "best_val_metric": best_val_metric,
            "best_test_metric": best_test_metric,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "args": vars(args),
        },
        path,
    )


def load_model_weights(model, path, device):
    state = torch.load(path, map_location=device)
    state_dict = (
        state["model"] if isinstance(state, dict) and "model" in state else state
    )
    model.load_state_dict(state_dict)
    return state


# --------------------------------------------------------------------------- #
# Pseudo-label dump + visualization
# --------------------------------------------------------------------------- #


def dump_pseudo_labels(preds_store, out_dir):
    """Per unlabeled image: top row = raw image | GT (train/labels); bottom row =
    M/L/H branch pseudo labels. Requires preds_store entries produced by
    compute_metrics() (for 'pred_bin', the M-branch mask) via run_inference()
    (for 'pred_L_bin' / 'pred_H_bin'). If run_inference() was called with
    confidence_threshold set, pixels excluded by --confidence_threshold (below
    threshold in *both* classes) render gray instead of black/white."""
    os.makedirs(out_dir, exist_ok=True)
    for p in preds_store:
        save_pseudo_grid(
            p["raw_image"],
            p["true"],
            (p["pred_bin"], p.get("keep_M")),
            (p["pred_L_bin"], p.get("keep_L")),
            (p["pred_H_bin"], p.get("keep_H")),
            os.path.join(out_dir, p["ID"] + ".png"),
        )


def dump_visualizations(preds_store, out_dir):
    """Test visualization, left to right: raw image | GT | prediction."""
    os.makedirs(out_dir, exist_ok=True)
    for p in preds_store:
        save_triplet(
            p["raw_image"],
            p["true"],
            p["pred_bin"],
            os.path.join(out_dir, p["ID"] + ".png"),
        )


# --------------------------------------------------------------------------- #
# Final best_val / best_test evaluation on testA / testB / testA+B
# --------------------------------------------------------------------------- #


def final_test_and_visualize(
    model, loaders, criterion, device, exp_dir, ckpt_path, tag, writer, step
):
    print("=" * 100)
    print("| Final evaluation using {}: {}".format(tag, ckpt_path))
    print("=" * 100)
    load_model_weights(model, ckpt_path, device)

    preds_A, loss_A, nA = run_inference(model, loaders["testA"], criterion, device)
    preds_B, loss_B, nB = run_inference(model, loaders["testB"], criterion, device)

    metrics_A = compute_metrics(preds_A, loss_A, nA)
    metrics_B = compute_metrics(preds_B, loss_B, nB)
    metrics_AB = compute_metrics(preds_A + preds_B, loss_A + loss_B, nA + nB)

    print(format_metrics("Test A [{}]".format(tag), metrics_A))
    print(format_metrics("Test B [{}]".format(tag), metrics_B))
    print(format_metrics("Test A+B [{}]".format(tag), metrics_AB))

    log_metrics_tb(writer, "Final_TestA/{}".format(tag), metrics_A, step)
    log_metrics_tb(writer, "Final_TestB/{}".format(tag), metrics_B, step)
    log_metrics_tb(writer, "Final_TestAB/{}".format(tag), metrics_AB, step)

    viz_dir = os.path.join(exp_dir, "visualization", tag)
    dump_visualizations(preds_A, os.path.join(viz_dir, "testA"))
    dump_visualizations(preds_B, os.path.join(viz_dir, "testB"))

    with open(os.path.join(exp_dir, "final_metrics_{}.json".format(tag)), "w") as f:
        json.dump(
            {"testA": metrics_A, "testB": metrics_B, "testA+B": metrics_AB}, f, indent=2
        )

    return metrics_A, metrics_B, metrics_AB


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()

    init_seeds(args.seed)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    exp_name = args.exp_name if args.exp_name else build_exp_name(args)
    exp_dir = os.path.join(args.save_root, "GlaS", exp_name)
    train_log_dir = os.path.join(exp_dir, "train_log")
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "pseudo_label"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualization"), exist_ok=True)

    # Tee stdout to a persistent, append-only log file *before* printing anything,
    # so the hyperparameters (and everything else) are captured in train_log too --
    # resuming must never truncate this file.
    log_filename = "val_only.log" if args.val_only else "train.log"
    log_file = open(os.path.join(train_log_dir, log_filename), "a")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 100)
    print("| Hyperparameters")
    print("=" * 100)
    print(json.dumps(vars(args), indent=2))
    print("| exp_dir: {}".format(exp_dir))

    datasets = build_datasets(args)
    loaders = build_loaders(datasets, args)

    print("=" * 100)
    print(
        "| Labeled files ({}): {}".format(
            len(datasets["labeled_ids"]), datasets["labeled_ids"]
        )
    )
    print("| Unlabeled files: {}".format(len(datasets["unlabeled_ids"])))
    print("| Val files: {}".format(len(datasets["val_ids"])))
    print("=" * 100)

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(exp_dir, "labeled_files.txt"), "w") as f:
        f.write("\n".join(datasets["labeled_ids"]))

    model = get_network(args.network, 3, 2).to(device)
    criterion = segmentation_loss(args.loss, False).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=5 * 10**args.wd,
    )
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    scheduler_warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warm_up_duration,
        after_scheduler=exp_lr_scheduler,
    )

    writer = SummaryWriter(log_dir=train_log_dir)

    start_epoch = 0
    best_val_metric = -1.0
    best_test_metric = -1.0
    loaded_ckpt_epoch = None

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            loaded_ckpt_epoch = state["epoch"]
            if not args.val_only:
                optimizer.load_state_dict(state["optimizer"])
                _load_warmup_scheduler_state(scheduler_warmup, state["scheduler"])
                start_epoch = state["epoch"] + 1
                best_val_metric = state.get("best_val_metric", -1.0)
                best_test_metric = state.get("best_test_metric", -1.0)
                if "torch_rng_state" in state:
                    torch.set_rng_state(state["torch_rng_state"].cpu())
                    torch.cuda.set_rng_state_all(
                        [s.cpu() for s in state["cuda_rng_state"]]
                    )
                    np.random.set_state(state["numpy_rng_state"])
                    random.setstate(state["python_rng_state"])
            print("| Loaded checkpoint {} (epoch {})".format(args.ckpt, state["epoch"]))
        else:
            model.load_state_dict(state)
            print("| Loaded raw weights from {}".format(args.ckpt))

    if args.val_only:
        eval_epoch = (loaded_ckpt_epoch + 1) if loaded_ckpt_epoch is not None else 0
        print("=" * 100)
        print("| Validation-only mode")
        print("=" * 100)
        metrics_val, preds_val = evaluate_split(
            model, loaders["val"], criterion, device
        )
        preds_A, loss_A, nA = run_inference(model, loaders["testA"], criterion, device)
        preds_B, loss_B, nB = run_inference(model, loaders["testB"], criterion, device)
        metrics_testAB = compute_metrics(preds_A + preds_B, loss_A + loss_B, nA + nB)
        metrics_lab, preds_lab = evaluate_split(
            model, loaders["train_labeled_eval"], criterion, device
        )
        metrics_unlab, preds_unlab = evaluate_split(
            model, loaders["train_unlabeled_eval"], criterion, device
        )

        print(format_metrics("Validation", metrics_val))
        print(format_metrics("Test (A+B)", metrics_testAB))
        print(
            format_metrics("Training dataset as validation (label part)", metrics_lab)
        )
        print(
            format_metrics(
                "Training dataset as validation (unlabel part)", metrics_unlab
            )
        )

        log_metrics_tb(writer, "ValOnly/Validation", metrics_val, eval_epoch)
        log_metrics_tb(writer, "ValOnly/Test_AB", metrics_testAB, eval_epoch)
        log_metrics_tb(writer, "ValOnly/Train_labeled", metrics_lab, eval_epoch)
        log_metrics_tb(writer, "ValOnly/Train_unlabeled", metrics_unlab, eval_epoch)

        final_test_and_visualize(
            model,
            loaders,
            criterion,
            device,
            exp_dir,
            args.ckpt,
            "val_only",
            writer,
            eval_epoch,
        )
        writer.close()
        return

    since = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_begin = time.time()
        model.train()

        sup_only = epoch < args.sup_only_epochs
        unsup_weight = 0.0 if sup_only else args.unsup_weight * (epoch + 1) / args.num_epochs

        sums = {"sup1": 0.0, "sup2": 0.0, "sup3": 0.0, "unsup": 0.0, "total": 0.0}
        kept_fraction_sum = 0.0

        if sup_only:
            # No unsup forward/backward at all during this phase -- cheaper than
            # just zeroing unsup_weight, and avoids feeding pseudo-label noise
            # into the model before it has any real signal to work from.
            n_batches = len(loaders["train_sup"])
            sup_iter = iter(loaders["train_sup"])
        else:
            n_batches = min(len(loaders["train_sup"]), len(loaders["train_unsup"]))
            sup_iter = iter(loaders["train_sup"])
            unsup_iter = iter(loaders["train_unsup"])

        for _ in range(n_batches):
            sup_batch = next(sup_iter)
            img_s = sup_batch["image"].to(device)
            L_s = sup_batch["L"].to(device)
            H_s = sup_batch["H"].to(device)
            mask_s = sup_batch["mask"].to(device)

            optimizer.zero_grad()

            # Unsup and sup forward/backward are kept separate (each backward runs
            # right after its own forward) so only one branch's activations are
            # resident in GPU memory at a time -- combining them into a single
            # graph roughly doubles peak memory at 512x512 crops and OOMs a 24GB GPU.
            if sup_only:
                loss_unsup_value = 0.0
            else:
                unsup_batch = next(unsup_iter)
                img_u = unsup_batch["image"].to(device)
                L_u = unsup_batch["L"].to(device)
                H_u = unsup_batch["H"].to(device)

                pred_u1, pred_u2, pred_u3 = model(img_u, L_u, H_u)
                max_u1, keep_u1 = confident_pseudo_label(pred_u1, args.confidence_threshold)
                max_u2, keep_u2 = confident_pseudo_label(pred_u2, args.confidence_threshold)
                max_u3, keep_u3 = confident_pseudo_label(pred_u3, args.confidence_threshold)
                loss_unsup = (
                    criterion(pred_u1, max_u2)
                    + criterion(pred_u2, max_u1)
                    + criterion(pred_u1, max_u3)
                    + criterion(pred_u3, max_u1)
                ) * unsup_weight
                loss_unsup.backward()
                loss_unsup_value = loss_unsup.item()
                kept_fraction_sum += (
                    keep_u1.float().mean().item()
                    + keep_u2.float().mean().item()
                    + keep_u3.float().mean().item()
                ) / 3

            pred_s1, pred_s2, pred_s3 = model(img_s, L_s, H_s)
            loss_sup1 = criterion(pred_s1, mask_s)
            loss_sup2 = criterion(pred_s2, mask_s)
            loss_sup3 = criterion(pred_s3, mask_s)
            loss_sup = loss_sup1 + loss_sup2 + loss_sup3
            loss_sup.backward()

            optimizer.step()

            loss_total = loss_sup.item() + loss_unsup_value
            sums["sup1"] += loss_sup1.item()
            sums["sup2"] += loss_sup2.item()
            sums["sup3"] += loss_sup3.item()
            sums["unsup"] += loss_unsup_value
            sums["total"] += loss_total

        scheduler_warmup.step()

        train_epoch_loss = {k: v / n_batches for k, v in sums.items()}
        kept_fraction = kept_fraction_sum / n_batches
        print("=" * 100)
        print(
            "| Epoch {}/{} | lr={:.6g} | Time={:.1f}s | Phase: {}".format(
                epoch + 1,
                args.num_epochs,
                optimizer.param_groups[0]["lr"],
                time.time() - epoch_begin,
                "sup-only" if sup_only else "sup+unsup",
            )
        )
        print(
            "| Train Sup Loss 1/2/3: {:.4f} / {:.4f} / {:.4f} | Unsup Loss: {:.4f} | Total: {:.4f} | "
            "Confident pseudo-label pixels: {:.1%} (thr={})".format(
                train_epoch_loss["sup1"],
                train_epoch_loss["sup2"],
                train_epoch_loss["sup3"],
                train_epoch_loss["unsup"],
                train_epoch_loss["total"],
                kept_fraction,
                args.confidence_threshold,
            )
        )

        writer.add_scalar("Train/Loss_sup1", train_epoch_loss["sup1"], epoch + 1)
        writer.add_scalar("Train/Loss_sup2", train_epoch_loss["sup2"], epoch + 1)
        writer.add_scalar("Train/Loss_sup3", train_epoch_loss["sup3"], epoch + 1)
        writer.add_scalar("Train/Loss_unsup", train_epoch_loss["unsup"], epoch + 1)
        writer.add_scalar("Train/Loss_total", train_epoch_loss["total"], epoch + 1)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.add_scalar("Train/Unsup_confident_pixel_fraction", kept_fraction, epoch + 1)

        # Pseudo-label dump every epoch, using the unlabeled-training-as-eval split.
        # confidence_threshold is passed so the dump can gray out pixels that
        # --confidence_threshold would exclude from the unsup consistency loss.
        metrics_unlab, preds_unlab = evaluate_split(
            model,
            loaders["train_unlabeled_eval"],
            criterion,
            device,
            confidence_threshold=args.confidence_threshold,
        )
        dump_pseudo_labels(
            preds_unlab,
            os.path.join(exp_dir, "pseudo_label", "epoch{}".format(epoch + 1)),
        )

        is_val_epoch = ((epoch + 1) % args.val_interval == 0) or (
            epoch + 1 == args.num_epochs
        )
        if is_val_epoch:
            metrics_val, _ = evaluate_split(model, loaders["val"], criterion, device)
            preds_A, loss_A, nA = run_inference(
                model, loaders["testA"], criterion, device
            )
            preds_B, loss_B, nB = run_inference(
                model, loaders["testB"], criterion, device
            )
            metrics_test = compute_metrics(preds_A + preds_B, loss_A + loss_B, nA + nB)
            metrics_lab, _ = evaluate_split(
                model, loaders["train_labeled_eval"], criterion, device
            )

            print("-" * 100)
            print(format_metrics("1. Validation", metrics_val))
            print(format_metrics("2. Test (A+B)", metrics_test))
            print(
                format_metrics(
                    "3. Training dataset as validation (label part)", metrics_lab
                )
            )
            print(
                format_metrics(
                    "4. Training dataset as validation (unlabel part)", metrics_unlab
                )
            )

            log_metrics_tb(writer, "Validation", metrics_val, epoch + 1)
            log_metrics_tb(writer, "Test_AB", metrics_test, epoch + 1)
            log_metrics_tb(writer, "Train_labeled_as_val", metrics_lab, epoch + 1)
            log_metrics_tb(writer, "Train_unlabeled_as_val", metrics_unlab, epoch + 1)

            combined_val = metrics_val["mIoU"] + metrics_val["mDice"]
            combined_test = metrics_test["mIoU"] + metrics_test["mDice"]

            if combined_val > best_val_metric:
                best_val_metric = combined_val
                save_checkpoint(
                    os.path.join(exp_dir, "best_val.pth"),
                    model,
                    optimizer,
                    scheduler_warmup,
                    epoch,
                    best_val_metric,
                    best_test_metric,
                    args,
                )
                print(
                    "| >>> New best_val.pth (mIoU+mDice={:.4f}) at epoch {}".format(
                        combined_val, epoch + 1
                    )
                )

            if combined_test > best_test_metric:
                best_test_metric = combined_test
                save_checkpoint(
                    os.path.join(exp_dir, "best_test.pth"),
                    model,
                    optimizer,
                    scheduler_warmup,
                    epoch,
                    best_val_metric,
                    best_test_metric,
                    args,
                )
                print(
                    "| >>> New best_test.pth (mIoU+mDice={:.4f}) at epoch {}".format(
                        combined_test, epoch + 1
                    )
                )

        save_checkpoint(
            os.path.join(exp_dir, "last.pth"),
            model,
            optimizer,
            scheduler_warmup,
            epoch,
            best_val_metric,
            best_test_metric,
            args,
        )

    time_elapsed = time.time() - since
    h, rem = divmod(time_elapsed, 3600)
    m, s = divmod(rem, 60)
    print("=" * 100)
    print("| Training completed in {:.0f}h {:.0f}m {:.0f}s".format(h, m, s))
    print(
        "| Best Val (mIoU+mDice): {:.4f} | Best Test (mIoU+mDice): {:.4f}".format(
            best_val_metric, best_test_metric
        )
    )
    print("=" * 100)

    for tag, ckpt_name in (
        ("best_val", "best_val.pth"),
        ("best_test", "best_test.pth"),
    ):
        ckpt_path = os.path.join(exp_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            final_test_and_visualize(
                model,
                loaders,
                criterion,
                device,
                exp_dir,
                ckpt_path,
                tag,
                writer,
                args.num_epochs,
            )
        else:
            print(
                "| Skipping final evaluation for {}: checkpoint not found".format(tag)
            )

    writer.close()


if __name__ == "__main__":
    main()
