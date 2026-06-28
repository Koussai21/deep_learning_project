"""
Train and compare three deep architectures for multi-label chest pathology
classification on ChestMNIST, with full MLflow tracking.

Usage:
    python -m training.train_classification --model cnn_scratch
    python -m training.train_classification --model densenet121
    python -m training.train_classification --model vit
    python -m training.train_classification --model hybrid

Each run logs hyperparameters, per-epoch metrics, ROC curves, the best
checkpoint and the final model to MLflow.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset import get_chest_mnist_loaders, get_class_weights
from models.cnn_scratch import CNNFromScratch
from models.transfer_learning import TransferModel
from models.vit import ViTClassifier, HybridCNNViT
from evaluation.metrics import compute_metrics, plot_roc_curves
from mlflow_utils.tracking import setup_mlflow, MLflowRun, log_epoch
from training.utils import set_seed, get_device, EarlyStopping, count_parameters


def build_model(name: str, image_size: int = config.IMAGE_SIZE) -> nn.Module:
    if name == "cnn_scratch":
        return CNNFromScratch()
    if name in ("densenet121", "resnet50", "efficientnet_b0"):
        return TransferModel(backbone_name=name)
    if name == "vit":
        return ViTClassifier(img_size=image_size)   # must match dataloader size
    if name == "hybrid":
        return HybridCNNViT()
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool = False):
    model.eval()
    losses, all_logits, all_targets = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=use_amp):
            logits = model(images)
            losses.append(criterion(logits, labels.float()).item())
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    y_logits = np.concatenate(all_logits)
    y_true   = np.concatenate(all_targets)
    metrics  = compute_metrics(y_true, y_logits)
    return float(np.mean(losses)), metrics, y_true, y_logits


def train(args):
    set_seed()
    device = get_device()
    use_amp = getattr(args, "use_amp", True) and torch.cuda.is_available()
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    grad_clip = getattr(args, "grad_clip", 1.0)
    patience  = getattr(args, "patience", config.PATIENCE)
    num_workers = getattr(args, "num_workers", 4)
    print(f"Device: {device}  |  AMP: {use_amp}  |  label_smoothing: {label_smoothing}")
    if use_amp:
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────
    loaders = get_chest_mnist_loaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # Class imbalance → weighted BCE (capped at 10 to prevent "predict all positive" collapse)
    print("Computing class weights …")
    pos_weight = get_class_weights(loaders["train"]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args.model, image_size=args.image_size).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} | trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = GradScaler(enabled=use_amp)

    ckpt_path = os.path.join(config.MODELS_DIR, f"classifier_{args.model}.pt")
    stopper   = EarlyStopping(patience=patience, mode="max", checkpoint_path=ckpt_path)

    # ── MLflow ────────────────────────────────────────────────────────────
    setup_mlflow(config.EXPERIMENT_CLASSIFICATION)
    params = {
        "model": args.model, "image_size": args.image_size,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": config.WEIGHT_DECAY, "epochs": args.epochs,
        "optimizer": "AdamW", "scheduler": "CosineAnnealing",
        "loss": "BCEWithLogitsLoss(pos_weight<=10)", "n_params": n_params,
        "use_amp": use_amp, "label_smoothing": label_smoothing,
        "grad_clip": grad_clip, "patience": patience,
    }

    with MLflowRun(run_name=args.model, params=params) as run:
        run.log_config(params)
        best_auc = 0.0

        for epoch in range(args.epochs):
            # ── Train ─────────────────────────────────────────────────────
            model.train()
            t0, train_losses = time.time(), []
            pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{args.epochs}")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                # Label smoothing: 1→(1-s), 0→s/2
                targets = labels.float()
                if label_smoothing > 0:
                    targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
                with autocast(enabled=use_amp):
                    logits = model(images)
                    loss   = criterion(logits, targets)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            train_loss = float(np.mean(train_losses))

            # ── Validate ──────────────────────────────────────────────────
            val_loss, val_metrics, _, _ = evaluate(
                model, loaders["val"], criterion, device, use_amp=use_amp,
            )
            epoch_time = time.time() - t0
            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_auc={val_metrics['auc_macro']:.4f}  ({epoch_time:.0f}s)")

            log_epoch(epoch, train_loss, val_loss, val_metrics)
            run.log_metrics({"epoch_time_s": epoch_time, "lr": scheduler.get_last_lr()[0]}, step=epoch)

            best_auc = max(best_auc, val_metrics["auc_macro"])
            if stopper(val_metrics["auc_macro"], model):
                print("Early stopping triggered.")
                break

        # ── Test with best checkpoint ─────────────────────────────────────
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_loss, test_metrics, y_true, y_logits = evaluate(
            model, loaders["test"], criterion, device
        )
        print(f"\nTest AUC (macro): {test_metrics['auc_macro']:.4f}")
        print(f"Test mAP (macro): {test_metrics['map_macro']:.4f}")

        run.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        run.log_figure(plot_roc_curves(y_true, y_logits), f"roc_{args.model}.png")
        run.log_best_checkpoint(ckpt_path)
        run.log_model(model, "model")

    print(f"\nDone. Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="densenet121",
                        choices=["cnn_scratch", "densenet121", "resnet50",
                                 "efficientnet_b0", "vit", "hybrid"])
    parser.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    parser.add_argument("--image_size", type=int,   default=config.IMAGE_SIZE)
    parser.add_argument("--patience",   type=int,   default=config.PATIENCE)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--no_amp",     action="store_true")
    args = parser.parse_args()
    args.use_amp = not args.no_amp
    train(args)
