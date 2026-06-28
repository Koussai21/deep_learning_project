"""
Optimal training campaign for all models on ChestMNIST.

What this script does vs the generic run_all_experiments.py:
  - Per-model hyperparameters tuned for ChestMNIST (image size, LR, epochs, patience)
  - Automatic Mixed Precision (AMP) on CUDA → ~2× faster on T4
  - pos_weight capped at 10.0 (fixes the "predict everything positive" collapse)
  - Gradient clipping (max_norm=1.0) for training stability
  - Label smoothing 0.05 to prevent overconfident predictions
  - 4 DataLoader workers + persistent_workers
  - Stronger data augmentation for classifiers (RandomCrop, RandomErasing)
  - AE/VAE trained WITHOUT RandomErasing (would corrupt the reconstruction target)

Estimated total time on Lightning AI T4 (16 GB):
  cnn_scratch  128px × 50 epochs  ≈ 20-25 min
  densenet121  224px × 30 epochs  ≈ 45-55 min
  vit          128px × 30 epochs  ≈ 35-45 min
  hybrid       128px × 30 epochs  ≈ 40-50 min
  ae           64px  × 30 epochs  ≈  2-3  min
  vae          64px  × 30 epochs  ≈  3-4  min
  ─────────────────────────────────────────────
  TOTAL                           ≈ 2h30-3h10

Usage:
    cd deep_learning_project
    python train_optimal.py                    # train all 6 models
    python train_optimal.py --model vit        # one classifier only
    python train_optimal.py --model ae         # one anomaly model only
    python train_optimal.py --dry_run          # preview commands without running
"""
import os
import sys
import argparse
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Per-model optimal hyperparameters ────────────────────────────────────────
#
# cnn_scratch:
#   - 128px: 4 MaxPool → 8×8 feature map, enough for a scratch CNN
#   - lr=3e-4: higher LR needed when training from random init
#   - epochs=50 + patience=12: needs more time to converge from scratch
#   - batch=64: larger batch stabilises gradients for shallow net
#
# densenet121:
#   - 224px: native pretrained resolution, avoids positional-embedding distortion
#   - lr=1e-4: CheXNet paper value, safe for fine-tuning ImageNet weights
#   - epochs=30 + patience=8: pretrained weights converge fast
#   - batch=32: limited by 224px image memory on T4
#
# vit (ViT-Small/patch16, ImageNet-21k pretrained):
#   - 128px: 64 tokens vs 196 at 224px → attention 9× cheaper, similar accuracy
#   - IMPORTANT: img_size is fixed at train time; streamlit uses 128px accordingly
#   - lr=5e-5: transformers are sensitive to LR, lower is safer
#
# hybrid (ResNet50-layer2 + 2-layer Transformer):
#   - 128px: stride-8 CNN → 16×16 = 256 tokens before transformer
#   - lr=8e-5: backbone pretrained, head random init → intermediate LR
#
# ae / vae:
#   - 64px: enough for reconstruction, 4× faster than 128px
#   - augment_train=False: the AE must reconstruct its input exactly;
#     RandomErasing would corrupt pixels in both input and target
#   - epochs=30 + patience=8

CLASSIFIER_CONFIGS = {
    "cnn_scratch": {
        "image_size":      128,
        "epochs":          50,
        "lr":              3e-4,
        "batch_size":      64,
        "patience":        12,
        "label_smoothing": 0.05,
        "grad_clip":       1.0,
    },
    "densenet121": {
        "image_size":      224,
        "epochs":          30,
        "lr":              1e-4,
        "batch_size":      32,
        "patience":        8,
        "label_smoothing": 0.05,
        "grad_clip":       1.0,
    },
    "vit": {
        "image_size":      128,
        "epochs":          30,
        "lr":              5e-5,
        "batch_size":      32,
        "patience":        8,
        "label_smoothing": 0.05,
        "grad_clip":       1.0,
    },
    "hybrid": {
        "image_size":      128,
        "epochs":          30,
        "lr":              8e-5,
        "batch_size":      32,
        "patience":        8,
        "label_smoothing": 0.05,
        "grad_clip":       1.0,
    },
}

ANOMALY_CONFIGS = {
    "ae": {
        "image_size":  64,
        "epochs":      30,
        "lr":          1e-4,
        "batch_size":  64,
    },
    "vae": {
        "image_size":  64,
        "epochs":      30,
        "lr":          1e-4,
        "batch_size":  64,
    },
}

ALL_MODELS = list(CLASSIFIER_CONFIGS.keys()) + list(ANOMALY_CONFIGS.keys())


def run_classifier(model_name: str, cfg: dict, dry_run: bool = False) -> float:
    cmd = [
        sys.executable, "-m", "training.train_classification",
        "--model",           model_name,
        "--image_size",      str(cfg["image_size"]),
        "--epochs",          str(cfg["epochs"]),
        "--lr",              str(cfg["lr"]),
        "--batch_size",      str(cfg["batch_size"]),
        "--patience",        str(cfg["patience"]),
        "--label_smoothing", str(cfg["label_smoothing"]),
        "--grad_clip",       str(cfg["grad_clip"]),
        "--num_workers",     "4",
    ]
    return _run(model_name, cmd, cfg, dry_run)


def run_anomaly(model_name: str, cfg: dict, dry_run: bool = False) -> float:
    cmd = [
        sys.executable, "-m", "training.train_anomaly",
        "--model",      model_name,
        "--image_size", str(cfg["image_size"]),
        "--epochs",     str(cfg["epochs"]),
        "--lr",         str(cfg["lr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--num_workers","4",
    ]
    return _run(model_name, cmd, cfg, dry_run)


def _run(model_name: str, cmd: list, cfg: dict, dry_run: bool) -> float:
    print("\n" + "=" * 72)
    print(f"  MODEL : {model_name}")
    print(f"  cfg   : image={cfg['image_size']}px  epochs={cfg['epochs']}"
          f"  lr={cfg['lr']}  bs={cfg['batch_size']}")
    print(f"  cmd   : {' '.join(cmd[2:])}")
    print("=" * 72)

    if dry_run:
        print("[dry-run] skipping execution")
        return 0.0

    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE, check=False)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  → {model_name} done in {elapsed/60:.1f} min  [{status}]")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Optimal training for all ChestMNIST models"
    )
    parser.add_argument(
        "--model",
        choices=ALL_MODELS + ["all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    if args.model == "all":
        clf_to_run  = list(CLASSIFIER_CONFIGS.keys())
        anom_to_run = list(ANOMALY_CONFIGS.keys())
    elif args.model in CLASSIFIER_CONFIGS:
        clf_to_run, anom_to_run = [args.model], []
    else:
        clf_to_run, anom_to_run = [], [args.model]

    print("\n  ChestMNIST — Optimal Training Campaign")
    print(f"    Classifiers : {', '.join(clf_to_run) or '(none)'}")
    print(f"    Anomaly     : {', '.join(anom_to_run) or '(none)'}")
    print(f"    AMP         : enabled on CUDA")
    print(f"    pos_weight  : capped at 10.0  (fixes predict-all-positive collapse)")
    print(f"    Augment     : RandomCrop + RandomErasing (classifiers only)")

    total_start = time.time()
    timings = {}

    for name in clf_to_run:
        timings[name] = run_classifier(name, CLASSIFIER_CONFIGS[name], args.dry_run)

    for name in anom_to_run:
        timings[name] = run_anomaly(name, ANOMALY_CONFIGS[name], args.dry_run)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for name, t in timings.items():
        print(f"  {name:<15} : {t/60:.1f} min")
    print(f"  {'TOTAL':<15} : {total_elapsed/60:.1f} min")
    print("\nClassifier checkpoints : saved_models/classifier_<model>.pt")
    print("Anomaly checkpoints    : saved_models/anomaly_<model>.pt")
    print("MLflow runs            : mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
