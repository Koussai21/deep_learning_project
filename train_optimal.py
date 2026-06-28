import os
import sys
import argparse
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))

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
    print(f"\n  {model_name} done in {elapsed/60:.1f} min  [{status}]")
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

    print("\nChestMNIST - Optimal Training Campaign")
    print(f"  Classifiers : {', '.join(clf_to_run) or '(none)'}")
    print(f"  Anomaly     : {', '.join(anom_to_run) or '(none)'}")
    print(f"  AMP         : enabled on CUDA")
    print(f"  pos_weight  : capped at 10.0")
    print(f"  Augment     : RandomCrop + RandomErasing (classifiers only)")

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
