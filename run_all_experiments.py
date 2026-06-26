"""
Convenience script to run the full experimental campaign required by the spec.

It launches, in sequence:
  - the three supervised architectures (CNN from scratch, DenseNet121, ViT, hybrid)
  - the two anomaly detectors (AE, VAE)
  - the multimodal comparison (image / text / fusion) if OpenI is present

All results land in the same MLflow tracking store (./mlruns).
Reduce --epochs for a quick smoke test.

Run:
    python run_all_experiments.py --epochs 5
"""
import os
import sys
import argparse
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list):
    print("\n" + "=" * 70)
    print(">>> " + " ".join(cmd))
    print("=" * 70)
    subprocess.run([sys.executable, "-m", *cmd], cwd=BASE, check=False)


def main(args):
    # ── Supervised classification ─────────────────────────────────────────
    for model in ["cnn_scratch", "densenet121", "vit", "hybrid"]:
        run(["training.train_classification", "--model", model,
             "--epochs", str(args.epochs)])

    # ── Anomaly detection ─────────────────────────────────────────────────
    for model in ["ae", "vae"]:
        run(["training.train_anomaly", "--model", model,
             "--epochs", str(args.epochs)])

    # ── Multimodal (skipped automatically if OpenI not downloaded) ─────────
    openi = os.path.join(BASE, "data", "raw", "openi", "ecgen-radiology")
    if os.path.isdir(openi) and os.listdir(openi):
        for mode in [["--mode", "image"], ["--mode", "text"],
                     ["--mode", "fusion", "--fusion", "late"]]:
            run(["training.train_multimodal", *mode, "--epochs", str(args.epochs)])
    else:
        print("\n[skip] OpenI dataset not found — multimodal experiments skipped.")
        print("       See data/multimodal_dataset.py for download instructions.")

    print("\nAll experiments finished. Launch `mlflow ui` to inspect runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    main(parser.parse_args())
