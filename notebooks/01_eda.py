"""
Exploratory Data Analysis for ChestMNIST.

Produces:
  - label distribution bar chart
  - label co-occurrence heatmap
  - sample image grid
  - a few exported PNG radiographs for testing the Streamlit demonstrator

Run:
    python notebooks/01_eda.py
Outputs are written to notebooks/eda_outputs/.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from medmnist import ChestMNIST
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)


def main():
    print("Loading ChestMNIST …")
    train = ChestMNIST(split="train", download=True, root=config.DATA_DIR, size=config.IMAGE_SIZE)
    test  = ChestMNIST(split="test",  download=True, root=config.DATA_DIR, size=config.IMAGE_SIZE)

    labels = train.labels                       # (N, 14)
    print(f"Train samples: {len(train)} | Test samples: {len(test)}")
    print(f"Label matrix shape: {labels.shape}")

    # ── 1. Label distribution ─────────────────────────────────────────────
    counts = labels.sum(axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(config.CLASS_NAMES, counts, color="steelblue")
    ax.set(title="Positive samples per pathology (train)", ylabel="Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "label_distribution.png"), dpi=150)
    print("Saved label_distribution.png")

    # Imbalance ratios
    total = len(labels)
    print("\nClass prevalence:")
    for name, c in zip(config.CLASS_NAMES, counts):
        print(f"  {name:20s} {c:6d}  ({100*c/total:.2f}%)")

    no_finding = (labels.sum(axis=1) == 0).sum()
    print(f"\n'No finding' (all-zero) samples: {no_finding} ({100*no_finding/total:.1f}%)")

    # ── 2. Co-occurrence heatmap ──────────────────────────────────────────
    cooc = labels.T @ labels                    # (14, 14)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cooc, xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                cmap="viridis", annot=False, ax=ax)
    ax.set(title="Label co-occurrence matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cooccurrence.png"), dpi=150)
    print("Saved cooccurrence.png")

    # ── 3. Sample grid ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for i, ax in enumerate(axes.flat):
        img, lbl = train[i]
        ax.imshow(np.array(img), cmap="gray")
        active = [config.CLASS_NAMES[j] for j in range(14) if lbl[j] == 1]
        ax.set_title(", ".join(active) if active else "No finding", fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sample_grid.png"), dpi=150)
    print("Saved sample_grid.png")

    # ── 4. Export a few PNGs for the Streamlit demo ───────────────────────
    print("\nExporting sample radiographs for the demonstrator …")
    for i in range(10):
        img, lbl = test[i]
        active = [config.CLASS_NAMES[j] for j in range(14) if lbl[j] == 1]
        tag = "_".join(active) if active else "no_finding"
        img.convert("L").save(os.path.join(OUT_DIR, "samples", f"test_{i:02d}_{tag}.png"))
    print(f"Done. See {OUT_DIR}")


if __name__ == "__main__":
    main()
