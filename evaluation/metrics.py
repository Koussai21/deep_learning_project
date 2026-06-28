import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_metrics(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    threshold: float = 0.5,
    class_names: list = config.CLASS_NAMES,
) -> dict:
    y_prob = 1 / (1 + np.exp(-y_pred_logits))
    y_bin  = (y_prob >= threshold).astype(int)

    metrics = {}

    try:
        metrics["auc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["auc_micro"] = roc_auc_score(y_true, y_prob, average="micro")
        per_class_auc = roc_auc_score(y_true, y_prob, average=None)
        for i, name in enumerate(class_names):
            metrics[f"auc_{name}"] = per_class_auc[i]
    except ValueError:
        metrics["auc_macro"] = float("nan")
        metrics["auc_micro"] = float("nan")

    try:
        metrics["map_macro"] = average_precision_score(y_true, y_prob, average="macro")
        metrics["map_micro"] = average_precision_score(y_true, y_prob, average="micro")
    except ValueError:
        metrics["map_macro"] = float("nan")

    metrics["f1_macro"]    = f1_score(y_true, y_bin, average="macro",    zero_division=0)
    metrics["f1_micro"]    = f1_score(y_true, y_bin, average="micro",    zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_bin, average="weighted", zero_division=0)
    metrics["hamming_loss"] = hamming_loss(y_true, y_bin)

    return metrics


def print_classification_report(y_true, y_pred_logits, threshold=0.5):
    y_prob = 1 / (1 + np.exp(-y_pred_logits))
    y_bin  = (y_prob >= threshold).astype(int)
    print(classification_report(y_true, y_bin,
                                target_names=config.CLASS_NAMES, zero_division=0))


def plot_roc_curves(y_true, y_pred_logits, save_path: str = None):
    from sklearn.metrics import roc_curve, auc
    y_prob = 1 / (1 + np.exp(-y_pred_logits))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(config.CLASS_NAMES):
        if y_true[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC curves per pathology")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_label_distribution(y_true, save_path: str = None):
    counts = y_true.sum(axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(config.CLASS_NAMES, counts)
    ax.set(title="Positive label distribution per class",
           ylabel="Count", xlabel="Pathology")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_reconstruction(original, reconstructed, n: int = 8, save_path: str = None):
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        for row, img in enumerate([original, reconstructed]):
            arr = img[i].cpu().numpy().transpose(1, 2, 0)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            axes[row, i].imshow(arr, cmap="gray")
            axes[row, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Reconstructed", loc="left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
