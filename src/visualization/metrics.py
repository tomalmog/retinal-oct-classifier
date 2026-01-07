"""Visualization utilities for metrics and results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, classification_report, roc_curve


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = confusion_matrix.astype("float") / confusion_matrix.sum(
            axis=1, keepdims=True
        )
        fmt = ".2%"
        title = "Normalized Confusion Matrix"
    else:
        cm = confusion_matrix
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot ROC curves for multi-class classification.

    Args:
        labels: True labels
        probabilities: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_classes = len(class_names)

    # Binarize labels for ROC
    labels_bin = np.eye(n_classes)[labels]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})"
        )

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves - Multi-class Classification", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curves to {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss plot
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], "b-", label="Training Loss", lw=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", lw=2)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if "train_acc" in history:
        axes[1].plot(
            epochs, history["train_acc"], "b-", label="Training Accuracy", lw=2
        )
    if "val_acc" in history:
        axes[1].plot(
            epochs, history["val_acc"], "r-", label="Validation Accuracy", lw=2
        )

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(
        "Training and Validation Accuracy", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history to {save_path}")

    return fig


def plot_class_distribution(
    distribution: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot class distribution bar chart.

    Args:
        distribution: Dictionary mapping class names to counts
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(distribution.keys())
    counts = list(distribution.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    bars = ax.bar(classes, counts, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution in Dataset", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved class distribution to {save_path}")

    return fig


def generate_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> str:
    """Generate and optionally save classification report.

    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names
        save_path: Path to save the report

    Returns:
        Classification report string
    """
    report = classification_report(
        labels, predictions, target_names=class_names, digits=4
    )

    if save_path:
        with open(save_path, "w") as f:
            f.write("Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"Saved classification report to {save_path}")

    return report
