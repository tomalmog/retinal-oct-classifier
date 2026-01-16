#!/usr/bin/env python3
"""Generate meaningful visualizations for high-accuracy models."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.manifold import TSNE

from src.data import get_dataloaders, get_val_transforms
from src.data.dataset import OCTDataset
from src.training import OCTLightningModule
from src.utils import load_config
from src.visualization import GradCAMVisualizer


def generate_all_visualizations(
    checkpoint_path: str, config_path: str, output_dir: str
):
    """Generate comprehensive visualizations."""
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = config.get("class_names", ["CNV", "DME", "DRUSEN", "NORMAL"])

    # Load model
    print("Loading model...")
    model = OCTLightningModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get test data
    print("Loading test data...")
    data_config = config.get("data", {})
    test_dataset = OCTDataset(
        data_dir=data_config.get("data_dir", "data/OCT2017"),
        split="test",
        transform=get_val_transforms(config),
    )
    _, _, test_loader = get_dataloaders(config)

    # Collect predictions with confidence
    print("Running inference...")
    all_labels = []
    all_preds = []
    all_probs = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(confs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_confidences = np.array(all_confidences)

    # Find errors and hard examples
    errors_mask = all_preds != all_labels
    error_indices = np.where(errors_mask)[0]

    correct_mask = ~errors_mask
    correct_confidences = all_confidences[correct_mask]
    hard_correct_indices = np.where(correct_mask)[0][
        np.argsort(correct_confidences)[:20]
    ]

    print(f"Found {len(error_indices)} errors")

    # 1. Error Analysis Gallery
    print("Generating error analysis...")
    plot_error_analysis(
        test_dataset,
        error_indices,
        all_labels,
        all_preds,
        all_confidences,
        class_names,
        output_dir,
    )

    # 2. Confidence Distribution
    print("Generating confidence distribution...")
    plot_confidence_distribution(
        all_confidences, all_labels, all_preds, class_names, output_dir
    )

    # 3. Hard Examples (low confidence but correct)
    print("Generating hard examples...")
    plot_hard_examples(
        test_dataset,
        hard_correct_indices,
        all_labels,
        all_preds,
        all_confidences,
        class_names,
        output_dir,
    )

    # 4. Grad-CAM on errors (if any)
    if len(error_indices) > 0:
        print("Generating Grad-CAM on errors...")
        plot_gradcam_errors(
            model,
            test_dataset,
            error_indices,
            all_labels,
            all_preds,
            all_confidences,
            class_names,
            device,
            output_dir,
        )

    # 5. Feature space visualization
    print("Generating feature space visualization...")
    plot_feature_space(model, test_loader, all_labels, class_names, device, output_dir)

    # 6. Standard Grad-CAM samples (keep this one)
    print("Generating Grad-CAM samples...")
    plot_gradcam_samples(model, test_dataset, class_names, device, output_dir)

    print(f"\nAll visualizations saved to {output_dir}")


def plot_error_analysis(
    dataset, error_indices, labels, preds, confidences, class_names, output_dir
):
    """Show all misclassified images."""
    n_errors = len(error_indices)
    if n_errors == 0:
        print("No errors to display!")
        return

    cols = min(4, n_errors)
    rows = (n_errors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_errors == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Misclassified Images ({n_errors} total)", fontsize=14, fontweight="bold"
    )

    for idx, error_idx in enumerate(error_indices):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        img_path, _ = dataset.samples[error_idx]
        img = Image.open(img_path).convert("RGB")

        true_label = class_names[labels[error_idx]]
        pred_label = class_names[preds[error_idx]]
        conf = confidences[error_idx]

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} ({conf:.1%})",
            fontsize=10,
            color="red",
        )
        ax.axis("off")

    # Hide empty subplots
    for idx in range(n_errors, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(confidences, labels, preds, class_names, output_dir):
    """Plot confidence distributions for correct vs incorrect predictions."""
    correct_mask = preds == labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall distribution
    ax = axes[0]
    ax.hist(
        confidences[correct_mask],
        bins=30,
        alpha=0.7,
        label="Correct",
        color="green",
        density=True,
    )
    if (~correct_mask).sum() > 0:
        ax.hist(
            confidences[~correct_mask],
            bins=30,
            alpha=0.7,
            label="Incorrect",
            color="red",
            density=True,
        )
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Prediction Confidence Distribution", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 1)

    # Per-class confidence boxplot
    ax = axes[1]
    data = [confidences[labels == i] for i in range(len(class_names))]
    bp = ax.boxplot(data, labels=class_names, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title("Confidence by Class", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(
        output_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def plot_hard_examples(
    dataset, hard_indices, labels, preds, confidences, class_names, output_dir
):
    """Show correctly classified images with lowest confidence."""
    n_show = min(12, len(hard_indices))
    cols = 4
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Hardest Correct Predictions (Lowest Confidence)",
        fontsize=14,
        fontweight="bold",
    )

    for idx, hard_idx in enumerate(hard_indices[:n_show]):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        img_path, _ = dataset.samples[hard_idx]
        img = Image.open(img_path).convert("RGB")

        true_label = class_names[labels[hard_idx]]
        conf = confidences[hard_idx]

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"{true_label}\nConf: {conf:.1%}",
            fontsize=10,
            color="orange" if conf < 0.9 else "green",
        )
        ax.axis("off")

    for idx in range(n_show, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "hard_examples.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_gradcam_errors(
    model,
    dataset,
    error_indices,
    labels,
    preds,
    confidences,
    class_names,
    device,
    output_dir,
):
    """Grad-CAM visualization on misclassified images."""
    gradcam = GradCAMVisualizer(model)

    n_errors = min(len(error_indices), 8)
    cols = min(4, n_errors)
    rows = (n_errors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = axes.reshape(1, -1)
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Grad-CAM on Misclassified Images", fontsize=14, fontweight="bold")

    for idx, error_idx in enumerate(error_indices[:n_errors]):
        row = idx // cols
        col_base = (idx % cols) * 2

        img_path, _ = dataset.samples[error_idx]
        orig_img = np.array(Image.open(img_path).convert("RGB"))
        tensor, _ = dataset[error_idx]
        tensor = tensor.unsqueeze(0).to(device)

        cam = gradcam.generate_cam(tensor)

        true_label = class_names[labels[error_idx]]
        pred_label = class_names[preds[error_idx]]
        conf = confidences[error_idx]

        # Original
        axes[row, col_base].imshow(orig_img)
        axes[row, col_base].set_title(f"True: {true_label}", fontsize=9)
        axes[row, col_base].axis("off")

        # Grad-CAM overlay
        cam_resized = (
            np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (orig_img.shape[1], orig_img.shape[0]), Image.BILINEAR
                )
            )
            / 255.0
        )

        axes[row, col_base + 1].imshow(orig_img)
        axes[row, col_base + 1].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[row, col_base + 1].set_title(
            f"Pred: {pred_label} ({conf:.0%})", fontsize=9, color="red"
        )
        axes[row, col_base + 1].axis("off")

    # Hide unused
    for idx in range(n_errors, rows * cols):
        row = idx // cols
        col_base = (idx % cols) * 2
        axes[row, col_base].axis("off")
        axes[row, col_base + 1].axis("off")

    gradcam.remove_hooks()
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_errors.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_space(model, test_loader, labels, class_names, device, output_dir):
    """t-SNE visualization of learned features."""
    model.eval()
    features_list = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            # Get features before classifier
            feat = model.model.backbone(images)
            if len(feat.shape) == 4:
                feat = model.model.global_pool(feat)
            feat = feat.view(feat.size(0), -1)
            features_list.append(feat.cpu().numpy())

    features = np.vstack(features_list)

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=class_name,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title("Feature Space Visualization (t-SNE)", fontsize=14, fontweight="bold")
    ax.legend(markerscale=2)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_space.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_gradcam_samples(model, dataset, class_names, device, output_dir):
    """Standard Grad-CAM visualization for each class."""
    gradcam = GradCAMVisualizer(model)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for class_idx, class_name in enumerate(class_names):
        # Find a sample of this class
        for i, (img_path, label) in enumerate(dataset.samples):
            if label == class_idx:
                orig_img = np.array(Image.open(img_path).convert("RGB"))
                tensor, _ = dataset[i]
                tensor = tensor.unsqueeze(0).to(device)

                cam = gradcam.generate_cam(tensor)
                cam_resized = (
                    np.array(
                        Image.fromarray((cam * 255).astype(np.uint8)).resize(
                            (orig_img.shape[1], orig_img.shape[0]), Image.BILINEAR
                        )
                    )
                    / 255.0
                )

                # Original
                axes[0, class_idx].imshow(orig_img)
                axes[0, class_idx].set_title(class_name, fontsize=12, fontweight="bold")
                axes[0, class_idx].axis("off")

                # Grad-CAM
                axes[1, class_idx].imshow(orig_img)
                axes[1, class_idx].imshow(cam_resized, cmap="jet", alpha=0.5)
                axes[1, class_idx].set_title("Grad-CAM", fontsize=11)
                axes[1, class_idx].axis("off")
                break

    gradcam.remove_hooks()
    plt.suptitle("Model Attention by Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_samples.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    generate_all_visualizations(args.checkpoint, args.config, args.output_dir)
