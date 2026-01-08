"""Grad-CAM visualization for model interpretability."""

from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAMVisualizer:
    """Grad-CAM visualization for understanding model predictions.

    Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients
    of any target concept flowing into the final convolutional layer to produce
    a coarse localization map highlighting important regions in the image.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """Initialize Grad-CAM visualizer.

        Args:
            model: The classification model
            target_layer: Name of the target layer for Grad-CAM (auto-detected if None)
        """
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None
        self.handles = []

        # Find target layer
        self.target_layer = self._find_target_layer(target_layer)
        self._register_hooks()

    def _find_target_layer(self, target_layer: Optional[str]) -> torch.nn.Module:
        """Find the target convolutional layer for Grad-CAM."""
        if target_layer is not None:
            # Find layer by name
            for name, module in self.model.named_modules():
                if name == target_layer:
                    return module

        # Auto-detect: find last conv layer in backbone
        last_conv = None
        for module in self.model.model.backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise ValueError("Could not find convolutional layer for Grad-CAM")

        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        """Remove registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def generate_cam(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Target class index (uses predicted class if None)

        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        cam = torch.mean(activations, dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / (cam.max() + 1e-8)  # Normalize

        return cam

    def visualize(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM visualization.

        Args:
            image: Original image as numpy array (H, W, C)
            input_tensor: Preprocessed input tensor (1, C, H, W)
            target_class: Target class (uses prediction if None)
            class_names: List of class names
            alpha: Transparency for overlay
            colormap: OpenCV colormap for heatmap

        Returns:
            Tuple of (visualization, predicted_class, confidence)
        """
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)

        visualization = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return visualization, pred_class, confidence

    def plot_gradcam_grid(
        self,
        images: List[np.ndarray],
        input_tensors: torch.Tensor,
        class_names: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ):
        """Plot a grid of Grad-CAM visualizations.

        Args:
            images: List of original images
            input_tensors: Batch of input tensors
            class_names: List of class names
            save_path: Path to save the figure
            figsize: Figure size
        """
        n_images = len(images)
        fig, axes = plt.subplots(2, n_images, figsize=figsize)

        if n_images == 1:
            axes = axes.reshape(2, 1)

        for i, (image, input_tensor) in enumerate(zip(images, input_tensors)):
            # Original image
            if image.max() <= 1:
                display_image = (image * 255).astype(np.uint8)
            else:
                display_image = image

            axes[0, i].imshow(display_image)
            axes[0, i].axis("off")
            axes[0, i].set_title("Original")

            # Grad-CAM
            visualization, pred_class, confidence = self.visualize(
                image, input_tensor.unsqueeze(0), class_names=class_names
            )

            axes[1, i].imshow(visualization)
            axes[1, i].axis("off")
            axes[1, i].set_title(
                f"Pred: {class_names[pred_class]}\nConf: {confidence:.2%}"
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved Grad-CAM visualization to {save_path}")

        plt.show()
        return fig
