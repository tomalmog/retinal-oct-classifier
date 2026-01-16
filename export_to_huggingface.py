"""Export trained model to Hugging Face format and optionally upload."""

import argparse
import shutil
from pathlib import Path

import torch

from src.models.classifier import OCTClassifier
from src.training.lightning_module import OCTLightningModule


def export_model(checkpoint_path: str, output_dir: str = "huggingface"):
    """Export Lightning checkpoint to standalone PyTorch format.

    Args:
        checkpoint_path: Path to .ckpt file from training
        output_dir: Directory to save exported model
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load the Lightning checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model state dict (remove 'model.' prefix from Lightning)
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("model."):
            new_key = key.replace("model.", "")
            state_dict[new_key] = value

    # Save in standard PyTorch format
    model_path = output_path / "pytorch_model.bin"
    torch.save(state_dict, model_path)
    print(f"Saved model weights to {model_path}")

    # Also save config
    if "hyper_parameters" in checkpoint:
        config_path = output_path / "config.json"
        import json

        with open(config_path, "w") as f:
            json.dump(checkpoint["hyper_parameters"], f, indent=2)
        print(f"Saved config to {config_path}")

    print(f"\nExport complete! Files in {output_path}/")
    print("\nNext steps:")
    print("1. Create a Hugging Face account at https://huggingface.co")
    print("2. Create a new model repository")
    print("3. Upload the contents of the huggingface/ directory")
    print("\nOr use the Hugging Face CLI:")
    print("  pip install huggingface_hub")
    print("  huggingface-cli login")
    print("  huggingface-cli upload your-username/oct-classifier huggingface/")


def upload_to_hub(repo_id: str, output_dir: str = "huggingface", private: bool = False):
    """Upload model to Hugging Face Hub.

    Args:
        repo_id: Repository ID (e.g., 'username/oct-classifier')
        output_dir: Directory containing model files
        private: Whether to make the repo private
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Upload all files
    output_path = Path(output_dir)

    print(f"\nUploading files from {output_path}...")
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nUpload complete!")
    print(f"View your model at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Export model to Hugging Face format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="huggingface",
        help="Output directory for exported files",
    )
    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="Hugging Face repo ID to upload to (e.g., 'username/oct-classifier')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )

    args = parser.parse_args()

    # Export model
    export_model(args.checkpoint, args.output_dir)

    # Copy visualization files
    outputs_dir = Path("outputs")
    hf_dir = Path(args.output_dir)

    for img_file in ["gradcam_samples.png", "confusion_matrix.png", "roc_curves.png"]:
        src = outputs_dir / img_file
        if src.exists():
            shutil.copy(src, hf_dir / img_file)
            print(f"Copied {img_file}")

    # Upload if requested
    if args.upload:
        upload_to_hub(args.upload, args.output_dir, args.private)


if __name__ == "__main__":
    main()
