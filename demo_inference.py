"""
Quick Demo Script for Inference
================================

Simple script to test your trained model on a few images.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from model import FastBinarizationModel


def load_model(checkpoint_path, device='cpu'):
    """Load trained model."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = FastBinarizationModel(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    if 'val_f1' in checkpoint:
        print(f"Model F1 Score: {checkpoint['val_f1']:.4f}")
    
    return model


def binarize_image(model, image_path, device='cpu'):
    """Binarize a single image."""
    # Load image
    img = Image.open(image_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Prepare tensor
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
    
    # Convert to binary
    binary = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    
    return img_np * 255, binary


def visualize_results(original, binary, save_path=None):
    """Visualize original and binarized images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title('Binarized Result')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def demo():
    """Run demo on test images."""
    print("="*80)
    print("Document Binarization Demo")
    print("="*80)
    
    # Setup
    checkpoint_path = "best_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Get test images
    test_images_dir = Path("split/test/images")
    
    if not test_images_dir.exists():
        print(f"\n❌ Test directory not found: {test_images_dir}")
        print("Please provide path to test images or use inference.py")
        return
    
    # Get first 3 .npy files
    test_files = sorted(list(test_images_dir.glob("*.npy")))[:3]
    
    if len(test_files) == 0:
        print(f"\n❌ No .npy files found in {test_images_dir}")
        return
    
    print(f"\nProcessing {len(test_files)} test images...\n")
    
    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process each image
    for idx, test_file in enumerate(test_files, 1):
        print(f"[{idx}/{len(test_files)}] Processing: {test_file.name}")
        
        # Load and process
        img_np = np.load(test_file)
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        
        # Add dimensions
        if len(img_np.shape) == 2:
            img_np_input = img_np[np.newaxis, ...]
        else:
            img_np_input = img_np
        
        img_tensor = torch.from_numpy(img_np_input).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
        
        binary = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        
        # Save results
        output_path = output_dir / f"{test_file.stem}_binarized.png"
        Image.fromarray(binary).save(output_path)
        
        # Visualize
        if idx <= 3:  # Only show first 3
            visualize_results(
                img_np if len(img_np.shape) == 2 else img_np[0],
                binary,
                save_path=output_dir / f"{test_file.stem}_comparison.png"
            )
    
    print(f"\n✅ Demo complete! Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    demo()
