"""
Generate probability maps for validation set using the trained model.

This script is a helper to prepare data for WOA optimization.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model import FastBinarizationModel


def generate_probability_maps(model_path, images_dir, output_dir, device='cpu'):
    """
    Generate probability maps for all images in a directory.
    
    Args:
        model_path: Path to trained model checkpoint
        images_dir: Directory with input images (.npy files)
        output_dir: Directory to save probability maps
        device: Device to use (cpu/cuda)
    """
    # Load model
    print(f"Loading model from {model_path}...")
    device = torch.device(device)
    model = FastBinarizationModel(pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'val_f1' in checkpoint:
        print(f"Model F1: {checkpoint['val_f1']:.4f}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files
    images_dir = Path(images_dir)
    image_files = sorted(images_dir.glob('*.npy'))
    
    print(f"\nFound {len(image_files)} images in {images_dir}")
    print(f"Generating probability maps...\n")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing"):
        # Load image
        img_np = np.load(img_file)
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        
        # Add dimensions
        if len(img_np.shape) == 2:
            img_np = img_np[np.newaxis, ...]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
        
        # Generate probability map
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
        
        # Convert to numpy
        prob_map = probs.squeeze().cpu().numpy()
        
        # Save
        output_file = output_dir / f"{img_file.stem}.npy"
        np.save(output_file, prob_map)
    
    print(f"\n✅ Generated {len(image_files)} probability maps")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate probability maps for WOA optimization'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory with input images (.npy)')
    parser.add_argument('--output', type=str, required=True,
                       help='Directory to save probability maps')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    generate_probability_maps(
        args.checkpoint,
        args.images,
        args.output,
        args.device
    )


if __name__ == '__main__':
    main()
