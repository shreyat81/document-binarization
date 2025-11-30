"""
Hybrid Document Binarization: Neural Network + Fuzzy Logic
===========================================================

This script demonstrates how to combine the trained neural network model
with the fuzzy logic system for enhanced document binarization.

The hybrid approach:
1. Uses the neural network to generate probability maps
2. Applies fuzzy logic to refine the output using local features
3. Produces final binary images with improved quality

Usage:
------
python hybrid_inference.py --checkpoint best_model.pth \
                           --input document.png \
                           --output result.png \
                           --mode hybrid

Modes:
------
- 'neural': Neural network only (baseline)
- 'fuzzy': Fuzzy system only
- 'hybrid': Neural + Fuzzy (recommended)
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import FastBinarizationModel
from src.fuzzy.fuzzy_system import FuzzyBinarizer


class HybridBinarizer:
    """
    Hybrid binarization system combining neural network and fuzzy logic.
    """
    
    def __init__(self, checkpoint_path: str, fuzzy_params: dict = None, device: str = 'cpu'):
        """
        Initialize hybrid binarizer.
        
        Parameters:
        -----------
        checkpoint_path : str
            Path to trained neural network checkpoint
        fuzzy_params : dict, optional
            Fuzzy system parameters (uses defaults if None)
        device : str
            Device for neural network ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Load neural network
        print(f"📦 Loading neural network from {checkpoint_path}")
        self.model = FastBinarizationModel(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        if 'val_f1' in checkpoint:
            print(f"   Model F1 Score: {checkpoint['val_f1']:.4f}")
        
        # Initialize fuzzy system
        print("🔧 Initializing fuzzy logic system")
        self.fuzzy = FuzzyBinarizer(window_size=9)
        self.fuzzy_params = fuzzy_params if fuzzy_params else self.fuzzy.default_params()
        
        print(f"   Fuzzy rules: {len(self.fuzzy_params['rules'])}")
        print("✅ Hybrid system ready!\n")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        img = Image.open(image_path).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        return img_np
    
    def neural_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Run neural network inference.
        
        Parameters:
        -----------
        image : np.ndarray
            Grayscale image [0, 1], shape (H, W)
            
        Returns:
        --------
        np.ndarray
            Probability map [0, 1], shape (H, W)
        """
        # Prepare tensor
        if len(image.shape) == 2:
            img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            img_tensor = torch.from_numpy(image).unsqueeze(0)
        
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)
        
        # Convert to numpy
        prob_map = probs.squeeze().cpu().numpy()
        
        return prob_map
    
    def binarize_neural(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binarize using neural network only.
        
        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image [0, 1]
        threshold : float
            Binarization threshold
            
        Returns:
        --------
        np.ndarray
            Binary image (0 or 255)
        """
        prob_map = self.neural_inference(image)
        binary = (prob_map > threshold).astype(np.uint8) * 255
        return binary
    
    def binarize_hybrid(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize using hybrid neural + fuzzy approach.
        
        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image [0, 1]
            
        Returns:
        --------
        np.ndarray
            Binary image (0 or 255)
        """
        # Get neural network probability map
        print("🧠 Running neural network inference...")
        prob_map = self.neural_inference(image)
        print(f"   Probability range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
        
        # Apply fuzzy refinement
        print("\n🔀 Applying fuzzy logic refinement...")
        binary = self.fuzzy.infer(image, prob_map, self.fuzzy_params)
        
        return binary
    
    def process_image(self, input_path: str, output_path: str, mode: str = 'hybrid'):
        """
        Process a single image.
        
        Parameters:
        -----------
        input_path : str
            Input image path
        output_path : str
            Output image path
        mode : str
            Processing mode: 'neural', 'fuzzy', or 'hybrid'
        """
        print(f"\n{'='*80}")
        print(f"Processing: {input_path}")
        print(f"Mode: {mode.upper()}")
        print(f"{'='*80}\n")
        
        # Load image
        image = self.load_image(input_path)
        print(f"📸 Loaded image: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Process based on mode
        if mode == 'neural':
            binary = self.binarize_neural(image)
        elif mode == 'hybrid':
            binary = self.binarize_hybrid(image)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Save result
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary).save(output_path)
        
        print(f"\n✅ Saved result to: {output_path}")
        print(f"   Foreground: {(binary == 255).sum()} pixels ({(binary == 255).sum() / binary.size * 100:.1f}%)")
        print(f"   Background: {(binary == 0).sum()} pixels ({(binary == 0).sum() / binary.size * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Hybrid Document Binarization')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to neural network checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image path')
    parser.add_argument('--mode', type=str, default='hybrid',
                       choices=['neural', 'hybrid'],
                       help='Processing mode (default: hybrid)')
    parser.add_argument('--fuzzy-params', type=str, default=None,
                       help='Path to fuzzy parameters JSON (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for neural network')
    
    args = parser.parse_args()
    
    # Load fuzzy params if provided
    fuzzy_params = None
    if args.fuzzy_params:
        fuzzy_params = FuzzyBinarizer.load_params(args.fuzzy_params)
        print(f"📋 Loaded fuzzy parameters from {args.fuzzy_params}")
    
    # Initialize hybrid system
    hybrid = HybridBinarizer(
        checkpoint_path=args.checkpoint,
        fuzzy_params=fuzzy_params,
        device=args.device
    )
    
    # Process image
    hybrid.process_image(args.input, args.output, mode=args.mode)


if __name__ == '__main__':
    main()
