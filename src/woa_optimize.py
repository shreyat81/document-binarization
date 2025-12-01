"""
Whale Optimization Algorithm (WOA) for Threshold Optimization

This module implements WOA to optimize the binarization threshold for the neural network
output, improving performance across different document types.

Reference: Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
           Advances in Engineering Software, 95, 51-67.
"""

import numpy as np
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import FastBinarizationModel


def calculate_f1(pred_binary, gt_binary):
    """
    Calculate F1 score between predicted and ground truth binary images.
    
    Args:
        pred_binary: Predicted binary image (0 or 255)
        gt_binary: Ground truth binary image (0 or 255)
    
    Returns:
        F1 score (float)
    """
    pred = (pred_binary > 127).astype(np.float32)
    gt = (gt_binary > 127).astype(np.float32)
    
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1


class WhaleOptimizer:
    """
    Whale Optimization Algorithm for threshold tuning.
    
    Implements the three main behaviors:
    1. Encircling prey (exploitation)
    2. Bubble-net attacking (exploitation via spiral)
    3. Search for prey (exploration)
    """
    
    def __init__(self, fitness_func, lower_bound=0.3, upper_bound=0.7, 
                 pop_size=10, max_iter=20):
        """
        Initialize WOA optimizer.
        
        Args:
            fitness_func: Function to evaluate (we maximize F1)
            lower_bound: Lower bound for threshold
            upper_bound: Upper bound for threshold
            pop_size: Population size (number of whales)
            max_iter: Maximum iterations
        """
        self.fitness_func = fitness_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # WOA parameter
        self.b = 1  # Spiral shape constant
        
        # Population (thresholds)
        self.positions = None
        self.fitness = None
        
        # Best solution
        self.best_position = None
        self.best_fitness = -np.inf
        
        # History
        self.fitness_history = []
    
    def initialize_population(self):
        """Initialize population with random thresholds."""
        self.positions = np.random.uniform(
            self.lower_bound, 
            self.upper_bound, 
            self.pop_size
        ).astype(np.float32)
        
        # Initialize fitness
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)
    
    def evaluate_population(self):
        """Evaluate fitness for all individuals."""
        for i in range(self.pop_size):
            self.fitness[i] = self.fitness_func(self.positions[i])
            
            # Update best
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.positions[i]
    
    def optimize(self, verbose=True):
        """
        Run WOA optimization.
        
        Args:
            verbose: Whether to show progress
        
        Returns:
            Best threshold found
        """
        # Initialize
        self.initialize_population()
        self.evaluate_population()
        
        # Progress bar
        pbar = tqdm(range(self.max_iter), desc="🐋 WOA Optimization", 
                    disable=not verbose)
        
        for t in pbar:
            # Linearly decrease a from 2 to 0
            a = 2 - 2 * t / self.max_iter
            
            for i in range(self.pop_size):
                # Update position of each whale
                r1 = np.random.random()
                r2 = np.random.random()
                
                A = 2 * a * r1 - a  # Eq. 2.3
                C = 2 * r2          # Eq. 2.4
                
                p = np.random.random()  # Random number [0,1]
                
                if p < 0.5:
                    if np.abs(A) < 1:
                        # Encircling prey (exploitation)
                        D = np.abs(C * self.best_position - self.positions[i])
                        self.positions[i] = self.best_position - A * D
                    else:
                        # Search for prey (exploration)
                        rand_idx = np.random.randint(0, self.pop_size)
                        X_rand = self.positions[rand_idx]
                        D = np.abs(C * X_rand - self.positions[i])
                        self.positions[i] = X_rand - A * D
                else:
                    # Bubble-net attacking (spiral update)
                    D_prime = np.abs(self.best_position - self.positions[i])
                    l = np.random.uniform(-1, 1)  # Random in [-1,1]
                    self.positions[i] = (D_prime * np.exp(self.b * l) * 
                                        np.cos(2 * np.pi * l) + 
                                        self.best_position)
                
                # Enforce bounds
                self.positions[i] = np.clip(
                    self.positions[i],
                    self.lower_bound,
                    self.upper_bound
                )
            
            # Evaluate new positions
            self.evaluate_population()
            
            # Record history
            self.fitness_history.append(self.best_fitness)
            
            # Update progress bar
            pbar.set_postfix({
                'Best F1': f'{self.best_fitness:.4f}',
                'Best Thresh': f'{self.best_position:.3f}',
                'Mean F1': f'{np.mean(self.fitness):.4f}'
            })
        
        return self.best_position


class NeuralBinarizer:
    """Wrapper for neural network binarization."""
    
    def __init__(self, checkpoint_path, device='cpu'):
        """Initialize neural network model."""
        self.device = torch.device(device)
        
        # Load model
        self.model = FastBinarizationModel(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded model from {checkpoint_path}")
        if 'val_f1' in checkpoint:
            print(f"   Model F1 Score: {checkpoint['val_f1']:.4f}")
    
    def predict_proba(self, image_path):
        """
        Get probability map for an image.
        
        Args:
            image_path: Path to image file (.npy, .png, .jpg, etc.)
        
        Returns:
            Probability map as numpy array [0, 1]
        """
        # Load image
        if str(image_path).endswith('.npy'):
            img_np = np.load(image_path).astype(np.float32)
            # Ensure it's in [0, 1] range
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
        else:
            img = Image.open(image_path).convert('L')
            img_np = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            prob = self.model(img_tensor)
            prob = torch.sigmoid(prob)
            prob_np = prob.cpu().numpy()[0, 0]
        
        return prob_np
    
    def binarize(self, prob_map, threshold=0.5):
        """
        Apply threshold to probability map.
        
        Args:
            prob_map: Probability map [0, 1]
            threshold: Binarization threshold
        
        Returns:
            Binary image (0 or 255)
        """
        binary = (prob_map > threshold).astype(np.uint8) * 255
        return binary


def load_dataset(images_dir, gt_dir, sample_limit=None):
    """
    Load validation dataset.
    
    Args:
        images_dir: Directory with images
        gt_dir: Directory with ground truth masks
        sample_limit: Maximum samples to load
    
    Returns:
        List of (image_path, gt_mask) tuples
    """
    images_dir = Path(images_dir)
    gt_dir = Path(gt_dir)
    
    dataset = []
    
    # Get all images (support .npy, .png, .jpg, .bmp)
    image_files = sorted(list(images_dir.glob('*.npy')) +
                        list(images_dir.glob('*.png')) + 
                        list(images_dir.glob('*.jpg')) +
                        list(images_dir.glob('*.bmp')))
    
    if sample_limit:
        image_files = image_files[:sample_limit]
    
    print(f"📂 Loading {len(image_files)} samples...")
    
    for img_path in tqdm(image_files, desc="Loading"):
        # Find corresponding GT
        if img_path.suffix == '.npy':
            # For .npy files, look for GT with _GT suffix
            base_name = img_path.stem
            gt_patterns = [
                gt_dir / f"{base_name.replace('_p', '_GT_p')}.npy",
                gt_dir / f"{base_name}_GT.npy",
                gt_dir / f"{base_name}_GT.png",
            ]
        else:
            gt_patterns = [
                gt_dir / img_path.name,
                gt_dir / img_path.name.replace('.png', '_GT.png'),
                gt_dir / img_path.name.replace('.jpg', '_GT.png'),
            ]
        
        gt_path = None
        for pattern in gt_patterns:
            if pattern.exists():
                gt_path = pattern
                break
        
        if gt_path is None:
            print(f"⚠️  Warning: No GT found for {img_path.name}")
            continue
        
        # Load GT
        if gt_path.suffix == '.npy':
            gt_np = np.load(gt_path)
            gt_np = (gt_np * 255).astype(np.uint8) if gt_np.max() <= 1.0 else gt_np.astype(np.uint8)
        else:
            gt_img = Image.open(gt_path).convert('L')
            gt_np = np.array(gt_img)
        
        dataset.append((str(img_path), gt_np))
    
    print(f"✅ Loaded {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Optimize binarization threshold using Whale Optimization Algorithm'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory with validation images')
    parser.add_argument('--gt', type=str, required=True,
                       help='Directory with ground truth masks')
    parser.add_argument('--pop', type=int, default=10,
                       help='Population size (default: 10)')
    parser.add_argument('--iters', type=int, default=20,
                       help='Number of iterations (default: 20)')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of validation samples to use (default: 50)')
    parser.add_argument('--output', type=str, default='woa_threshold.json',
                       help='Output JSON file for best threshold')
    parser.add_argument('--mode', type=str, choices=['quick', 'normal', 'full'],
                       default='normal',
                       help='Optimization mode (quick/normal/full)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Adjust parameters based on mode
    if args.mode == 'quick':
        pop_size = 6
        max_iter = 10
        sample_limit = 20
    elif args.mode == 'full':
        pop_size = 15
        max_iter = 30
        sample_limit = 100
    else:  # normal
        pop_size = args.pop
        max_iter = args.iters
        sample_limit = args.samples
    
    print("=" * 80)
    print("🐋 Whale Optimization Algorithm - Threshold Optimization")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Population: {pop_size} whales")
    print(f"Iterations: {max_iter}")
    print(f"Samples: {sample_limit}")
    print("=" * 80)
    print()
    
    # Load model
    binarizer = NeuralBinarizer(args.checkpoint, args.device)
    
    # Load dataset
    dataset = load_dataset(args.images, args.gt, sample_limit)
    
    if len(dataset) == 0:
        print("❌ Error: No samples loaded!")
        return
    
    # Pre-compute probability maps
    print("\n📊 Computing probability maps...")
    prob_maps = []
    gt_masks = []
    
    for img_path, gt_mask in tqdm(dataset, desc="Processing"):
        prob_map = binarizer.predict_proba(img_path)
        prob_maps.append(prob_map)
        gt_masks.append(gt_mask)
    
    # Define fitness function
    def fitness_function(threshold):
        """Evaluate threshold on all samples."""
        f1_scores = []
        for prob_map, gt_mask in zip(prob_maps, gt_masks):
            binary = binarizer.binarize(prob_map, threshold)
            f1 = calculate_f1(binary, gt_mask)
            f1_scores.append(f1)
        return np.mean(f1_scores)
    
    # Evaluate baseline (threshold=0.5)
    print("\n📏 Evaluating baseline (threshold=0.5)...")
    baseline_f1 = fitness_function(0.5)
    print(f"   Baseline F1: {baseline_f1:.4f}")
    
    # Run WOA
    print(f"\n🐋 Starting Whale Optimization...\n")
    
    woa = WhaleOptimizer(
        fitness_func=fitness_function,
        lower_bound=0.3,
        upper_bound=0.7,
        pop_size=pop_size,
        max_iter=max_iter
    )
    
    best_threshold = woa.optimize(verbose=True)
    
    # Results
    print(f"\n{'=' * 80}")
    print("✅ Optimization Complete!")
    print(f"{'=' * 80}")
    print(f"Baseline F1 (t=0.5):  {baseline_f1:.4f}")
    print(f"Optimized F1:         {woa.best_fitness:.4f}")
    print(f"Best Threshold:       {best_threshold:.4f}")
    print(f"Improvement:          {(woa.best_fitness - baseline_f1):.4f} "
          f"({100 * (woa.best_fitness - baseline_f1) / baseline_f1:+.2f}%)")
    print(f"{'=' * 80}")
    
    # Save result
    result = {
        'best_threshold': float(best_threshold),
        'best_f1': float(woa.best_fitness),
        'baseline_f1': float(baseline_f1),
        'improvement': float(woa.best_fitness - baseline_f1),
        'mode': args.mode,
        'population_size': pop_size,
        'iterations': max_iter,
        'samples_used': len(dataset)
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()
