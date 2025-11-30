"""
Whale Optimization Algorithm (WOA) for Fuzzy Binarizer Parameter Tuning

This module implements WOA to optimize fuzzy membership functions and rule weights
for improved document binarization performance.

Reference: Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
           Advances in Engineering Software, 95, 51-67.
"""

import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fuzzy.fuzzy_system import FuzzyBinarizer


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


def params_to_vector(params):
    """
    Convert fuzzy parameters dict to flat vector for optimization.
    
    Includes:
    - Membership function mu and sigma for each variable (prob, contrast, entropy)
    - Rule weights
    - Final threshold
    
    Args:
        params: Parameters dictionary from FuzzyBinarizer
    
    Returns:
        Flat numpy vector
    """
    vector = []
    
    # Membership functions: prob, contrast, entropy (each has low, mid, high)
    for var in ['prob', 'contrast', 'entropy']:
        for level in ['low', 'mid', 'high']:
            mu, sigma = params[var][level]
            vector.extend([mu, sigma])
    
    # Rule weights
    for rule in params['rules']:
        vector.append(rule['weight'])
    
    # Threshold
    vector.append(params['threshold'])
    
    return np.array(vector, dtype=np.float32)


def vector_to_params(vector, template_params):
    """
    Convert flat vector back to fuzzy parameters dict.
    
    Args:
        vector: Flat numpy vector
        template_params: Template parameters for structure
    
    Returns:
        Parameters dictionary
    """
    params = json.loads(json.dumps(template_params))  # Deep copy
    
    idx = 0
    
    # Membership functions
    for var in ['prob', 'contrast', 'entropy']:
        for level in ['low', 'mid', 'high']:
            mu = float(vector[idx])
            sigma = float(vector[idx + 1])
            params[var][level] = [mu, sigma]
            idx += 2
    
    # Rule weights
    for i, rule in enumerate(params['rules']):
        rule['weight'] = float(vector[idx])
        idx += 1
    
    # Threshold
    params['threshold'] = float(vector[idx])
    
    return params


def get_param_bounds(template_params):
    """
    Get reasonable bounds for each parameter based on template.
    
    Args:
        template_params: Template parameters to get number of rules
    
    Returns:
        Tuple of (lower_bounds, upper_bounds) as numpy arrays
    """
    lower = []
    upper = []
    
    # Membership functions
    # prob: mu in [0,1], sigma in [0.05, 0.5]
    for _ in range(3):  # low, mid, high
        lower.extend([0.0, 0.05])
        upper.extend([1.0, 0.5])
    
    # contrast: mu in [0,0.5], sigma in [0.01, 0.3]
    for _ in range(3):
        lower.extend([0.0, 0.01])
        upper.extend([0.5, 0.3])
    
    # entropy: mu in [0,5], sigma in [0.1, 3.0]
    for _ in range(3):
        lower.extend([0.0, 0.1])
        upper.extend([5.0, 3.0])
    
    # Rule weights: [0.1, 2.0]
    num_rules = len(template_params['rules'])
    for _ in range(num_rules):
        lower.append(0.1)
        upper.append(2.0)
    
    # Threshold: [0.3, 0.7]
    lower.append(0.3)
    upper.append(0.7)
    
    return np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)


def enforce_bounds(vector, lower, upper):
    """
    Enforce parameter bounds by clipping.
    
    Args:
        vector: Parameter vector
        lower: Lower bounds
        upper: Upper bounds
    
    Returns:
        Clipped vector
    """
    return np.clip(vector, lower, upper)


def fitness_function(vector, template_params, fuzzy_binarizer, 
                     orig_images, prob_maps, gt_masks):
    """
    Evaluate fitness (mean F1 score) for a parameter vector.
    
    Args:
        vector: Parameter vector
        template_params: Template parameters structure
        fuzzy_binarizer: FuzzyBinarizer instance
        orig_images: List of original images
        prob_maps: List of probability maps
        gt_masks: List of ground truth masks
    
    Returns:
        Fitness score (mean F1)
    """
    # Convert vector to params
    params = vector_to_params(vector, template_params)
    
    f1_scores = []
    
    # Evaluate on all validation samples (verbose=False for speed)
    for orig, prob, gt in zip(orig_images, prob_maps, gt_masks):
        try:
            # Run fuzzy inference with verbose=False
            binary = fuzzy_binarizer.infer(orig, prob, params, verbose=False)
            
            # Calculate F1
            f1 = calculate_f1(binary, gt)
            f1_scores.append(f1)
        except Exception as e:
            # If inference fails, return poor fitness
            return 0.0
    
    # Return mean F1 score
    return np.mean(f1_scores)


class WhaleOptimizer:
    """
    Whale Optimization Algorithm for parameter tuning.
    
    Implements the three main behaviors:
    1. Encircling prey (exploitation)
    2. Bubble-net attacking (exploitation via spiral)
    3. Search for prey (exploration)
    """
    
    def __init__(self, fitness_func, dim, pop_size=12, max_iter=30, 
                 lower_bounds=None, upper_bounds=None):
        """
        Initialize WOA optimizer.
        
        Args:
            fitness_func: Function to minimize (we maximize F1, so negate)
            dim: Dimension of search space
            pop_size: Population size
            max_iter: Maximum iterations
            lower_bounds: Lower bounds for parameters
            upper_bounds: Upper bounds for parameters
        """
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        # WOA parameters
        self.b = 1  # Spiral shape constant
        
        # Population
        self.positions = None
        self.fitness = None
        
        # Best solution
        self.best_position = None
        self.best_fitness = -np.inf
        
        # History
        self.fitness_history = []
    
    def initialize_population(self, initial_position):
        """
        Initialize population around initial position with Gaussian noise.
        
        Args:
            initial_position: Starting point (default params)
        """
        self.positions = np.zeros((self.pop_size, self.dim), dtype=np.float32)
        
        # First individual is the initial position
        self.positions[0] = initial_position.copy()
        
        # Rest with Gaussian noise
        for i in range(1, self.pop_size):
            noise = np.random.normal(0, 0.1, self.dim)
            self.positions[i] = initial_position + noise
            self.positions[i] = enforce_bounds(
                self.positions[i], 
                self.lower_bounds, 
                self.upper_bounds
            )
        
        # Initialize fitness
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)
    
    def evaluate_population(self):
        """Evaluate fitness for all individuals."""
        for i in range(self.pop_size):
            self.fitness[i] = self.fitness_func(self.positions[i])
            
            # Update best
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.positions[i].copy()
    
    def optimize(self, initial_position, verbose=True):
        """
        Run WOA optimization.
        
        Args:
            initial_position: Starting parameter vector
            verbose: Whether to show progress
        
        Returns:
            Best position found
        """
        # Initialize
        self.initialize_population(initial_position)
        self.evaluate_population()
        
        # Progress bar
        pbar = tqdm(range(self.max_iter), desc="WOA Optimization", 
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
                self.positions[i] = enforce_bounds(
                    self.positions[i],
                    self.lower_bounds,
                    self.upper_bounds
                )
            
            # Evaluate new positions
            self.evaluate_population()
            
            # Record history
            self.fitness_history.append(self.best_fitness)
            
            # Update progress bar
            pbar.set_postfix({
                'Best F1': f'{self.best_fitness:.4f}',
                'Mean F1': f'{np.mean(self.fitness):.4f}'
            })
        
        return self.best_position


def load_validation_data(imgs_dir, probs_dir, gts_dir, sample_limit=None):
    """
    Load validation images, probability maps, and ground truth masks.
    
    Args:
        imgs_dir: Directory with original images
        probs_dir: Directory with probability .npy files
        gts_dir: Directory with ground truth masks
        sample_limit: Maximum number of samples to load (for fast tuning)
    
    Returns:
        Tuple of (orig_images, prob_maps, gt_masks)
    """
    imgs_dir = Path(imgs_dir)
    probs_dir = Path(probs_dir)
    gts_dir = Path(gts_dir)
    
    orig_images = []
    prob_maps = []
    gt_masks = []
    
    # Get all .npy files from probs directory
    prob_files = sorted(probs_dir.glob('*.npy'))
    
    if sample_limit:
        prob_files = prob_files[:sample_limit]
    
    print(f"Loading {len(prob_files)} validation samples...")
    
    for prob_file in tqdm(prob_files, desc="Loading data"):
        # Load probability map
        prob = np.load(prob_file)
        
        # Find corresponding original image
        base_name = prob_file.stem
        
        # Try different patterns
        img_patterns = [
            imgs_dir / f"{base_name}.npy",
            imgs_dir / f"{base_name}.png",
            imgs_dir / f"{base_name}.jpg",
        ]
        
        img_file = None
        for pattern in img_patterns:
            if pattern.exists():
                img_file = pattern
                break
        
        if img_file is None:
            print(f"Warning: Could not find image for {prob_file.name}")
            continue
        
        # Load original image
        if img_file.suffix == '.npy':
            orig = np.load(img_file)
        else:
            from PIL import Image
            orig = np.array(Image.open(img_file).convert('L')).astype(np.float32) / 255.0
        
        # Find ground truth
        if '_p' in base_name:
            gt_name = base_name.replace('_p', '_GT_p')
        else:
            gt_name = base_name + '_GT'
        
        gt_patterns = [
            gts_dir / f"{gt_name}.npy",
            gts_dir / f"{gt_name}.png",
        ]
        
        gt_file = None
        for pattern in gt_patterns:
            if pattern.exists():
                gt_file = pattern
                break
        
        if gt_file is None:
            print(f"Warning: Could not find GT for {prob_file.name}")
            continue
        
        # Load ground truth
        if gt_file.suffix == '.npy':
            gt = np.load(gt_file)
            gt = (gt * 255).astype(np.uint8)
        else:
            from PIL import Image
            gt = np.array(Image.open(gt_file).convert('L'))
        
        # Ensure shapes match
        if orig.shape != prob.shape or orig.shape != gt.shape:
            print(f"Warning: Shape mismatch for {prob_file.name}")
            continue
        
        orig_images.append(orig)
        prob_maps.append(prob)
        gt_masks.append(gt)
    
    print(f"Successfully loaded {len(orig_images)} samples")
    
    return orig_images, prob_maps, gt_masks


def main():
    parser = argparse.ArgumentParser(
        description='Optimize fuzzy binarizer parameters using WOA'
    )
    parser.add_argument('--imgs', type=str, required=True,
                       help='Directory with original validation images')
    parser.add_argument('--probs', type=str, required=True,
                       help='Directory with probability maps (.npy)')
    parser.add_argument('--gts', type=str, required=True,
                       help='Path to ground truth masks directory')
    parser.add_argument('--pop', type=int, default=6,
                       help='Population size (default: 6, optimized for CPU)')
    parser.add_argument('--iters', type=int, default=8,
                       help='Number of iterations (default: 8, optimized for CPU)')
    parser.add_argument('--out', type=str, default='best_params.json',
                       help='Output file for best parameters')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--sample_limit', type=int, default=15,
                       help='Limit validation samples for fast tuning (default: 15 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("Whale Optimization Algorithm - Fuzzy Parameter Tuning")
    print("=" * 80)
    
    # Load validation data
    orig_images, prob_maps, gt_masks = load_validation_data(
        args.imgs, args.probs, args.gts, args.sample_limit
    )
    
    if len(orig_images) == 0:
        print("Error: No validation samples loaded!")
        return
    
    # Initialize fuzzy binarizer
    print("\nInitializing FuzzyBinarizer...")
    fuzzy_binarizer = FuzzyBinarizer()
    template_params = fuzzy_binarizer.default_params()
    
    # Get initial vector
    initial_vector = params_to_vector(template_params)
    print(f"Parameter dimension: {len(initial_vector)}")
    
    # Get bounds
    lower_bounds, upper_bounds = get_param_bounds(template_params)
    print(f"Parameter bounds dimension: {len(lower_bounds)}")
    
    # Define fitness function wrapper
    def fitness_wrapper(vector):
        return fitness_function(
            vector, template_params, fuzzy_binarizer,
            orig_images, prob_maps, gt_masks
        )
    
    # Evaluate initial fitness
    print("\nEvaluating initial parameters...")
    initial_fitness = fitness_wrapper(initial_vector)
    print(f"Initial F1 Score: {initial_fitness:.4f}")
    
    # Initialize WOA
    print(f"\nStarting WOA optimization...")
    print(f"Population size: {args.pop}")
    print(f"Iterations: {args.iters}")
    print(f"Validation samples: {len(orig_images)}")
    print()
    
    woa = WhaleOptimizer(
        fitness_func=fitness_wrapper,
        dim=len(initial_vector),
        pop_size=args.pop,
        max_iter=args.iters,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds
    )
    
    # Run optimization
    best_vector = woa.optimize(initial_vector, verbose=True)
    
    # Convert to params
    best_params = vector_to_params(best_vector, template_params)
    
    # Save best parameters
    print(f"\n{'=' * 80}")
    print("Optimization Complete!")
    print(f"{'=' * 80}")
    print(f"Initial F1: {initial_fitness:.4f}")
    print(f"Best F1:    {woa.best_fitness:.4f}")
    print(f"Improvement: {(woa.best_fitness - initial_fitness):.4f} "
          f"({100 * (woa.best_fitness - initial_fitness) / initial_fitness:.2f}%)")
    
    # Save to file
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nBest parameters saved to: {output_path}")
    
    # Print fitness history
    print(f"\nFitness History (first 10 iterations):")
    for i, fitness in enumerate(woa.fitness_history[:10], 1):
        print(f"  Iteration {i:2d}: F1 = {fitness:.4f}")
    
    if len(woa.fitness_history) > 10:
        print("  ...")
        for i, fitness in enumerate(woa.fitness_history[-3:], 
                                    len(woa.fitness_history) - 2):
            print(f"  Iteration {i:2d}: F1 = {fitness:.4f}")
    
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
