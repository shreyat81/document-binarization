"""
Fuzzy Binarization System for Document Images
==============================================

This module implements a Mamdani-style fuzzy inference system that combines
neural network probability outputs with local image features to produce
high-quality document binarization.

Features:
---------
- Gaussian membership functions for prob, contrast, and entropy
- Configurable fuzzy rules with weights
- Local feature computation (contrast, entropy) using sliding windows
- Mamdani-style inference with centroid defuzzification
- Optimized with numba (optional) for fast local computations
- JSON-based parameter saving/loading

Example Usage:
--------------
>>> import numpy as np
>>> from fuzzy_system import FuzzyBinarizer
>>> 
>>> # Initialize
>>> fuzzy = FuzzyBinarizer()
>>> 
>>> # Create sample inputs
>>> orig = np.random.rand(256, 256).astype(np.float32)
>>> prob_map = np.random.rand(256, 256).astype(np.float32)
>>> 
>>> # Run fuzzy inference
>>> binary = fuzzy.infer(orig, prob_map)
>>> print(binary.shape, binary.dtype)  # (256, 256), uint8
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import generic_filter
from scipy.stats import entropy as scipy_entropy

# Try to import numba for optimization
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class FuzzyBinarizer:
    """
    Fuzzy Logic Binarization System
    
    Combines neural network probability maps with local image features
    (contrast and entropy) using fuzzy logic rules to produce improved
    binary document images.
    
    The system uses:
    - Gaussian membership functions for input fuzzification
    - Mamdani-style fuzzy inference with weighted rules
    - Centroid defuzzification for crisp output
    
    Attributes:
    -----------
    window_size : int
        Size of local window for feature computation (default: 9)
    """
    
    def __init__(self, window_size: int = 9):
        """
        Initialize FuzzyBinarizer.
        
        Parameters:
        -----------
        window_size : int, optional
            Size of sliding window for local features (default: 9)
        """
        self.window_size = window_size
    
    @staticmethod
    def default_params() -> dict:
        """
        Get default fuzzy system parameters.
        
        Returns:
        --------
        dict
            Dictionary containing membership function parameters and fuzzy rules.
            
        Structure:
        ----------
        {
            "prob": {
                "low": [mu, sigma],
                "mid": [mu, sigma],
                "high": [mu, sigma]
            },
            "contrast": {...},
            "entropy": {...},
            "rules": [
                {
                    "if": {"prob": "high", "contrast": "low", "entropy": "low"},
                    "out": 1.0,
                    "weight": 1.0
                },
                ...
            ],
            "threshold": 0.5
        }
        
        Example:
        --------
        >>> params = FuzzyBinarizer.default_params()
        >>> print(params['prob']['low'])
        [0.2, 0.15]
        """
        return {
            # Probability membership functions (from neural network)
            "prob": {
                "low": [0.2, 0.15],      # mu=0.2, sigma=0.15
                "mid": [0.5, 0.2],       # mu=0.5, sigma=0.2
                "high": [0.8, 0.15]      # mu=0.8, sigma=0.15
            },
            
            # Contrast membership functions (local std deviation)
            "contrast": {
                "low": [0.05, 0.03],     # Low contrast (uniform areas)
                "mid": [0.15, 0.05],     # Medium contrast
                "high": [0.3, 0.1]       # High contrast (edges)
            },
            
            # Entropy membership functions (local information content)
            "entropy": {
                "low": [0.3, 0.2],       # Low entropy (uniform)
                "mid": [1.5, 0.5],       # Medium entropy
                "high": [2.5, 0.5]       # High entropy (complex)
            },
            
            # Fuzzy rules: IF prob AND contrast AND entropy THEN output
            "rules": [
                # High probability cases (likely foreground)
                {"if": {"prob": "high", "contrast": "low", "entropy": "low"}, 
                 "out": 1.0, "weight": 1.0},
                {"if": {"prob": "high", "contrast": "mid", "entropy": "low"}, 
                 "out": 1.0, "weight": 1.0},
                {"if": {"prob": "high", "contrast": "high", "entropy": "mid"}, 
                 "out": 1.0, "weight": 1.0},
                
                # Medium probability cases (uncertain)
                {"if": {"prob": "mid", "contrast": "high", "entropy": "high"}, 
                 "out": 1.0, "weight": 0.8},
                {"if": {"prob": "mid", "contrast": "mid", "entropy": "mid"}, 
                 "out": 0.5, "weight": 0.6},
                {"if": {"prob": "mid", "contrast": "low", "entropy": "low"}, 
                 "out": 0.0, "weight": 0.8},
                
                # Low probability cases (likely background)
                {"if": {"prob": "low", "contrast": "low", "entropy": "low"}, 
                 "out": 0.0, "weight": 1.0},
                {"if": {"prob": "low", "contrast": "mid", "entropy": "low"}, 
                 "out": 0.0, "weight": 1.0},
                {"if": {"prob": "low", "contrast": "high", "entropy": "mid"}, 
                 "out": 0.0, "weight": 0.9},
                
                # Edge cases
                {"if": {"prob": "mid", "contrast": "high", "entropy": "low"}, 
                 "out": 0.8, "weight": 0.7},
                {"if": {"prob": "high", "contrast": "low", "entropy": "high"}, 
                 "out": 0.9, "weight": 0.8},
            ],
            
            # Final threshold for binary output
            "threshold": 0.5
        }
    
    @staticmethod
    def load_params(path: str) -> dict:
        """
        Load fuzzy system parameters from JSON file.
        
        Parameters:
        -----------
        path : str
            Path to JSON file containing parameters
            
        Returns:
        --------
        dict
            Loaded parameters dictionary
            
        Example:
        --------
        >>> params = FuzzyBinarizer.load_params('fuzzy_params.json')
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_params(params: dict, path: str):
        """
        Save fuzzy system parameters to JSON file.
        
        Parameters:
        -----------
        params : dict
            Parameters dictionary to save
        path : str
            Output JSON file path
            
        Example:
        --------
        >>> params = FuzzyBinarizer.default_params()
        >>> FuzzyBinarizer.save_params(params, 'my_params.json')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
    
    @staticmethod
    def _gaussian_membership(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        Compute Gaussian membership function.
        
        Parameters:
        -----------
        x : np.ndarray
            Input values
        mu : float
            Mean (center) of Gaussian
        sigma : float
            Standard deviation (width)
            
        Returns:
        --------
        np.ndarray
            Membership values in [0, 1]
        """
        return np.exp(-0.5 * ((x - mu) / (sigma + 1e-10)) ** 2)
    
    def _compute_local_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Compute local contrast (standard deviation) over sliding window.
        
        Parameters:
        -----------
        image : np.ndarray
            Grayscale image, shape (H, W)
            
        Returns:
        --------
        np.ndarray
            Local contrast map, shape (H, W)
        """
        if NUMBA_AVAILABLE:
            return self._compute_local_contrast_numba(image, self.window_size)
        else:
            return self._compute_local_contrast_scipy(image, self.window_size)
    
    @staticmethod
    def _compute_local_contrast_scipy(image: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute local contrast using scipy (slower but no numba dependency).
        """
        def local_std(window):
            return np.std(window)
        
        contrast = generic_filter(image, local_std, size=window_size, mode='reflect')
        return contrast.astype(np.float32)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_local_contrast_numba(image: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute local contrast using numba (faster).
        """
        h, w = image.shape
        contrast = np.zeros_like(image, dtype=np.float32)
        half_win = window_size // 2
        
        for i in range(h):
            for j in range(w):
                # Define window bounds
                i_min = max(0, i - half_win)
                i_max = min(h, i + half_win + 1)
                j_min = max(0, j - half_win)
                j_max = min(w, j + half_win + 1)
                
                # Extract window
                window = image[i_min:i_max, j_min:j_max]
                
                # Compute std
                contrast[i, j] = np.std(window)
        
        return contrast
    
    def _compute_local_entropy(self, image: np.ndarray) -> np.ndarray:
        """
        Compute local Shannon entropy over sliding window.
        
        Parameters:
        -----------
        image : np.ndarray
            Grayscale image, shape (H, W)
            
        Returns:
        --------
        np.ndarray
            Local entropy map, shape (H, W)
        """
        # Quantize to reduce computation
        quantized = (image * 255).astype(np.uint8)
        
        def local_entropy(window):
            # Compute histogram
            hist, _ = np.histogram(window, bins=32, range=(0, 256))
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) <= 1:
                return 0.0
            # Normalize and compute entropy
            prob = hist / hist.sum()
            return -np.sum(prob * np.log2(prob + 1e-10))
        
        entropy_map = generic_filter(quantized, local_entropy, 
                                     size=self.window_size, mode='reflect')
        return entropy_map.astype(np.float32)
    
    def _fuzzify(self, value: np.ndarray, memberships: dict) -> dict:
        """
        Compute membership degrees for a given input.
        
        Parameters:
        -----------
        value : np.ndarray
            Input values (e.g., prob, contrast, entropy)
        memberships : dict
            Membership function parameters {"low": [mu, sigma], ...}
            
        Returns:
        --------
        dict
            Membership degrees {"low": array, "mid": array, "high": array}
        """
        result = {}
        for label, (mu, sigma) in memberships.items():
            result[label] = self._gaussian_membership(value, mu, sigma)
        return result
    
    def _evaluate_rules(self, 
                       prob_fuzzy: dict, 
                       contrast_fuzzy: dict, 
                       entropy_fuzzy: dict, 
                       rules: List[dict]) -> np.ndarray:
        """
        Evaluate fuzzy rules using Mamdani-style inference.
        
        Parameters:
        -----------
        prob_fuzzy : dict
            Fuzzified probability values
        contrast_fuzzy : dict
            Fuzzified contrast values
        entropy_fuzzy : dict
            Fuzzified entropy values
        rules : list
            List of fuzzy rules
            
        Returns:
        --------
        np.ndarray
            Defuzzified output (crisp values in [0, 1])
        """
        h, w = list(prob_fuzzy.values())[0].shape
        output = np.zeros((h, w), dtype=np.float32)
        total_weight = np.zeros((h, w), dtype=np.float32)
        
        for rule in rules:
            conditions = rule["if"]
            rule_output = rule["out"]
            rule_weight = rule["weight"]
            
            # Compute rule activation (min for AND operation)
            activation = np.minimum(
                np.minimum(
                    prob_fuzzy[conditions["prob"]],
                    contrast_fuzzy[conditions["contrast"]]
                ),
                entropy_fuzzy[conditions["entropy"]]
            )
            
            # Weight the activation
            weighted_activation = activation * rule_weight
            
            # Accumulate weighted outputs (Mamdani aggregation)
            output += weighted_activation * rule_output
            total_weight += weighted_activation
        
        # Defuzzification: weighted average (centroid approximation)
        defuzzified = np.divide(output, total_weight, 
                               out=np.zeros_like(output),
                               where=total_weight > 1e-10)
        
        return defuzzified
    
    def infer(self, 
             orig: np.ndarray, 
             prob_map: np.ndarray, 
             params: Optional[dict] = None,
             verbose: bool = False) -> np.ndarray:
        """
        Perform fuzzy inference for document binarization.
        
        This method:
        1. Computes local features (contrast, entropy) from original image
        2. Fuzzifies inputs (prob, contrast, entropy) using Gaussian memberships
        3. Evaluates fuzzy rules using Mamdani-style inference
        4. Defuzzifies output using weighted average
        5. Applies threshold to produce binary image
        
        Parameters:
        -----------
        orig : np.ndarray
            Original grayscale image, float32 [0..1], shape (H, W)
        prob_map : np.ndarray
            Neural network probability map, float32 [0..1], shape (H, W)
        params : dict, optional
            Fuzzy system parameters. If None, uses default_params()
            
        Returns:
        --------
        np.ndarray
            Binary image, uint8 (0 or 255), shape (H, W)
            
        Example:
        --------
        >>> fuzzy = FuzzyBinarizer()
        >>> orig = np.random.rand(256, 256).astype(np.float32)
        >>> prob = np.random.rand(256, 256).astype(np.float32)
        >>> binary = fuzzy.infer(orig, prob)
        >>> print(binary.shape, binary.dtype)
        (256, 256) uint8
        """
        # Validate inputs
        assert orig.ndim == 2, "orig must be 2D grayscale image"
        assert prob_map.ndim == 2, "prob_map must be 2D"
        assert orig.shape == prob_map.shape, "orig and prob_map must have same shape"
        assert orig.dtype in [np.float32, np.float64], "orig must be float32 or float64"
        assert prob_map.dtype in [np.float32, np.float64], "prob_map must be float"
        
        # Use default params if none provided
        if params is None:
            params = self.default_params()
        
        # Ensure inputs are in [0, 1]
        orig = np.clip(orig, 0.0, 1.0).astype(np.float32)
        prob_map = np.clip(prob_map, 0.0, 1.0).astype(np.float32)
        
        if verbose:
            print(f"Computing local features (window size: {self.window_size})...")
        
        # Compute local features
        contrast = self._compute_local_contrast(orig)
        entropy = self._compute_local_entropy(orig)
        
        # Normalize features to [0, 1]
        contrast = np.clip(contrast / (contrast.max() + 1e-10), 0.0, 1.0)
        entropy = np.clip(entropy / (entropy.max() + 1e-10), 0.0, 1.0)
        
        if verbose:
            print("Fuzzifying inputs...")
        
        # Fuzzify inputs
        prob_fuzzy = self._fuzzify(prob_map, params["prob"])
        contrast_fuzzy = self._fuzzify(contrast, params["contrast"])
        entropy_fuzzy = self._fuzzify(entropy, params["entropy"])
        
        if verbose:
            print(f"Evaluating {len(params['rules'])} fuzzy rules...")
        
        # Evaluate fuzzy rules
        fuzzy_output = self._evaluate_rules(
            prob_fuzzy, contrast_fuzzy, entropy_fuzzy, params["rules"]
        )
        
        if verbose:
            print("Applying threshold...")
        
        # Apply threshold to get binary output
        threshold = params.get("threshold", 0.5)
        binary = (fuzzy_output > threshold).astype(np.uint8) * 255
        
        if verbose:
            print(f"✅ Fuzzy inference complete. Output range: [{binary.min()}, {binary.max()}]")
        
        return binary


# Test example
if __name__ == "__main__":
    print("=" * 80)
    print("Fuzzy Binarization System - Test Run")
    print("=" * 80)
    
    # Create test data
    print("\n📊 Creating test data...")
    np.random.seed(42)
    
    # Simulate a document with text (higher prob in center)
    H, W = 256, 256
    orig = np.random.rand(H, W).astype(np.float32) * 0.3 + 0.7  # Light background
    
    # Add some "text" regions (darker)
    y, x = np.ogrid[:H, :W]
    text_mask = ((x - W//2)**2 + (y - H//2)**2) < (W//4)**2
    orig[text_mask] *= 0.4  # Darken text region
    
    # Simulate neural network probability map (high where text is)
    prob_map = np.zeros((H, W), dtype=np.float32)
    prob_map[text_mask] = 0.8 + np.random.rand(text_mask.sum()) * 0.2
    prob_map[~text_mask] = 0.1 + np.random.rand((~text_mask).sum()) * 0.2
    
    print(f"   Original image shape: {orig.shape}, dtype: {orig.dtype}")
    print(f"   Original range: [{orig.min():.3f}, {orig.max():.3f}]")
    print(f"   Prob map shape: {prob_map.shape}, dtype: {prob_map.dtype}")
    print(f"   Prob range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # Initialize fuzzy system
    print("\n🔧 Initializing FuzzyBinarizer...")
    fuzzy = FuzzyBinarizer(window_size=9)
    
    # Get default parameters
    print("\n📋 Loading default parameters...")
    params = fuzzy.default_params()
    print(f"   Membership functions: prob, contrast, entropy")
    print(f"   Number of fuzzy rules: {len(params['rules'])}")
    print(f"   Threshold: {params['threshold']}")
    
    # Test parameter save/load
    print("\n💾 Testing parameter save/load...")
    test_params_path = "test_fuzzy_params.json"
    fuzzy.save_params(params, test_params_path)
    loaded_params = fuzzy.load_params(test_params_path)
    print(f"   ✅ Saved and loaded parameters to {test_params_path}")
    
    # Run inference
    print("\n🚀 Running fuzzy inference...")
    print("-" * 80)
    binary = fuzzy.infer(orig, prob_map, params)
    print("-" * 80)
    
    # Display results
    print("\n📊 Results:")
    print(f"   Binary output shape: {binary.shape}")
    print(f"   Binary output dtype: {binary.dtype}")
    print(f"   Binary unique values: {np.unique(binary)}")
    print(f"   Foreground pixels: {(binary == 255).sum()} ({(binary == 255).sum() / binary.size * 100:.1f}%)")
    print(f"   Background pixels: {(binary == 0).sum()} ({(binary == 0).sum() / binary.size * 100:.1f}%)")
    
    # Performance info
    if NUMBA_AVAILABLE:
        print("\n⚡ Numba optimization: ENABLED (fast mode)")
    else:
        print("\n⚠️  Numba optimization: DISABLED (using scipy fallback)")
        print("   Install numba for better performance: pip install numba")
    
    print("\n" + "=" * 80)
    print("✅ Fuzzy binarization test completed successfully!")
    print("=" * 80)
    
    # Cleanup
    import os
    if os.path.exists(test_params_path):
        os.remove(test_params_path)
        print(f"\n🗑️  Cleaned up test file: {test_params_path}")
