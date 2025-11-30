# Fuzzy Logic Document Binarization

## Overview

This module implements a **Mamdani-style fuzzy inference system** that enhances document binarization by combining neural network probability outputs with local image features.

## Features

- ✅ **Gaussian membership functions** for prob, contrast, and entropy
- ✅ **11 configurable fuzzy rules** with individual weights
- ✅ **Local feature extraction**: contrast (std dev) and entropy
- ✅ **Mamdani-style inference** with MIN operator for AND
- ✅ **Centroid defuzzification** using weighted average
- ✅ **Numba optimization** for fast local computations (optional)
- ✅ **JSON parameter** save/load functionality

## Architecture

```
Input: Original Image + Neural Network Probability Map
   ↓
Local Features Computation (9×9 window)
   ├─ Contrast (local std deviation)
   └─ Entropy (Shannon entropy)
   ↓
Fuzzification (Gaussian membership functions)
   ├─ Probability: {low, mid, high}
   ├─ Contrast: {low, mid, high}
   └─ Entropy: {low, mid, high}
   ↓
Fuzzy Inference (11 weighted rules)
   - MIN operator for AND
   - Weighted aggregation
   ↓
Defuzzification (centroid approximation)
   ↓
Thresholding → Binary Output
```

## Installation

```bash
# Required dependencies
pip install numpy scipy

# Optional (for 10x speed improvement)
pip install numba

# For hybrid inference
pip install torch torchvision pillow
```

## Quick Start

### 1. Standalone Fuzzy System

```python
import numpy as np
from src.fuzzy.fuzzy_system import FuzzyBinarizer

# Initialize
fuzzy = FuzzyBinarizer(window_size=9)

# Create sample inputs
orig = np.random.rand(256, 256).astype(np.float32)
prob_map = np.random.rand(256, 256).astype(np.float32)

# Run inference
binary = fuzzy.infer(orig, prob_map)

print(f"Output: {binary.shape}, dtype: {binary.dtype}")
# Output: (256, 256), dtype: uint8
```

### 2. Hybrid Neural + Fuzzy

```bash
# Using default fuzzy parameters
python hybrid_inference.py --checkpoint best_model.pth \
                           --input document.png \
                           --output result_hybrid.png \
                           --mode hybrid

# Using custom fuzzy parameters
python hybrid_inference.py --checkpoint best_model.pth \
                           --input document.png \
                           --output result_hybrid.png \
                           --mode hybrid \
                           --fuzzy-params custom_params.json
```

### 3. Compare Modes

```bash
# Neural network only
python hybrid_inference.py --checkpoint best_model.pth \
                           --input document.png \
                           --output result_neural.png \
                           --mode neural

# Hybrid approach
python hybrid_inference.py --checkpoint best_model.pth \
                           --input document.png \
                           --output result_hybrid.png \
                           --mode hybrid
```

## Parameters

### Default Fuzzy Parameters

```python
params = {
    # Probability membership functions
    "prob": {
        "low": [0.2, 0.15],    # mu=0.2, sigma=0.15
        "mid": [0.5, 0.2],     # mu=0.5, sigma=0.2
        "high": [0.8, 0.15]    # mu=0.8, sigma=0.15
    },
    
    # Contrast membership functions
    "contrast": {
        "low": [0.05, 0.03],   # Low contrast (uniform areas)
        "mid": [0.15, 0.05],   # Medium contrast
        "high": [0.3, 0.1]     # High contrast (edges)
    },
    
    # Entropy membership functions
    "entropy": {
        "low": [0.3, 0.2],     # Low entropy (uniform)
        "mid": [1.5, 0.5],     # Medium entropy
        "high": [2.5, 0.5]     # High entropy (complex)
    },
    
    # Fuzzy rules (11 rules)
    "rules": [
        # IF prob=high AND contrast=low AND entropy=low THEN output=1.0 (weight=1.0)
        {"if": {"prob": "high", "contrast": "low", "entropy": "low"}, 
         "out": 1.0, "weight": 1.0},
        # ... more rules ...
    ],
    
    # Final threshold
    "threshold": 0.5
}
```

### Customize Parameters

```python
from src.fuzzy.fuzzy_system import FuzzyBinarizer

# Get default parameters
params = FuzzyBinarizer.default_params()

# Customize
params["threshold"] = 0.6
params["prob"]["high"] = [0.9, 0.1]  # Adjust high probability

# Add custom rule
params["rules"].append({
    "if": {"prob": "mid", "contrast": "mid", "entropy": "high"},
    "out": 0.7,
    "weight": 0.9
})

# Save for reuse
FuzzyBinarizer.save_params(params, "custom_params.json")

# Use in inference
fuzzy = FuzzyBinarizer()
binary = fuzzy.infer(orig, prob_map, params)
```

## API Reference

### FuzzyBinarizer Class

#### Methods

**`__init__(window_size=9)`**
- Initialize fuzzy binarizer
- `window_size`: Size of sliding window for local features (default: 9)

**`default_params() -> dict`** (static)
- Returns default fuzzy system parameters
- Includes membership functions, rules, and threshold

**`load_params(path: str) -> dict`** (static)
- Load parameters from JSON file

**`save_params(params: dict, path: str)`** (static)
- Save parameters to JSON file

**`infer(orig, prob_map, params=None) -> np.ndarray`**
- Main inference method
- `orig`: Grayscale image [0, 1], shape (H, W), dtype float32
- `prob_map`: Neural network output [0, 1], shape (H, W), dtype float32
- `params`: Optional custom parameters (uses defaults if None)
- Returns: Binary image (0 or 255), shape (H, W), dtype uint8

## Fuzzy Rules Explained

Each rule follows the format:

```
IF (prob = X) AND (contrast = Y) AND (entropy = Z)
THEN output = O WITH weight = W
```

**Example rules:**

1. **High confidence foreground:**
   - `IF prob=high AND contrast=low AND entropy=low THEN output=1.0`
   - Meaning: Neural network is confident (high prob), area is uniform (low contrast/entropy) → definitely foreground

2. **Uncertain regions:**
   - `IF prob=mid AND contrast=mid AND entropy=mid THEN output=0.5`
   - Meaning: Neural network uncertain, medium texture → borderline case

3. **High confidence background:**
   - `IF prob=low AND contrast=low AND entropy=low THEN output=0.0`
   - Meaning: Neural network says background (low prob), uniform area → definitely background

## Performance

### Speed Comparison

| Configuration | Time per 256×256 image | Speedup |
|--------------|------------------------|---------|
| Scipy (no numba) | ~450ms | 1.0x |
| Numba (JIT compiled) | ~45ms | 10.0x |

**Recommendation:** Install `numba` for 10x faster inference!

```bash
pip install numba
```

### Memory Usage

- **Fuzzy system:** ~5MB per 256×256 image
- **Total (Neural + Fuzzy):** ~150MB (model: ~145MB, fuzzy: ~5MB)

## Examples

### Example 1: Basic Usage

```python
from src.fuzzy.fuzzy_system import FuzzyBinarizer
import numpy as np

# Initialize
fuzzy = FuzzyBinarizer()

# Load images (your code here)
orig = load_grayscale_image("document.png")  # [0, 1]
prob_map = neural_network_output  # [0, 1]

# Infer
binary = fuzzy.infer(orig, prob_map)

# Save
save_image(binary, "result.png")
```

### Example 2: Custom Parameters

```python
# Adjust for high-contrast documents
params = FuzzyBinarizer.default_params()
params["contrast"]["high"] = [0.4, 0.15]  # More sensitive to edges
params["threshold"] = 0.45  # Lower threshold

fuzzy = FuzzyBinarizer()
binary = fuzzy.infer(orig, prob_map, params)
```

### Example 3: Batch Processing

```python
from pathlib import Path
from PIL import Image
import numpy as np

fuzzy = FuzzyBinarizer()

for img_path in Path("documents/").glob("*.png"):
    # Load
    orig = np.array(Image.open(img_path).convert('L')).astype(np.float32) / 255
    prob_map = get_neural_prob_map(orig)
    
    # Process
    binary = fuzzy.infer(orig, prob_map)
    
    # Save
    Image.fromarray(binary).save(f"results/{img_path.stem}_fuzzy.png")
```

## Testing

Run the built-in test:

```bash
python src/fuzzy/fuzzy_system.py
```

Expected output:
```
================================================================================
Fuzzy Binarization System - Test Run
================================================================================

📊 Creating test data...
   Original image shape: (256, 256), dtype: float32
   ...

✅ Fuzzy binarization test completed successfully!
================================================================================
```

## Troubleshooting

### Issue: Slow performance

**Solution:** Install numba
```bash
pip install numba
```

### Issue: ImportError for numba

**Solution:** Fallback to scipy (slower but works)
- The system automatically uses scipy if numba is not available
- No code changes needed

### Issue: Poor binarization quality

**Solutions:**
1. Adjust membership function parameters
2. Modify rule weights
3. Change final threshold
4. Tune window size for feature extraction

### Issue: Memory error on large images

**Solution:** Process in patches
```python
# Split large image into patches
patches = split_image_into_patches(large_image, patch_size=256)
results = [fuzzy.infer(patch, prob_patch) for patch, prob_patch in patches]
final = reconstruct_from_patches(results)
```

## Citation

If you use this fuzzy system in your research, please cite:

```
@software{fuzzy_binarization,
  title={Fuzzy Logic Document Binarization System},
  author={Document Binarization Project},
  year={2025},
  url={https://github.com/shreyat81/document-binarization}
}
```

## License

This module is part of the Document Binarization project and follows the same license.

---

**For more information, see:**
- Main project: `README.md`
- Neural network: `model.py`
- Hybrid inference: `hybrid_inference.py`
