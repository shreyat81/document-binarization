# Document Binarization — Results Summary

This repository contains a document binarization model (EfficientNet-B0 encoder + U-Net decoder) and results produced by post-training analysis using the Whale Optimization Algorithm (WOA) to fine-tune the binarization threshold.

For a concise, reproducible collection of evaluation figures and sample comparisons suitable for inclusion in your research paper, open the Jupyter notebook:

```
results_analysis/Results_Analysis_Notebook.ipynb
```

The `results_analysis/` directory contains ready-to-use figures, sample visualizations, and machine-readable metrics (JSON/CSV).

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| **F1 Score** | 0.9907 |
| **Precision** | 0.9890 |
| **Recall** | 0.9924 |
| **Accuracy** | 0.9836 |

**Test Dataset**: 1,037 document patches from DIBCO (2009-2017)

## 📁 Project Structure

```
.
├── model.py                      # FastBinarizationModel (4.4M params)
├── dataset.py                    # Dataset loader
├── inference.py                  # Main inference script
├── demo_inference.py             # Demo with visualizations
├── best_model.pth                # Trained model checkpoint
├── test_evaluation_results.json  # Test set metrics
├── requirements.txt              # Python dependencies
├── src/
│   └── woa_optimize.py           # Whale Optimization Algorithm
└── split/                        # Preprocessed dataset
    ├── train/                    # 3,631 training patches
    ├── val/                      # 520 validation patches
    └── test/                     # 1,037 test patches
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

**Evaluate on test set:**
```bash
python inference.py --checkpoint best_model.pth \
                   --input split/test/images \
                   --gt_dir split/test/gt \
                   --mode evaluate
```

**Process single image:**
```bash
python inference.py --checkpoint best_model.pth \
                   --input document.png \
                   --output binarized.png
```

**Batch processing:**
```bash
python inference.py --checkpoint best_model.pth \
                   --input documents/ \
                   --output results/
```

**Demo with visualization:**
```bash
python demo_inference.py
```

## 🐋 Whale Optimization Algorithm (WOA)

This project uses **Whale Optimization Algorithm** for optimizing binarization thresholds - a nature-inspired metaheuristic algorithm that mimics humpback whale hunting behavior.

### Benefits of WOA:
- **Adaptive threshold finding** for different document types
- **Better generalization** across varied image conditions
- **Nature-inspired optimization** - no gradient required
- **Fast convergence** for threshold optimization

### Run WOA Optimization:
```bash
# Quick optimization (fast, 2-3 minutes)
python src/woa_optimize.py --mode quick \
                           --checkpoint best_model.pth \
                           --images split/val/images \
                           --gt split/val/gt

# Full optimization (better results, 10-15 minutes)
python src/woa_optimize.py --mode full \
                           --checkpoint best_model.pth \
                           --images split/val/images \
                           --gt split/val/gt \
                           --output optimized_threshold.json
```

## 🔧 Advanced Usage

### Custom threshold
```bash
python inference.py --checkpoint best_model.pth \
                   --input image.png \
                   --output result.png \
                   --threshold 0.7
```

### Python API
```python
from inference import DocumentBinarizer

# Initialize
binarizer = DocumentBinarizer('best_model.pth', device='cpu')

# Predict
binary = binarizer.predict('document.png')

# Evaluate
metrics = binarizer.evaluate_on_test_set(
    'split/test/images',
    'split/test/gt'
)
```

## 📊 Model Architecture

- **Encoder**: EfficientNet-B0 (pretrained on ImageNet)
- **Decoder**: Lightweight U-Net with GroupNorm
- **Parameters**: 4,448,707 (< 5M)
- **Input**: Grayscale images (256×256 patches)
- **Output**: Binary document images

## 🎓 Training Details

- **Framework**: PyTorch 2.7.1
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-5)
- **Loss**: Combined BCE + Dice Loss (0.5 each)
- **Scheduler**: Cosine Annealing
- **Batch Size**: 64 (GPU) / 16 (CPU)
- **Epochs**: 15 (with early stopping)
- **Training Time**: ~20-30 minutes (Kaggle GPU T4)
- **Validation F1**: 0.9901

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- tqdm

## 📝 Citation

If you use this model, please cite the DIBCO benchmark:

```
DIBCO: Document Image Binarization Contest (2009-2017)
International Conference on Document Analysis and Recognition
```

## 📄 License

This project is for educational and research purposes.

---

**Model Status**: ✅ Production Ready  
**Last Updated**: November 2025  
**Performance**: State-of-the-art (99.07% F1)
