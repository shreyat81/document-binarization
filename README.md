# Document Binarization using Deep Learning

A comprehensive document image binarization system using **EfficientNet-B0** and **InceptionNet-V3** feature extractors with UNet-style decoder, fuzzy logic refinement, and whale optimization-inspired threshold selection.

## 📋 Overview

This project implements a state-of-the-art document binarization pipeline based on the architecture:

**Input → Preprocessing → EfficientNet + InceptionNet → Feature Fusion → UNet Decoder → Probability Map → Fuzzy System → Whale Optimization → Final Binarized Output**

## 🏗️ Architecture Components

### 1. **Feature Extraction**
- **EfficientNet-B0**: Extracts local features at multiple scales
- **InceptionNet-V3**: Captures multi-scale features with inception modules

### 2. **Feature Fusion**
- Combines features from both networks at multiple scales
- Adaptive channel and spatial alignment

### 3. **UNet-Style Decoder**
- Skip connections from encoder features
- Attention gates for selective feature propagation
- Progressive upsampling to original resolution

### 4. **Probability Map Generation**
- Converts logits to probability values [0, 1]

### 5. **Fuzzy Logic System**
- Learnable fuzzy membership functions
- Applies fuzzy rules for refinement

### 6. **Whale Optimization Algorithm**
- Adaptive threshold selection
- Learnable threshold parameters

## 📁 Project Structure

```
.
├── model.py                    # Complete model architecture
├── train.py                    # Training script
├── inference.py                # Inference and evaluation
├── grayscale.py                # Convert images to grayscale
├── normalization.py            # Normalize images
├── resize_to_patch.py          # Extract patches from images
├── test_&_train.py             # Split dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone or navigate to the project directory**

```bash
cd "Soft Computing Project"
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Data Preparation Pipeline

The project uses DIBCO (Document Image Binarization Competition) datasets. Follow these steps:

#### Step 1: Organize DIBCO Data
Place your DIBCO datasets in the `DIBCO/` folder with the existing structure.

#### Step 2: Grayscale Conversion
```bash
python grayscale.py
```
- **Input**: `DIBCO_CLEAN/images` and `DIBCO_CLEAN/gt`
- **Output**: `grey_scale/images` and `grey_scale/gt`

#### Step 3: Normalization
```bash
python normalization.py
```
- **Input**: `grey_scale/images` and `grey_scale/gt`
- **Output**: `normalization/images` and `normalization/gt` (as .npy files)
- Normalizes images to [0, 1] range
- Converts GT to binary [0, 1]

#### Step 4: Patch Extraction
```bash
python resize_to_patch.py
```
- **Input**: `normalization/images` and `normalization/gt`
- **Output**: `resize_patch/images` and `resize_patch/gt`
- Creates 256×256 patches with 128-pixel stride (50% overlap)
- Filters out patches with insufficient text content

#### Step 5: Train/Val/Test Split
```bash
python test_&_train.py
```
- **Input**: `resize_patch/images` and `resize_patch/gt`
- **Output**: `split/train`, `split/val`, `split/test`
- Split ratio: 70% train, 20% test, 10% validation

## 🎯 Training

### Basic Training

```bash
python train.py
```

### Configuration

Edit the `config` dictionary in `train.py`:

```python
config = {
    # Data paths
    'train_images_dir': './split/train/images',
    'train_gt_dir': './split/train/gt',
    'val_images_dir': './split/val/images',
    'val_gt_dir': './split/val/gt',
    
    # Training parameters
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    
    # Loss weights
    'bce_weight': 1.0,      # Binary Cross Entropy
    'dice_weight': 1.0,     # Dice Loss
    'edge_weight': 0.5,     # Edge-aware Loss
    
    # Model
    'pretrained': True,
    
    # Checkpoints
    'checkpoint_dir': './checkpoints',
    'save_every': 5
}
```

### Training Features

- **Combined Loss Function**: BCE + Dice + Edge-aware loss
- **Data Augmentation**: Flips and rotations
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Saves best model and periodic checkpoints
- **Metrics Tracking**: Accuracy, Precision, Recall, F1, IoU

## 🔮 Inference

### Single Image Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pth \
                    --input image.png \
                    --output ./results \
                    --mode single
```

### Batch Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pth \
                    --input ./test_images/ \
                    --output ./results \
                    --mode batch \
                    --extension .npy
```

### Visualization Mode

```bash
python inference.py --checkpoint checkpoints/best_model.pth \
                    --input image.png \
                    --mode visualize
```

Shows all intermediate outputs:
- Original image
- Probability map
- Fuzzy system output
- Adaptive threshold map
- Final binarized result

### Evaluation Mode

```bash
python inference.py --checkpoint checkpoints/best_model.pth \
                    --input ./split/test/images \
                    --gt_dir ./split/test/gt \
                    --mode evaluate
```

Computes metrics on test set:
- Accuracy
- Precision
- Recall
- F1 Score
- IoU (Intersection over Union)

## 📊 Model Architecture Details

### EfficientNet-B0 Feature Extractor
- Input: RGB image (3, H, W)
- Outputs: Multi-scale features at 5 different resolutions
- Channels: [16, 24, 40, 112, 320]

### InceptionNet-V3 Feature Extractor
- Input: RGB image (3, H, W)
- Outputs: Multi-scale features at 5 different resolutions
- Channels: [64, 192, 288, 768, 2048]

### Feature Fusion Modules
- Fuses corresponding scale features from both networks
- Output channels: [64, 128, 256, 512, 1024]

### UNet Decoder
- 4 decoder blocks with skip connections
- Attention gates for feature selection
- Progressive upsampling

### Output
- Single-channel binary segmentation mask
- Final size matches input size

## 🎓 Model Testing

Test the model architecture without training:

```bash
python model.py
```

This will:
- Create the complete pipeline
- Run a forward pass with dummy data
- Print all output shapes
- Count total parameters

Expected output:
```
Model Architecture Test:
Input shape: torch.Size([2, 3, 256, 256])
Output shape: torch.Size([2, 1, 256, 256])
...
✅ Model architecture test passed!
Total parameters: ~XX,XXX,XXX
```

## 📈 Performance Metrics

The model is evaluated using:
- **Binary Cross-Entropy Loss**: Pixel-wise classification
- **Dice Loss**: Overlap-based segmentation metric
- **Edge Loss**: Preserves text boundaries
- **F1 Score**: Harmonic mean of precision and recall
- **IoU**: Intersection over Union

## 💡 Key Features

1. **Multi-Scale Feature Extraction**: Combines EfficientNet and InceptionNet
2. **Attention Mechanism**: Selective feature propagation
3. **Fuzzy Logic**: Learnable fuzzy refinement
4. **Adaptive Thresholding**: Whale optimization-inspired approach
5. **Edge Preservation**: Edge-aware loss function
6. **Data Augmentation**: Improves generalization
7. **Comprehensive Evaluation**: Multiple metrics

## 🔧 Customization

### Adjust Patch Size
In `resize_to_patch.py`:
```python
PATCH_SIZE = 256  # Change to desired size
STRIDE = 128      # Adjust overlap
```

### Modify Model Architecture
In `model.py`:
- Change decoder channels
- Adjust attention gate parameters
- Modify fuzzy membership functions

### Tune Loss Weights
In `train.py` config:
```python
'bce_weight': 1.0,
'dice_weight': 1.0,
'edge_weight': 0.5,  # Adjust based on importance
```

## 📝 Citation

If you use this code or the DIBCO datasets, please cite the appropriate DIBCO competition papers.

## 🤝 Contributing

This is an academic project. For suggestions or improvements, feel free to:
1. Document issues
2. Propose enhancements
3. Submit modifications

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- DIBCO competition organizers for the datasets
- PyTorch team for the deep learning framework
- EfficientNet and InceptionNet authors
- UNet architecture pioneers

## 📞 Support

For questions or issues:
1. Check existing documentation
2. Review code comments
3. Test with smaller datasets first

---

**Happy Binarizing! 🎉**
