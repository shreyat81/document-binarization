# Large Files - Download Instructions

Due to GitHub's file size limitations, the following files are not included in this repository:

## 📦 Model Checkpoint

**File**: `best_model.pth` (51.4 MB)

**Download from**:
- Kaggle: [Your Kaggle notebook output]
- Google Drive: [Upload and share link]
- Hugging Face: [Upload to model hub]

**After downloading**, place it in the project root directory:
```bash
mv best_model.pth "/Users/shreyatiwari/Documents/Soft Computing Project/"
```

## 📊 Dataset

**File**: `dibco-dataset.zip` (267.8 MB)

The preprocessed DIBCO dataset is already in the `split/` folder.

If you need the original dataset:
- DIBCO 2009-2017: http://dibco.univ-lr.fr/

## 🚀 Quick Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/shreyat81/document-binarization.git
   cd document-binarization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download `best_model.pth` from Kaggle or the link above

4. Run inference:
   ```bash
   python inference.py --checkpoint best_model.pth \
                      --input split/test/images \
                      --gt_dir split/test/gt \
                      --mode evaluate
   ```

## 📈 Model Performance

Already evaluated on test set (included in repo):
- F1 Score: **0.9907** (99.07%)
- Precision: 0.9890
- Recall: 0.9924
- Accuracy: 0.9836

See `test_evaluation_results.json` for detailed metrics.

## 💾 Alternative Hosting Options

**For the model file** (`best_model.pth`):
1. **Kaggle**: Already on Kaggle as notebook output
2. **Hugging Face Hub**: Best for ML models - https://huggingface.co/
3. **Google Drive**: Easy sharing
4. **Git LFS**: For version control (requires setup)

**Recommended**: Upload to Hugging Face Model Hub for permanent hosting and easy sharing.
