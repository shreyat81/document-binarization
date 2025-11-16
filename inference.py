"""
Inference Script for Document Binarization
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

from model import FastBinarizationModel


class DocumentBinarizer:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        
        print(f"📦 Loading model from {checkpoint_path}")
        self.model = FastBinarizationModel(pretrained=False)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        if 'val_f1' in checkpoint:
            print(f"   Validation F1: {checkpoint['val_f1']:.4f}")
        if 'epoch' in checkpoint:
            print(f"   Trained epochs: {checkpoint['epoch']}")
        
        print(f"   Device: {self.device}")
        print("✅ Model loaded successfully!\n")
    
    def predict(self, image_path, threshold=0.5):
        # Load image
        if str(image_path).endswith('.npy'):
            img_np = np.load(image_path)
            img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        else:
            img = Image.open(image_path).convert('L')
            img_np = np.array(img).astype(np.float32) / 255.0
        
        # Add dimensions
        if len(img_np.shape) == 2:
            img_np = img_np[np.newaxis, ...]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)
        
        # Binarize
        binary = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
        
        return binary
    
    def process_single_image(self, input_path, output_path, threshold=0.5):
        print(f"Processing: {input_path}")
        
        binary = self.predict(input_path, threshold)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(binary).save(output_path)
        print(f"✅ Saved to: {output_path}")
    
    def process_batch(self, input_dir, output_dir, threshold=0.5):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            if ext != '.npy':
                image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            print(f"❌ No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images in {input_dir}\n")
        
        for img_path in tqdm(image_files, desc="Processing images"):
            output_path = output_dir / f"{img_path.stem}_binarized.png"
            
            try:
                binary = self.predict(img_path, threshold)
                Image.fromarray(binary).save(output_path)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        
        print(f"\n✅ Batch processing complete! Results saved to: {output_dir}")
    
    def evaluate_on_test_set(self, images_dir, gt_dir, threshold=0.5):
        images_dir = Path(images_dir)
        gt_dir = Path(gt_dir)
        
        image_files = sorted(list(images_dir.glob('*.npy')))
        
        if len(image_files) == 0:
            print(f"❌ No .npy files found in {images_dir}")
            return
        
        print(f"Evaluating on {len(image_files)} test samples...\n")
        
        all_metrics = []
        
        for img_file in tqdm(image_files, desc="Evaluating"):
            # Load image
            img_np = np.load(img_file)
            img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
            
            if len(img_np.shape) == 2:
                img_np = img_np[np.newaxis, ...]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
            
            # Load GT
            base_name = img_file.stem
            if '_p' in base_name:
                gt_file = gt_dir / f"{base_name.replace('_p', '_GT_p')}.npy"
            else:
                gt_file = gt_dir / f"{base_name}_GT.npy"
            
            gt_np = np.load(gt_file)
            gt_np = np.clip(gt_np, 0.0, 1.0)
            
            # Predict
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Calculate metrics
            pred_binary = (probs > threshold).astype(np.float32)
            gt_binary = (gt_np > 0.5).astype(np.float32)
            
            tp = (pred_binary * gt_binary).sum()
            fp = (pred_binary * (1 - gt_binary)).sum()
            fn = ((1 - pred_binary) * gt_binary).sum()
            tn = ((1 - pred_binary) * (1 - gt_binary)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            
            all_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            })
        
        # Calculate averages
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1': np.mean([m['f1'] for m in all_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in all_metrics])
        }
        
        print(f"\n{'='*80}")
        print("📊 TEST SET EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Samples: {len(image_files)}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall:    {avg_metrics['recall']:.4f}")
        print(f"F1 Score:  {avg_metrics['f1']:.4f}")
        print(f"Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"{'='*80}\n")
        
        # Save results
        results_path = Path('test_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'average_metrics': avg_metrics,
                'per_sample_metrics': all_metrics,
                'num_samples': len(image_files)
            }, f, indent=2)
        
        print(f"📁 Detailed results saved to: {results_path}")
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Document Binarization Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='output',
                       help='Output path or directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binarization threshold')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--mode', type=str, default='predict',
                       choices=['predict', 'evaluate'],
                       help='Mode: predict or evaluate')
    parser.add_argument('--gt_dir', type=str, default=None,
                       help='Ground truth directory (for evaluate mode)')
    
    args = parser.parse_args()
    
    binarizer = DocumentBinarizer(args.checkpoint, device=args.device)
    
    input_path = Path(args.input)
    
    if args.mode == 'evaluate':
        if args.gt_dir is None:
            print("❌ Error: --gt_dir required for evaluation mode")
            return
        
        binarizer.evaluate_on_test_set(args.input, args.gt_dir, args.threshold)
    
    elif input_path.is_file():
        binarizer.process_single_image(args.input, args.output, args.threshold)
    
    elif input_path.is_dir():
        binarizer.process_batch(args.input, args.output, args.threshold)
    
    else:
        print(f"❌ Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()
