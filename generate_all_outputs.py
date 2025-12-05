"""
Generate binarization outputs for entire DIBCO test dataset
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from model import FastBinarizationModel

def process_all_test_samples(threshold=0.504, output_dir='dibco_outputs'):
    """
    Process all test samples from DIBCO dataset and save outputs.
    
    Args:
        threshold: Binarization threshold (default: 0.504 from WOA)
        output_dir: Directory to save all outputs
    """
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    (output_dir / 'original').mkdir(exist_ok=True)
    (output_dir / 'ground_truth').mkdir(exist_ok=True)
    (output_dir / 'probability_maps').mkdir(exist_ok=True)
    (output_dir / 'binarized').mkdir(exist_ok=True)
    (output_dir / 'composites').mkdir(exist_ok=True)
    
    # Load model
    print("="*70)
    print("📦 LOADING TRAINED MODEL")
    print("="*70)
    device = torch.device('cpu')
    model = FastBinarizationModel(pretrained=False)
    
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully")
    print(f"   Validation F1: {checkpoint.get('val_f1', 'N/A')}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Threshold: {threshold} (WOA-optimized)")
    print()
    
    # Get all test images
    test_images_dir = Path('split/test/images')
    test_gt_dir = Path('split/test/gt')
    
    image_files = sorted(list(test_images_dir.glob('*.npy')))
    
    print("="*70)
    print(f"📊 FOUND {len(image_files)} TEST SAMPLES")
    print("="*70)
    print(f"Images directory: {test_images_dir}")
    print(f"Ground truth directory: {test_gt_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    if len(image_files) == 0:
        print("❌ No test images found!")
        return
    
    # Process each image
    print("="*70)
    print("🔄 PROCESSING ALL TEST SAMPLES")
    print("="*70)
    
    results = []
    
    for idx, img_file in enumerate(tqdm(image_files, desc="Processing")):
        try:
            # Load image
            img_np = np.load(img_file)
            img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
            
            # Load ground truth
            base_name = img_file.stem
            if '_p' in base_name:
                gt_file = test_gt_dir / (base_name.replace('_p', '_GT_p') + '.npy')
            else:
                gt_file = test_gt_dir / (base_name + '_GT.npy')
            
            gt_np = np.load(gt_file)
            gt_np = np.clip(gt_np, 0.0, 1.0).astype(np.float32)
            
            # Prepare tensor
            if len(img_np.shape) == 2:
                img_np = img_np[np.newaxis, ...]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits)
            
            prob_map = probs.squeeze().cpu().numpy()
            
            # Binarize
            binary = (prob_map > threshold).astype(np.uint8) * 255
            
            # Calculate metrics
            gt_binary = (gt_np > 0.5).astype(np.uint8) * 255
            
            tp = np.sum((binary == 255) & (gt_binary == 255))
            fp = np.sum((binary == 255) & (gt_binary == 0))
            fn = np.sum((binary == 0) & (gt_binary == 255))
            tn = np.sum((binary == 0) & (gt_binary == 0))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
            
            # Save outputs
            sample_name = img_file.stem
            
            # 1. Original (as PNG)
            original_img = (img_np.squeeze() * 255).astype(np.uint8)
            Image.fromarray(original_img).save(output_dir / 'original' / f'{sample_name}.png')
            
            # 2. Ground truth
            Image.fromarray(gt_binary).save(output_dir / 'ground_truth' / f'{sample_name}_GT.png')
            
            # 3. Probability map (save as grayscale image for speed)
            prob_img = (prob_map * 255).astype(np.uint8)
            Image.fromarray(prob_img).save(output_dir / 'probability_maps' / f'{sample_name}_prob.png')
            
            # 4. Binarized output
            Image.fromarray(binary).save(output_dir / 'binarized' / f'{sample_name}_binary.png')
            
            # 5. Composite comparison (every 10th sample to save space, or first 50)
            if idx < 50 or idx % 10 == 0:
                # Error map
                error_map = np.zeros((*binary.shape, 3), dtype=np.uint8)
                error_map[...] = 255  # White background
                
                # Correct pixels (gray)
                correct = (binary == gt_binary)
                error_map[correct] = [200, 200, 200]
                
                # False positives (red)
                fp_mask = (binary == 255) & (gt_binary == 0)
                error_map[fp_mask] = [255, 0, 0]
                
                # False negatives (blue)
                fn_mask = (binary == 0) & (gt_binary == 255)
                error_map[fn_mask] = [0, 0, 255]
                
                # True positives (green)
                tp_mask = (binary == 255) & (gt_binary == 255)
                error_map[tp_mask] = [0, 255, 0]
                
                # Create composite
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                axes[0, 0].imshow(original_img, cmap='gray')
                axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(gt_binary, cmap='gray')
                axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(prob_map, cmap='RdYlGn', vmin=0, vmax=1)
                axes[0, 2].set_title('Probability Map', fontsize=12, fontweight='bold')
                axes[0, 2].axis('off')
                
                axes[1, 0].imshow(binary, cmap='gray')
                axes[1, 0].set_title(f'Predicted Binary\n(threshold={threshold})', fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(error_map)
                axes[1, 1].set_title('Error Map\n(Red=FP, Blue=FN, Green=TP)', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                # Metrics text
                axes[1, 2].axis('off')
                metrics_text = f"""
Sample: {sample_name}

Metrics:
  F1 Score:   {f1:.4f}
  Precision:  {precision:.4f}
  Recall:     {recall:.4f}
  Accuracy:   {accuracy:.4f}

Confusion Matrix:
  TP: {tp:,}
  FP: {fp:,}
  FN: {fn:,}
  TN: {tn:,}

Total pixels: {tp+fp+fn+tn:,}
                """
                axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                               verticalalignment='center')
                
                plt.suptitle(f'Sample {idx+1}/{len(image_files)}: {sample_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_dir / 'composites' / f'{sample_name}_composite.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            # Store results
            results.append({
                'sample': sample_name,
                'f1': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'accuracy': float(accuracy),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            })
            
        except Exception as e:
            print(f"\n❌ Error processing {img_file.name}: {e}")
            continue
    
    # Save summary statistics
    print("\n" + "="*70)
    print("💾 SAVING SUMMARY STATISTICS")
    print("="*70)
    
    summary = {
        'total_samples': len(results),
        'threshold': threshold,
        'average_f1': np.mean([r['f1'] for r in results]),
        'average_precision': np.mean([r['precision'] for r in results]),
        'average_recall': np.mean([r['recall'] for r in results]),
        'average_accuracy': np.mean([r['accuracy'] for r in results]),
        'std_f1': np.std([r['f1'] for r in results]),
        'min_f1': np.min([r['f1'] for r in results]),
        'max_f1': np.max([r['f1'] for r in results]),
        'per_sample_results': results
    }
    
    with open(output_dir / 'all_results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved summary to: {output_dir / 'all_results_summary.json'}")
    
    # Print final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"Total samples processed: {len(results)}")
    print(f"Threshold used: {threshold}")
    print(f"\nAverage Performance:")
    print(f"  F1 Score:   {summary['average_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"  Precision:  {summary['average_precision']:.4f}")
    print(f"  Recall:     {summary['average_recall']:.4f}")
    print(f"  Accuracy:   {summary['average_accuracy']:.4f}")
    print(f"\nF1 Score Range: [{summary['min_f1']:.4f}, {summary['max_f1']:.4f}]")
    print("\n" + "="*70)
    print("✅ ALL OUTPUTS SAVED TO:")
    print("="*70)
    print(f"📁 {output_dir}/")
    print(f"   ├── original/         ({len(results)} images)")
    print(f"   ├── ground_truth/     ({len(results)} images)")
    print(f"   ├── probability_maps/ ({len(results)} heatmaps)")
    print(f"   ├── binarized/        ({len(results)} binary images)")
    print(f"   ├── composites/       (50+ comparison panels)")
    print(f"   └── all_results_summary.json")
    print("="*70)
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate outputs for all DIBCO test samples')
    parser.add_argument('--threshold', type=float, default=0.504, help='Binarization threshold (default: 0.504)')
    parser.add_argument('--output_dir', type=str, default='dibco_outputs', help='Output directory')
    
    args = parser.parse_args()
    
    process_all_test_samples(threshold=args.threshold, output_dir=args.output_dir)
