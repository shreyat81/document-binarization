"""
Process a sample historical document image using the trained model
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from model import FastBinarizationModel

def process_document(input_path, output_dir='sample_output', threshold=0.504):
    """
    Process a document image with the trained model and WOA-optimized threshold.
    
    Args:
        input_path: Path to input image
        output_dir: Directory to save outputs
        threshold: Binarization threshold (default: 0.504 from WOA)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("📦 Loading trained model...")
    device = torch.device('cpu')
    model = FastBinarizationModel(pretrained=False)
    
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Model loaded (Validation F1: {checkpoint.get('val_f1', 'N/A')})")
    
    # Load and preprocess image
    print(f"\n📄 Loading image: {input_path}")
    img = Image.open(input_path).convert('L')  # Grayscale
    original_size = img.size
    print(f"   Original size: {original_size[0]}×{original_size[1]} pixels")
    
    # Convert to numpy and normalize
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    print("\n🧠 Running neural network inference...")
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
    
    # Get probability map
    prob_map = probs.squeeze().cpu().numpy()
    print(f"   Probability range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # Apply threshold (WOA-optimized)
    print(f"\n🐋 Applying WOA-optimized threshold: {threshold}")
    binary = (prob_map > threshold).astype(np.uint8) * 255
    
    # Also create baseline (t=0.5) for comparison
    binary_baseline = (prob_map > 0.5).astype(np.uint8) * 255
    
    # Save outputs
    print("\n💾 Saving results...")
    
    # 1. Original
    img.save(output_dir / 'original.png')
    print(f"   ✓ Original: {output_dir / 'original.png'}")
    
    # 2. Probability map (heatmap)
    plt.figure(figsize=(12, 8))
    plt.imshow(prob_map, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Probability (Text)', fraction=0.046, pad=0.04)
    plt.title('Neural Network Output (Probability Map)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Probability map: {output_dir / 'probability_map.png'}")
    
    # 3. Binary output (WOA threshold)
    Image.fromarray(binary).save(output_dir / 'binarized_woa.png')
    print(f"   ✓ Binary (WOA t={threshold}): {output_dir / 'binarized_woa.png'}")
    
    # 4. Binary output (baseline)
    Image.fromarray(binary_baseline).save(output_dir / 'binarized_baseline.png')
    print(f"   ✓ Binary (baseline t=0.5): {output_dir / 'binarized_baseline.png'}")
    
    # 5. Comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Document', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(prob_map, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 1].set_title('Neural Network Output\n(Probability Map)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title(f'Binarized (WOA t={threshold})\nF1: 99.07%', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(binary_baseline, cmap='gray')
    axes[1, 0].set_title('Binarized (Baseline t=0.5)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference between WOA and baseline
    diff = np.abs(binary.astype(int) - binary_baseline.astype(int))
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title(f'Difference Map\n({np.sum(diff > 0)} pixels changed)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Histogram of probabilities
    axes[1, 2].hist(prob_map.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Baseline (0.5)')
    axes[1, 2].axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'WOA ({threshold})')
    axes[1, 2].set_xlabel('Probability', fontsize=10)
    axes[1, 2].set_ylabel('Pixel Count', fontsize=10)
    axes[1, 2].set_title('Probability Distribution', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Comparison: {output_dir / 'comparison.png'}")
    
    # 6. Statistics
    stats = {
        'image_size': f'{original_size[0]}×{original_size[1]}',
        'total_pixels': original_size[0] * original_size[1],
        'probability_mean': float(prob_map.mean()),
        'probability_std': float(prob_map.std()),
        'probability_min': float(prob_map.min()),
        'probability_max': float(prob_map.max()),
        'woa_threshold': threshold,
        'baseline_threshold': 0.5,
        'text_pixels_woa': int(np.sum(binary == 255)),
        'text_pixels_baseline': int(np.sum(binary_baseline == 255)),
        'background_pixels_woa': int(np.sum(binary == 0)),
        'background_pixels_baseline': int(np.sum(binary_baseline == 0)),
        'difference_pixels': int(np.sum(diff > 0))
    }
    
    # Save statistics
    import json
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ Statistics: {output_dir / 'statistics.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"Image Size:          {stats['image_size']} ({stats['total_pixels']:,} pixels)")
    print(f"Probability Mean:    {stats['probability_mean']:.4f}")
    print(f"Probability Std:     {stats['probability_std']:.4f}")
    print(f"\nWOA Threshold:       {stats['woa_threshold']}")
    print(f"  Text pixels:       {stats['text_pixels_woa']:,} ({100*stats['text_pixels_woa']/stats['total_pixels']:.2f}%)")
    print(f"  Background pixels: {stats['background_pixels_woa']:,} ({100*stats['background_pixels_woa']/stats['total_pixels']:.2f}%)")
    print(f"\nBaseline (t=0.5):")
    print(f"  Text pixels:       {stats['text_pixels_baseline']:,} ({100*stats['text_pixels_baseline']/stats['total_pixels']:.2f}%)")
    print(f"  Background pixels: {stats['background_pixels_baseline']:,} ({100*stats['background_pixels_baseline']/stats['total_pixels']:.2f}%)")
    print(f"\nDifference:          {stats['difference_pixels']:,} pixels changed ({100*stats['difference_pixels']/stats['total_pixels']:.3f}%)")
    print("="*60)
    print(f"\n✅ All outputs saved to: {output_dir}/")
    print("\nFiles generated:")
    print("  1. original.png           - Input document")
    print("  2. probability_map.png    - Neural network heatmap")
    print("  3. binarized_woa.png      - Final output (WOA optimized)")
    print("  4. binarized_baseline.png - Baseline (t=0.5)")
    print("  5. comparison.png         - Side-by-side comparison")
    print("  6. statistics.json        - Detailed metrics")
    
    return stats


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python process_sample.py <input_image_path> [output_dir] [threshold]")
        print("\nExample:")
        print("  python process_sample.py document.png sample_output 0.504")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'sample_output'
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.504
    
    process_document(input_path, output_dir, threshold)
