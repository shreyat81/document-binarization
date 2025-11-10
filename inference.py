import torch
import numpy as np
import cv2
import os
from model import CompleteBinarizationPipeline
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = CompleteBinarizationPipeline(pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'best_f1' in checkpoint:
        print(f"Best F1 Score: {checkpoint['best_f1']:.4f}")
    
    return model


def preprocess_image(image_path, target_size=None):
    """
    Preprocess a single image for inference
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_shape = img.shape
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Resize if target size is specified
    if target_size:
        img = cv2.resize(img, (target_size, target_size))
    
    # Convert to 3-channel for pretrained models
    img = np.stack([img, img, img], axis=0)  # (3, H, W)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor, original_shape


def preprocess_npy_image(npy_path):
    """
    Preprocess a .npy image for inference
    """
    # Load numpy array
    img = np.load(npy_path)
    
    original_shape = img.shape
    
    # Ensure float32 in range [0, 1]
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    
    # Handle dimensions
    if len(img.shape) == 2:
        # Convert to 3-channel
        img = np.stack([img, img, img], axis=0)  # (3, H, W)
    elif len(img.shape) == 3 and img.shape[0] == 1:
        # Convert grayscale to 3-channel
        img = np.repeat(img, 3, axis=0)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor, original_shape


def postprocess_output(output, original_shape):
    """
    Postprocess model output to original image size
    """
    # Squeeze batch and channel dimensions
    output = output.squeeze().cpu().numpy()
    
    # Resize to original shape if needed
    if output.shape != original_shape:
        output = cv2.resize(output, (original_shape[1], original_shape[0]))
    
    # Convert to uint8 [0, 255]
    output = (output * 255).astype(np.uint8)
    
    return output


def infer_single_image(model, image_path, device, save_path=None, return_intermediate=False):
    """
    Run inference on a single image
    """
    # Check if it's a .npy file or regular image
    if image_path.endswith('.npy'):
        img_tensor, original_shape = preprocess_npy_image(image_path)
    else:
        img_tensor, original_shape = preprocess_image(image_path)
    
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        if return_intermediate:
            output, intermediates = model(img_tensor, return_intermediate=True)
        else:
            output = model(img_tensor)
            intermediates = None
    
    # Postprocess
    binary_result = postprocess_output(output, original_shape)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, binary_result)
        print(f"Saved result to {save_path}")
    
    if return_intermediate:
        # Postprocess intermediate outputs
        intermediate_results = {
            'binary': binary_result,
            'probability_map': postprocess_output(intermediates['probability_map'], original_shape),
            'fuzzy_output': postprocess_output(intermediates['fuzzy_output'], original_shape),
            'adaptive_threshold': postprocess_output(intermediates['adaptive_threshold'], original_shape)
        }
        return intermediate_results
    
    return binary_result


def batch_inference(model, input_dir, output_dir, device, file_extension='.npy'):
    """
    Run inference on all images in a directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(file_extension)]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        
        # Output path
        output_name = img_file.replace(file_extension, '_binarized.png')
        output_path = os.path.join(output_dir, output_name)
        
        # Run inference
        try:
            infer_single_image(model, img_path, device, save_path=output_path)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    print(f"✅ Batch inference completed! Results saved to {output_dir}")


def visualize_results(image_path, model, device, save_path=None):
    """
    Visualize all intermediate results
    """
    # Get intermediate results
    results = infer_single_image(model, image_path, device, return_intermediate=True)
    
    # Load original image
    if image_path.endswith('.npy'):
        original = np.load(image_path)
        if original.dtype != np.uint8:
            original = (original * 255).astype(np.uint8)
    else:
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Probability map
    axes[0, 1].imshow(results['probability_map'], cmap='hot')
    axes[0, 1].set_title('Probability Map', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Fuzzy output
    axes[0, 2].imshow(results['fuzzy_output'], cmap='gray')
    axes[0, 2].set_title('Fuzzy System Output', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Adaptive threshold
    axes[1, 0].imshow(results['adaptive_threshold'], cmap='viridis')
    axes[1, 0].set_title('Adaptive Threshold Map', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Final binary result
    axes[1, 1].imshow(results['binary'], cmap='gray')
    axes[1, 1].set_title('Final Binarized Result', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Comparison (original vs binarized)
    axes[1, 2].imshow(np.hstack([original, results['binary']]), cmap='gray')
    axes[1, 2].set_title('Original vs Binarized', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def evaluate_on_test_set(model, test_images_dir, test_gt_dir, device):
    """
    Evaluate model on test set with ground truth
    """
    from train import calculate_metrics
    
    # Get all test images
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.npy')])
    
    print(f"Evaluating on {len(image_files)} test images...")
    
    all_metrics = []
    
    for img_file in tqdm(image_files, desc="Evaluating"):
        img_path = os.path.join(test_images_dir, img_file)
        
        # Load ground truth
        gt_filename = img_file.replace('.npy', '_GT.npy')
        if not os.path.exists(os.path.join(test_gt_dir, gt_filename)):
            base = img_file.replace('.npy', '')
            gt_filename = base.replace('_p', '_GT_p') + '.npy'
        
        gt_path = os.path.join(test_gt_dir, gt_filename)
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {img_file}")
            continue
        
        gt = np.load(gt_path)
        if gt.max() > 1:
            gt = (gt > 127).astype(np.float32)
        
        # Run inference
        img_tensor, _ = preprocess_npy_image(img_path)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert to same shape as GT
        output_np = output.squeeze().cpu().numpy()
        if output_np.shape != gt.shape:
            output_np = cv2.resize(output_np, (gt.shape[1], gt.shape[0]))
        
        # Calculate metrics
        output_tensor = torch.from_numpy(output_np).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        
        metrics = calculate_metrics(output_tensor, gt_tensor)
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\n" + "=" * 80)
    print("Test Set Evaluation Results:")
    print("=" * 80)
    for key, value in avg_metrics.items():
        print(f"{key.capitalize():15s}: {value:.4f}")
    print("=" * 80)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Document Binarization Inference')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'visualize', 'evaluate'],
                       default='single', help='Inference mode')
    parser.add_argument('--gt_dir', type=str, default=None,
                       help='Ground truth directory (for evaluation mode)')
    parser.add_argument('--extension', type=str, default='.npy',
                       help='File extension for batch processing')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Run inference based on mode
    if args.mode == 'single':
        output_path = os.path.join(args.output, 'result.png')
        os.makedirs(args.output, exist_ok=True)
        result = infer_single_image(model, args.input, device, save_path=output_path)
        print(f"✅ Inference completed! Result saved to {output_path}")
        
    elif args.mode == 'batch':
        batch_inference(model, args.input, args.output, device, args.extension)
        
    elif args.mode == 'visualize':
        output_path = os.path.join(args.output, 'visualization.png')
        os.makedirs(args.output, exist_ok=True)
        visualize_results(args.input, model, device, save_path=output_path)
        
    elif args.mode == 'evaluate':
        if args.gt_dir is None:
            print("Error: --gt_dir required for evaluation mode")
            return
        evaluate_on_test_set(model, args.input, args.gt_dir, device)


if __name__ == "__main__":
    # If running without command line arguments, use default test
    import sys
    
    if len(sys.argv) == 1:
        print("Running in test mode with default settings...")
        print("\nUsage examples:")
        print("  Single image:")
        print("    python inference.py --checkpoint checkpoints/best_model.pth --input image.png --mode single")
        print("\n  Batch processing:")
        print("    python inference.py --checkpoint checkpoints/best_model.pth --input test_images/ --output results/ --mode batch")
        print("\n  Visualization:")
        print("    python inference.py --checkpoint checkpoints/best_model.pth --input image.png --mode visualize")
        print("\n  Evaluation:")
        print("    python inference.py --checkpoint checkpoints/best_model.pth --input test_images/ --gt_dir test_gt/ --mode evaluate")
    else:
        main()
