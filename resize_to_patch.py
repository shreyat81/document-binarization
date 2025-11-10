import os
import numpy as np
import cv2
from tqdm import tqdm

# Input paths (normalized data)
images_dir = r"/Users/shreyatiwari/Documents/Soft Computing Project/normalization/images"
gt_dir     = r"/Users/shreyatiwari/Documents/Soft Computing Project/normalization/gt"

# Output paths (patches)
out_images_dir = r"/Users/shreyatiwari/Documents/Soft Computing Project/resize_patch/images"
out_gt_dir     = r"/Users/shreyatiwari/Documents/Soft Computing Project/resize_patch/gt"

os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_gt_dir, exist_ok=True)

# Patch configuration
PATCH_SIZE = 256  # Size of each patch
STRIDE = 128      # Overlap between patches (50% overlap)
MIN_TEXT_RATIO = 0.05  # Minimum ratio of text pixels to keep patch


def extract_patches(image, patch_size, stride):
    """
    Extract overlapping patches from an image
    """
    patches = []
    positions = []
    
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    # Handle remaining edge regions
    # Right edge
    if w % stride != 0:
        for y in range(0, h - patch_size + 1, stride):
            x = w - patch_size
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    # Bottom edge
    if h % stride != 0:
        for x in range(0, w - patch_size + 1, stride):
            y = h - patch_size
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    # Bottom-right corner
    if (h % stride != 0) and (w % stride != 0):
        y = h - patch_size
        x = w - patch_size
        patch = image[y:y+patch_size, x:x+patch_size]
        patches.append(patch)
        positions.append((y, x))
    
    return patches, positions


def is_valid_patch(gt_patch, min_text_ratio=MIN_TEXT_RATIO):
    """
    Check if patch contains enough text (foreground) pixels
    """
    text_ratio = np.mean(gt_patch > 0.5)
    if text_ratio > min_text_ratio and text_ratio < (1 - min_text_ratio):
        return True
    return False


def resize_if_needed(image, target_size):
    """
    Resize image if it's smaller than target size
    """
    h, w = image.shape[:2]
    if h < target_size or w < target_size:
        scale = max(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image


def process_images_to_patches():
    """
    Process all images and extract patches
    """
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    
    print(f"Found {len(image_files)} images to process")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, Stride: {STRIDE}")
    
    total_patches = 0
    total_valid_patches = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(images_dir, img_file)
        image = np.load(img_path)
        
        gt_file = img_file.replace('.npy', '_GT.npy')
        gt_path = os.path.join(gt_dir, gt_file)
        
        if not os.path.exists(gt_path):
            continue
        
        gt = np.load(gt_path)
        
        image = resize_if_needed(image, PATCH_SIZE)
        gt = resize_if_needed(gt, PATCH_SIZE)
        
        if image.shape[:2] != gt.shape[:2]:
            min_h = min(image.shape[0], gt.shape[0])
            min_w = min(image.shape[1], gt.shape[1])
            image = image[:min_h, :min_w]
            gt = gt[:min_h, :min_w]
        
        img_patches, positions = extract_patches(image, PATCH_SIZE, STRIDE)
        gt_patches, _ = extract_patches(gt, PATCH_SIZE, STRIDE)
        
        base_name = img_file.replace('.npy', '')
        
        for idx, (img_patch, gt_patch, pos) in enumerate(zip(img_patches, gt_patches, positions)):
            total_patches += 1
            
            if not is_valid_patch(gt_patch):
                continue
            
            total_valid_patches += 1
            
            patch_name = f"{base_name}_p{idx:04d}"
            np.save(os.path.join(out_images_dir, f"{patch_name}.npy"), img_patch)
            np.save(os.path.join(out_gt_dir, f"{patch_name}_GT.npy"), gt_patch)
    
    print(f"\n✅ Patch extraction completed!")
    print(f"Total patches: {total_patches}, Valid patches: {total_valid_patches}")


if __name__ == "__main__":
    process_images_to_patches()
