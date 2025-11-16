import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class DocumentBinarizationDataset(Dataset):
    """
    Optimized PyTorch Dataset for document binarization with caching.
    
    Loads .npy files containing 256x256 patches with matching ground truth masks.
    Applies data augmentation for training split.
    Features RAM caching for faster training.
    
    Args:
        images_dir: Path to directory containing image patches (.npy files)
        gt_dir: Path to directory containing ground truth masks (.npy files)
        augment: Whether to apply data augmentation (default: False)
        cache_in_ram: Cache loaded samples in RAM (default: True for speed)
    """
    
    def __init__(self, images_dir, gt_dir, augment=False, cache_in_ram=True):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cache = {} if cache_in_ram else None
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .npy files found in {images_dir}")
        
        print(f"Found {len(self.image_files)} samples in {images_dir}")
        if cache_in_ram:
            print(f"   RAM caching enabled for faster loading")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Check cache first
        if self.cache_in_ram and idx in self.cache:
            image, gt = self.cache[idx]
        else:
            # Load image
            img_file = self.image_files[idx]
            img_path = os.path.join(self.images_dir, img_file)
            image = np.load(img_path)
            
            # Load corresponding ground truth
            # Handle both naming patterns: filename.npy -> filename_GT.npy or filename_GT_pXXXX.npy
            base_name = img_file.replace('.npy', '')
            
            # Try different GT naming patterns
            if '_p' in base_name:
                # Pattern: 2009_img001_p0042.npy -> 2009_img001_GT_p0042.npy
                gt_file = base_name.replace('_p', '_GT_p') + '.npy'
            else:
                # Pattern: filename.npy -> filename_GT.npy
                gt_file = base_name + '_GT.npy'
            
            gt_path = os.path.join(self.gt_dir, gt_file)
            
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Ground truth not found: {gt_path}")
            
            gt = np.load(gt_path)
            
            # CRITICAL: Ensure proper normalization
            # Images should be [0, 1], GT should be strictly {0, 1}
            image = np.clip(image, 0.0, 1.0).astype(np.float32)
            gt = np.clip(gt, 0.0, 1.0).astype(np.float32)
            
            # Convert to torch tensors
            # Add channel dimension if grayscale
            if len(image.shape) == 2:
                image = image[np.newaxis, ...]  # (H, W) -> (1, H, W)
            
            if len(gt.shape) == 2:
                gt = gt[np.newaxis, ...]  # (H, W) -> (1, H, W)
            
            # Convert to torch tensors
            image = torch.from_numpy(image).float()
            gt = torch.from_numpy(gt).float()
            
            # Cache if enabled
            if self.cache_in_ram:
                self.cache[idx] = (image.clone(), gt.clone())
        
        # Apply augmentations for training (on-the-fly, not cached)
        if self.augment:
            image, gt = self._apply_augmentations(image, gt)
        
        return image, gt
    
    def _apply_augmentations(self, image, gt):
        """
        Lightweight augmentations for FAST training.
        
        Augmentations (minimal for speed):
        - Random horizontal flip
        - Random vertical flip
        """
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
        
        # Random vertical flip (50% chance)
        if random.random() > 0.5:
            image = TF.vflip(image)
            gt = TF.vflip(gt)
        
        # Skip heavy augmentations (rotations, brightness) for speed
        
        return image, gt


def get_dataloaders(train_images_dir, train_gt_dir,
                   val_images_dir, val_gt_dir,
                   test_images_dir=None, test_gt_dir=None,
                   batch_size=4, num_workers=0):
    """
    Create DataLoaders for train, validation, and optionally test sets.
    
    Args:
        train_images_dir: Path to training images
        train_gt_dir: Path to training ground truths
        val_images_dir: Path to validation images
        val_gt_dir: Path to validation ground truths
        test_images_dir: Path to test images (optional)
        test_gt_dir: Path to test ground truths (optional)
        batch_size: Batch size for training (default: 4)
        num_workers: Number of worker processes (default: 0)
    
    Returns:
        train_loader, val_loader, (test_loader if test dirs provided)
    """
    # Create datasets
    train_dataset = DocumentBinarizationDataset(
        train_images_dir, train_gt_dir, augment=True
    )
    
    val_dataset = DocumentBinarizationDataset(
        val_images_dir, val_gt_dir, augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Set to True if using GPU
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Optional test loader
    if test_images_dir and test_gt_dir:
        test_dataset = DocumentBinarizationDataset(
            test_images_dir, test_gt_dir, augment=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Test the dataset
    print("Testing DocumentBinarizationDataset...")
    
    train_dataset = DocumentBinarizationDataset(
        images_dir="split/train/images",
        gt_dir="split/train/gt",
        augment=True
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    
    # Test loading a sample
    image, gt = train_dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"GT shape: {gt.shape}, dtype: {gt.dtype}")
    print(f"GT unique values: {torch.unique(gt).tolist()}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = get_dataloaders(
        train_images_dir="split/train/images",
        train_gt_dir="split/train/gt",
        val_images_dir="split/val/images",
        val_gt_dir="split/val/gt",
        batch_size=4
    )
    
    # Get one batch
    images_batch, gt_batch = next(iter(train_loader))
    print(f"Batch images shape: {images_batch.shape}")
    print(f"Batch GT shape: {gt_batch.shape}")
    
    print("\n✅ Dataset test passed!")
