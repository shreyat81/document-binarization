import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import CompleteBinarizationPipeline
import torchvision.transforms as transforms


class DocumentBinarizationDataset(Dataset):
    """
    Dataset for document binarization
    Loads preprocessed .npy files
    """
    def __init__(self, images_dir, gt_dir, transform=None, augment=False):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.augment = augment
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        
        print(f"Found {len(self.image_files)} samples in {images_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = np.load(img_path)
        
        # Load corresponding ground truth
        gt_filename = self.image_files[idx].replace('.npy', '_GT.npy')
        if not os.path.exists(os.path.join(self.gt_dir, gt_filename)):
            # Try alternative naming convention
            base = self.image_files[idx].replace('.npy', '')
            gt_filename = base.replace('_p', '_GT_p') + '.npy'
        
        gt_path = os.path.join(self.gt_dir, gt_filename)
        
        if os.path.exists(gt_path):
            ground_truth = np.load(gt_path)
        else:
            print(f"Warning: GT not found for {self.image_files[idx]}, using zeros")
            ground_truth = np.zeros_like(image)
        
        # Ensure image is float32 and in range [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Ensure ground truth is binary
        if ground_truth.max() > 1:
            ground_truth = (ground_truth > 127).astype(np.float32)
        else:
            ground_truth = ground_truth.astype(np.float32)
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        if len(ground_truth.shape) == 2:
            ground_truth = np.expand_dims(ground_truth, 0)
        
        # Convert grayscale to 3-channel for pretrained models
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        ground_truth = torch.from_numpy(ground_truth).float()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = torch.flip(image, [2])
                ground_truth = torch.flip(ground_truth, [2])
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                image = torch.flip(image, [1])
                ground_truth = torch.flip(ground_truth, [1])
            
            # Random rotation (90, 180, 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = torch.rot90(image, k, [1, 2])
                ground_truth = torch.rot90(ground_truth, k, [1, 2])
        
        return image, ground_truth


class CombinedLoss(nn.Module):
    """
    Combined loss function for document binarization
    Combines BCE, Dice, and edge-aware losses
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, edge_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target, smooth=1.0):
        """
        Dice loss for segmentation
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def edge_loss(self, pred, target):
        """
        Edge-aware loss to preserve text boundaries
        """
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)
        
        # Compute edges
        pred_edge_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        
        target_edge_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_edge_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        # MSE on edges
        edge_loss = torch.nn.functional.mse_loss(pred_edge, target_edge)
        return edge_loss
    
    def forward(self, pred, target):
        """
        Compute combined loss
        """
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.edge_weight * edge)
        
        return total_loss, {
            'bce': bce.item(),
            'dice': dice.item(),
            'edge': edge.item(),
            'total': total_loss.item()
        }


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate evaluation metrics
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # True Positives, False Positives, False Negatives, True Negatives
    TP = ((pred_binary == 1) & (target_binary == 1)).float().sum()
    FP = ((pred_binary == 1) & (target_binary == 0)).float().sum()
    FN = ((pred_binary == 0) & (target_binary == 1)).float().sum()
    TN = ((pred_binary == 0) & (target_binary == 0)).float().sum()
    
    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item(),
        'iou': iou.item()
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    total_metrics = {'bce': 0, 'dice': 0, 'edge': 0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model.model(images)  # Get logits from main model
        
        # Calculate loss
        loss, loss_components = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += loss_components[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'bce': loss_components['bce'],
            'dice': loss_components['dice']
        })
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0
    total_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'iou': 0}
    loss_components_sum = {'bce': 0, 'dice': 0, 'edge': 0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
    
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model.model(images)  # Get logits
            
            # Calculate loss
            loss, loss_components = criterion(outputs, targets)
            total_loss += loss.item()
            
            for key in loss_components_sum:
                loss_components_sum[key] += loss_components[key]
            
            # Calculate metrics
            pred_prob = torch.sigmoid(outputs)
            metrics = calculate_metrics(pred_prob, targets)
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'f1': metrics['f1_score']
            })
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    avg_loss_components = {k: v / len(dataloader) for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_metrics, avg_loss_components


def train_model(config):
    """
    Main training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DocumentBinarizationDataset(
        images_dir=config['train_images_dir'],
        gt_dir=config['train_gt_dir'],
        augment=True
    )
    
    val_dataset = DocumentBinarizationDataset(
        images_dir=config['val_images_dir'],
        gt_dir=config['val_gt_dir'],
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = CompleteBinarizationPipeline(pretrained=config['pretrained'])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(
        bce_weight=config['bce_weight'],
        dice_weight=config['dice_weight'],
        edge_weight=config['edge_weight']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_iou': []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics, val_loss_components = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics - F1: {val_metrics['f1_score']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Loss Components - BCE: {val_loss_components['bce']:.4f}, "
              f"Dice: {val_loss_components['dice']:.4f}, "
              f"Edge: {val_loss_components['edge']:.4f}")
        
        # Save best model
        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            best_epoch = epoch
            
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"✅ Saved best model with F1: {best_f1:.4f}")
        
        # Save checkpoint every N epochs
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best F1 Score: {best_f1:.4f} at epoch {best_epoch}")
    
    # Plot training history
    plot_training_history(history, config['checkpoint_dir'])
    
    return model, history


def plot_training_history(history, save_dir):
    """
    Plot and save training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score
    axes[0, 1].plot(history['val_f1'], label='Val F1', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU
    axes[1, 0].plot(history['val_iou'], label='Val IoU', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Validation IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined metrics
    axes[1, 1].plot(history['val_f1'], label='F1 Score', color='green')
    axes[1, 1].plot(history['val_iou'], label='IoU', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
    print(f"📊 Training history plot saved to {save_dir}/training_history.png")
    plt.close()


if __name__ == "__main__":
    # Training configuration
    config = {
        # Data paths - UPDATE THESE BASED ON YOUR DATA STRUCTURE
        'train_images_dir': '/Users/shreyatiwari/Documents/Soft Computing Project/split/train/images',
        'train_gt_dir': '/Users/shreyatiwari/Documents/Soft Computing Project/split/train/gt',
        'val_images_dir': '/Users/shreyatiwari/Documents/Soft Computing Project/split/val/images',
        'val_gt_dir': '/Users/shreyatiwari/Documents/Soft Computing Project/split/val/gt',
        
        # Training parameters
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        
        # Loss weights
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'edge_weight': 0.5,
        
        # Model parameters
        'pretrained': True,
        
        # Checkpoint
        'checkpoint_dir': './checkpoints',
        'save_every': 5
    }
    
    # Train model
    model, history = train_model(config)
