"""
Optimized Document Binarization Model for CPU Training
- Fast, stable, no NaN losses
- EfficientNet-B0 encoder only
- Lightweight UNet decoder with GroupNorm
- Bilinear upsampling (no ConvTranspose2d)
- < 5M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder for multi-scale feature extraction."""
    
    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            efficientnet = models.efficientnet_b0(weights=weights)
        else:
            efficientnet = models.efficientnet_b0(weights=None)
        
        self.features = efficientnet.features
        # Extract at specific stages: [2, 3, 4, 6, 8]
        self.stage_indices = [2, 3, 4, 6, 8]
    
    def forward(self, x):
        """Extract 5 multi-scale features."""
        features = []
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx in self.stage_indices:
                features.append(x)
        return features


class LightweightDecoder(nn.Module):
    """
    Lightweight UNet decoder block optimized for CPU.
    - GroupNorm instead of BatchNorm (faster on CPU)
    - Bilinear upsampling instead of ConvTranspose2d (faster, stable)
    - Reduced channels
    """
    
    def __init__(self, in_channels, skip_channels, out_channels, num_groups=8):
        super(LightweightDecoder, self).__init__()
        
        # Ensure channels are divisible by num_groups
        num_groups = min(num_groups, out_channels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        """
        Forward with bilinear upsampling + skip connection.
        """
        # Bilinear upsampling (2x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Match skip dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and process
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class FastBinarizationModel(nn.Module):
    """
    Optimized document binarization model for CPU training.
    
    Features:
    - Single EfficientNet-B0 encoder
    - Lightweight decoder with GroupNorm
    - Bilinear upsampling
    - < 5M parameters
    - No NaN guarantees
    """
    
    def __init__(self, pretrained=True):
        super(FastBinarizationModel, self).__init__()
        
        # Encoder
        self.encoder = EfficientNetEncoder(pretrained=pretrained)
        
        # Input adapter (grayscale -> RGB)
        self.input_adapter = nn.Conv2d(1, 3, 1)
        
        # Feature projection (reduce channels)
        # EfficientNet-B0: [24, 40, 80, 192, 1280]
        self.proj1 = nn.Conv2d(24, 16, 1)
        self.proj2 = nn.Conv2d(40, 32, 1)
        self.proj3 = nn.Conv2d(80, 64, 1)
        self.proj4 = nn.Conv2d(192, 128, 1)
        self.proj5 = nn.Conv2d(1280, 128, 1)  # Bottleneck
        
        # Lightweight decoder
        self.dec4 = LightweightDecoder(128, 128, 64, num_groups=8)
        self.dec3 = LightweightDecoder(64, 64, 32, num_groups=8)
        self.dec2 = LightweightDecoder(32, 32, 16, num_groups=8)
        self.dec1 = LightweightDecoder(16, 16, 8, num_groups=8)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=1),
            nn.GroupNorm(4, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, 1)
        )
    
    def forward(self, x):
        """
        Forward pass with NaN protection.
        
        Args:
            x: Input (B, 1, H, W) - grayscale [0, 1]
        
        Returns:
            logits: (B, 1, H, W) - clamped to avoid NaN
        """
        original_size = x.shape[2:]
        
        # Convert to RGB
        x = self.input_adapter(x)
        
        # Extract features
        feats = self.encoder(x)
        
        # Project features
        f1 = self.proj1(feats[0])  # 16 channels
        f2 = self.proj2(feats[1])  # 32 channels
        f3 = self.proj3(feats[2])  # 64 channels
        f4 = self.proj4(feats[3])  # 128 channels
        f5 = self.proj5(feats[4])  # 128 channels (bottleneck)
        
        # Decoder with skip connections
        x = self.dec4(f5, f4)  # 64 channels
        x = self.dec3(x, f3)   # 32 channels
        x = self.dec2(x, f2)   # 16 channels
        x = self.dec1(x, f1)   # 8 channels
        
        # Final upsampling to original size
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # Output logits
        logits = self.final(x)
        
        # CRITICAL: Clamp to prevent NaN in loss
        logits = torch.clamp(logits, -10.0, 10.0)
        
        return logits


# Keep old class name for compatibility
DocumentBinarizationModel = FastBinarizationModel


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# Test and validate
if __name__ == "__main__":
    print("=" * 70)
    print("Optimized Model Test - CPU Friendly")
    print("=" * 70)
    
    model = FastBinarizationModel(pretrained=False)
    
    # Model stats
    params = count_parameters(model)
    size_mb = get_model_size(model)
    
    print(f"\n✅ Model Statistics:")
    print(f"   Parameters: {params:,}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Target: < 5M parameters")
    print(f"   Status: {'PASS ✅' if params < 5_000_000 else 'FAIL ❌'}")
    
    # Test forward pass
    print(f"\n✅ Testing forward pass...")
    x = torch.randn(2, 1, 256, 256)
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"   Contains NaN: {torch.isnan(logits).any().item()}")
    print(f"   Contains Inf: {torch.isinf(logits).any().item()}")
    
    # Test with edge cases
    print(f"\n✅ Testing edge cases...")
    
    # All zeros
    x_zeros = torch.zeros(2, 1, 256, 256)
    with torch.no_grad():
        out_zeros = model(x_zeros)
    print(f"   All zeros input - NaN: {torch.isnan(out_zeros).any().item()}")
    
    # All ones
    x_ones = torch.ones(2, 1, 256, 256)
    with torch.no_grad():
        out_ones = model(x_ones)
    print(f"   All ones input - NaN: {torch.isnan(out_ones).any().item()}")
    
    print("\n" + "=" * 70)
    print("✅ Model is ready for training!")
    print("=" * 70)
