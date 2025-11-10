import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet-B0 for extracting local features at multiple scales
    """
    def __init__(self, pretrained=True):
        super(EfficientNetFeatureExtractor, self).__init__()
        
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        self.features = efficientnet.features
        
        # Extract features at different scales for skip connections
        # EfficientNet-B0 structure:
        # - Stage 0: 32 channels
        # - Stage 1: 16 channels
        # - Stage 2: 24 channels
        # - Stage 3: 40 channels
        # - Stage 4: 80 channels
        # - Stage 5: 112 channels
        # - Stage 6: 192 channels
        # - Stage 7: 320 channels
        # - Stage 8: 1280 channels
        
    def forward(self, x):
        """
        Extract multi-scale features from EfficientNet
        Returns features at different resolutions for skip connections
        """
        features = []
        
        # Progressive feature extraction
        for idx, layer in enumerate(self.features):
            x = layer(x)
            # Save features at specific stages for skip connections
            if idx in [1, 2, 3, 4, 6]:  # Different resolution stages
                features.append(x)
        
        return features


class InceptionNetFeatureExtractor(nn.Module):
    """
    InceptionNet-V3 for extracting multi-scale features
    """
    def __init__(self, pretrained=True):
        super(InceptionNetFeatureExtractor, self).__init__()
        
        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=pretrained, aux_logits=False)
        
        # Extract the convolutional layers
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        
    def forward(self, x):
        """
        Extract multi-scale features from InceptionV3
        """
        features = []
        
        # Initial convolutions
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        features.append(x)  # First feature level
        
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        features.append(x)  # Second feature level
        
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        features.append(x)  # Third feature level
        
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        features.append(x)  # Fourth feature level
        
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        features.append(x)  # Fifth feature level
        
        return features


class FeatureFusion(nn.Module):
    """
    Fuses features from EfficientNet and InceptionNet
    """
    def __init__(self, efficient_channels, inception_channels, out_channels):
        super(FeatureFusion, self).__init__()
        
        # Adaptive pooling to match dimensions
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 1x1 convolutions to match channel dimensions
        self.efficient_conv = nn.Conv2d(efficient_channels, out_channels, 1)
        self.inception_conv = nn.Conv2d(inception_channels, out_channels, 1)
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, efficient_feat, inception_feat):
        """
        Fuse features from both networks
        """
        # Match spatial dimensions
        if efficient_feat.shape[2:] != inception_feat.shape[2:]:
            target_size = efficient_feat.shape[2:]
            inception_feat = F.interpolate(inception_feat, size=target_size, 
                                          mode='bilinear', align_corners=False)
        
        # Match channel dimensions
        efficient_feat = self.efficient_conv(efficient_feat)
        inception_feat = self.inception_conv(inception_feat)
        
        # Concatenate and fuse
        fused = torch.cat([efficient_feat, inception_feat], dim=1)
        fused = self.fusion_conv(fused)
        
        return fused


class UNetDecoderBlock(nn.Module):
    """
    UNet decoder block with skip connections
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 
                                           kernel_size=2, stride=2)
        
        # Convolution after concatenation with skip connection
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        """
        Decoder forward pass with optional skip connection
        """
        x = self.upsample(x)
        
        if skip is not None:
            # Match dimensions if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], 
                                 mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention mechanism for skip connections
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        g: gating signal from coarser scale
        x: skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Match spatial dimensions
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], 
                              mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class DocumentBinarizationModel(nn.Module):
    """
    Complete document binarization model combining:
    - EfficientNet-B0 (local features)
    - InceptionNet-V3 (multi-scale features)
    - Feature Fusion
    - UNet-style Decoder with Attention
    """
    def __init__(self, pretrained=True):
        super(DocumentBinarizationModel, self).__init__()
        
        # Feature extractors
        self.efficientnet = EfficientNetFeatureExtractor(pretrained=pretrained)
        self.inceptionnet = InceptionNetFeatureExtractor(pretrained=pretrained)
        
        # Feature fusion modules at different scales
        # These dimensions need to match the actual output channels
        self.fusion_modules = nn.ModuleList([
            FeatureFusion(16, 64, 64),    # Early stage
            FeatureFusion(24, 192, 128),  # Mid-early stage
            FeatureFusion(40, 288, 256),  # Mid stage
            FeatureFusion(112, 768, 512), # Mid-late stage
            FeatureFusion(320, 2048, 1024) # Late stage
        ])
        
        # Attention gates for skip connections
        self.attention1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attention2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attention3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.attention4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        
        # UNet-style decoder
        self.decoder4 = UNetDecoderBlock(1024, 512, 512)
        self.decoder3 = UNetDecoderBlock(512, 256, 256)
        self.decoder2 = UNetDecoderBlock(256, 128, 128)
        self.decoder1 = UNetDecoderBlock(128, 64, 64)
        
        # Final upsampling and output
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Output layer (single channel - binary segmentation)
        self.output = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the complete architecture
        Input: x - degraded document image (B, C, H, W)
        Output: binary segmentation logits (B, 1, H, W)
        """
        input_size = x.shape[2:]
        
        # Extract features from both networks
        efficient_features = self.efficientnet(x)
        inception_features = self.inceptionnet(x)
        
        # Fuse features at multiple scales
        fused_features = []
        num_fusion = min(len(efficient_features), len(inception_features), 
                        len(self.fusion_modules))
        
        for i in range(num_fusion):
            fused = self.fusion_modules[i](efficient_features[i], 
                                          inception_features[i])
            fused_features.append(fused)
        
        # UNet decoder with attention and skip connections
        # Start from the deepest fused features
        x = fused_features[-1]
        
        # Decoder path with attention-gated skip connections
        if len(fused_features) > 1:
            skip = self.attention1(x, fused_features[-2])
            x = self.decoder4(x, skip)
        
        if len(fused_features) > 2:
            skip = self.attention2(x, fused_features[-3])
            x = self.decoder3(x, skip)
        
        if len(fused_features) > 3:
            skip = self.attention3(x, fused_features[-4])
            x = self.decoder2(x, skip)
        
        if len(fused_features) > 4:
            skip = self.attention4(x, fused_features[-5])
            x = self.decoder1(x, skip)
        
        # Final upsampling to match input resolution
        x = self.final_upsample(x)
        
        # Resize to exact input dimensions if needed
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', 
                            align_corners=False)
        
        # Output logits (no activation - use BCEWithLogitsLoss)
        logits = self.output(x)
        
        return logits


class ProbabilityMapGenerator(nn.Module):
    """
    Generates probability map from model logits
    """
    def __init__(self):
        super(ProbabilityMapGenerator, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, logits):
        """
        Convert logits to probability map [0, 1]
        """
        return self.sigmoid(logits)


class FuzzySystem(nn.Module):
    """
    Fuzzy logic system for refining binarization
    Applies fuzzy rules to improve edge preservation and noise reduction
    """
    def __init__(self, num_membership_functions=3):
        super(FuzzySystem, self).__init__()
        
        # Learnable fuzzy membership function parameters
        self.low_params = nn.Parameter(torch.tensor([0.0, 0.3]))
        self.mid_params = nn.Parameter(torch.tensor([0.3, 0.7]))
        self.high_params = nn.Parameter(torch.tensor([0.7, 1.0]))
        
        # Fuzzy rule weights
        self.rule_weights = nn.Parameter(torch.ones(num_membership_functions))
        
    def triangular_mf(self, x, a, b, c):
        """
        Triangular membership function
        """
        return torch.clamp((x - a) / (b - a), 0, 1) * torch.clamp((c - x) / (c - b), 0, 1)
    
    def forward(self, prob_map):
        """
        Apply fuzzy logic to refine probability map
        """
        # Calculate membership degrees
        low = self.triangular_mf(prob_map, 0.0, self.low_params[0], self.low_params[1])
        mid = self.triangular_mf(prob_map, self.mid_params[0], 
                                (self.mid_params[0] + self.mid_params[1]) / 2, 
                                self.mid_params[1])
        high = self.triangular_mf(prob_map, self.high_params[0], self.high_params[1], 1.0)
        
        # Apply fuzzy rules with learnable weights
        output = (low * self.rule_weights[0] * 0.0 + 
                 mid * self.rule_weights[1] * 0.5 + 
                 high * self.rule_weights[2] * 1.0)
        
        # Normalize
        weight_sum = low * self.rule_weights[0] + mid * self.rule_weights[1] + high * self.rule_weights[2]
        output = output / (weight_sum + 1e-6)
        
        return output


class WhaleOptimization(nn.Module):
    """
    Whale Optimization inspired refinement module
    Uses learnable parameters to optimize threshold selection
    """
    def __init__(self):
        super(WhaleOptimization, self).__init__()
        
        # Learnable threshold parameters
        self.threshold_base = nn.Parameter(torch.tensor(0.5))
        self.threshold_range = nn.Parameter(torch.tensor(0.2))
        
        # Adaptive threshold convolution
        self.adaptive_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, fuzzy_output):
        """
        Apply adaptive threshold optimization
        """
        # Generate adaptive threshold map
        adaptive_threshold = self.adaptive_conv(fuzzy_output)
        
        # Combine with learnable global threshold
        final_threshold = self.threshold_base + self.threshold_range * (adaptive_threshold - 0.5)
        final_threshold = torch.clamp(final_threshold, 0.0, 1.0)
        
        # Apply threshold
        binary_output = (fuzzy_output > final_threshold).float()
        
        return binary_output, final_threshold


class CompleteBinarizationPipeline(nn.Module):
    """
    Complete end-to-end document binarization pipeline
    Includes: Feature Extraction -> Fusion -> UNet Decoder -> 
              Probability Map -> Fuzzy System -> Whale Optimization
    """
    def __init__(self, pretrained=True):
        super(CompleteBinarizationPipeline, self).__init__()
        
        # Main binarization model
        self.model = DocumentBinarizationModel(pretrained=pretrained)
        
        # Probability map generator
        self.prob_generator = ProbabilityMapGenerator()
        
        # Fuzzy system
        self.fuzzy_system = FuzzySystem()
        
        # Whale optimization
        self.whale_optimization = WhaleOptimization()
        
    def forward(self, x, return_intermediate=False):
        """
        Complete forward pass through the pipeline
        
        Args:
            x: Input degraded document image
            return_intermediate: If True, return all intermediate outputs
            
        Returns:
            final_binary: Final binarized image
            (optional) intermediate outputs dictionary
        """
        # Step 1: Get logits from main model
        logits = self.model(x)
        
        # Step 2: Generate probability map
        prob_map = self.prob_generator(logits)
        
        # Step 3: Apply fuzzy system
        fuzzy_output = self.fuzzy_system(prob_map)
        
        # Step 4: Whale optimization for final binarization
        final_binary, adaptive_threshold = self.whale_optimization(fuzzy_output)
        
        if return_intermediate:
            return final_binary, {
                'logits': logits,
                'probability_map': prob_map,
                'fuzzy_output': fuzzy_output,
                'adaptive_threshold': adaptive_threshold
            }
        
        return final_binary


def test_model():
    """
    Test function to verify model architecture
    """
    # Create model
    model = CompleteBinarizationPipeline(pretrained=False)
    
    # Test input (batch_size=2, channels=3, height=256, width=256)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output, intermediates = model(x, return_intermediate=True)
    
    print("Model Architecture Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Logits shape: {intermediates['logits'].shape}")
    print(f"Probability map shape: {intermediates['probability_map'].shape}")
    print(f"Fuzzy output shape: {intermediates['fuzzy_output'].shape}")
    print(f"Adaptive threshold shape: {intermediates['adaptive_threshold'].shape}")
    print("\n✅ Model architecture test passed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    test_model()
