import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.structure_tensor import compute_beltrami_metric
from models.bpe import BeltramiPositionalEncoding
from models.transformer_block import BeltramiTransformerBlock
from models.pcm import PhysicsConstrainedFusion

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2)

    def forward(self, x):
        return self.reduction(x)

class Upsample(nn.Module):
    def __init__(self, in_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features // 2
        self.expand = nn.ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2)

    def forward(self, x):
        return self.expand(x)

class UltraSharp(nn.Module):
    """
    UltraSharp: Geometric Transformer for Ultrasound Super-Resolution.
    Hierarchical U-shaped transformer with Beltrami-guided attention and physics constraints.
    """
    def __init__(self, in_channels=1, out_channels=1, scale=4, dim=64, num_heads=8, window_size=8, K_bpe=8, num_blocks=[2, 2, 2]):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.dim = dim
        
        # Initial Denoising feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
        # BPE generator
        self.bpe = BeltramiPositionalEncoding(K=K_bpe, T=5, dt=0.1, alpha=2.0)
        
        # Embed BPE channels into standard dimension
        self.bpe_embed = nn.Conv2d(K_bpe, dim, kernel_size=1)
        
        # Encoder BTBs
        self.enc1 = nn.Sequential(*[BeltramiTransformerBlock(dim, num_heads, window_size) for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)
        
        self.enc2 = nn.Sequential(*[BeltramiTransformerBlock(dim*2, num_heads*2, window_size) for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim*2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(*[BeltramiTransformerBlock(dim*4, num_heads*4, window_size) for _ in range(num_blocks[2])])
        
        # Decoder BTBs (Up -> Concat Skip -> Conv -> BTB)
        self.up2 = Upsample(dim*4, dim*2)
        self.dec2 = nn.Sequential(*[BeltramiTransformerBlock(dim*2, num_heads*2, window_size) for _ in range(num_blocks[1])])
        
        self.up1 = Upsample(dim*2, dim)
        self.dec1 = nn.Sequential(*[BeltramiTransformerBlock(dim, num_heads, window_size) for _ in range(num_blocks[0])])
        
        # Super-resolution Upsampling branch (PixelShuffle)
        self.sr_upsample = nn.Sequential(
            nn.Conv2d(dim, dim * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        )
        
        # Downsample features for Physics Constrained Fusion
        # Physics constrained fusion block
        self.pcm = PhysicsConstrainedFusion(in_channels=dim, out_channels=out_channels)

    def forward(self, lr_img):
        """
        lr_img: shape (B, in_channels, H, W)
        """
        # Interpolate image for structural tensor computations at native or target resolution
        # Compute tensor on upsampled base
        base_hr = F.interpolate(lr_img, scale_factor=self.scale, mode='bicubic', align_corners=False)
        base_hr = torch.clamp(base_hr, 0.0, 1.0)
        
        # Compute Structure Tensor and Beltrami Inverse Metric
        gxx, gxy, gyy, lambda_edge = compute_beltrami_metric(base_hr, alpha=2.0, sigma=1.0, rho=2.0)
        
        # Beltrami Positional Encoding
        bpe_enc = self.bpe(base_hr, lambda_edge, gxx, gxy, gyy) # (B, K_bpe, H_hr, W_hr)
        
        # Generate initial features
        x = self.initial_conv(lr_img) # (B, dim, H_lr, W_lr)
        
        # Helper: Create BPE encodings appropriate for each scale level
        # Since BPE is computed at HR, we must pool it down for encoder layers
        bpe_lvl1 = F.adaptive_avg_pool2d(bpe_enc, x.shape[2:]) 
        
        # Encoder Level 1
        x1 = x
        for block in self.enc1: x1 = block(x1, bpe_lvl1)
        
        # Downsample
        x_down1 = self.down1(x1)
        bpe_lvl2 = F.adaptive_avg_pool2d(bpe_enc, x_down1.shape[2:])
        
        # Encoder Level 2
        x2 = x_down1
        for block in self.enc2: x2 = block(x2, bpe_lvl2)
        
        # Downsample
        x_down2 = self.down2(x2)
        bpe_lvl3 = F.adaptive_avg_pool2d(bpe_enc, x_down2.shape[2:])
        
        # Bottleneck
        xb = x_down2
        for block in self.bottleneck: xb = block(xb, bpe_lvl3)
        
        # Decoder Level 2
        x_up2 = self.up2(xb)
        x_dec2 = x_up2 + x2 # Skip connection
        for block in self.dec2: x_dec2 = block(x_dec2, bpe_lvl2)
        
        # Decoder Level 1
        x_up1 = self.up1(x_dec2)
        x_dec1 = x_up1 + x1 # Skip connection
        for block in self.dec1: x_dec1 = block(x_dec1, bpe_lvl1)
        
        # Perform SR Upsampling mapping (Pixel Shuffle)
        hr_features = self.sr_upsample[0:2](x_dec1) # pixel shuffle
        hr_img_base = self.sr_upsample[2](hr_features) 
        
        # PCM logic at High-Resolution space
        # Pass features before final SR layer into PCM
        fusion_out = self.pcm(hr_features, hr_img_base)
        
        # Combine base SR output and physics-constrained correction
        final_out = base_hr + fusion_out
        
        return final_out
