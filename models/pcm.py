import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhysicsConstrainedFusion(nn.Module):
    """
    Physics-Constrained PSF-fusion decoder branch.
    Injects ultrasound formation priors ensuring reconstruction consistency.
    Uses parallel anisotropic Gaussian kernels with learnable gating weights.
    """
    def __init__(self, in_channels, out_channels=1, kernel_size=11, num_psfs=4):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_psfs = num_psfs
        
        # Parallel anisotropic Gaussian kernels
        # Initialized to the PSF Bank specified in the paper
        psf_params = [
            (0.5, 1.0),
            (0.8, 1.5),
            (1.1, 2.0),
            (1.4, 2.5)
        ][:num_psfs]
        
        # We make the parameters learnable to adapt to actual device PSF if needed
        self.sigma_axials = nn.Parameter(torch.tensor([p[0] for p in psf_params], dtype=torch.float32))
        self.sigma_laterals = nn.Parameter(torch.tensor([p[1] for p in psf_params], dtype=torch.float32))
        
        # Gating network to determine which PSF to emphasize
        # Global average pooling followed by an MLP
        self.gate_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_psfs, 1),
            nn.Softmax(dim=1)
        )
        
        # Final fusion convolution
        self.fusion_conv = nn.Conv2d(in_channels + num_psfs * out_channels, out_channels, kernel_size=3, padding=1)

    def _generate_gaussian_kernel(self, sigma_axial, sigma_lateral, device, dtype):
        half_size = self.kernel_size // 2
        y = torch.arange(-half_size, half_size + 1, device=device, dtype=dtype)
        x = torch.arange(-half_size, half_size + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Guard against zero sigma
        sigma_axial = torch.clamp(sigma_axial, min=0.1)
        sigma_lateral = torch.clamp(sigma_lateral, min=0.1)
        
        psf = torch.exp(-0.5 * ((yy / sigma_axial)**2 + (xx / sigma_lateral)**2))
        psf = psf / torch.sum(psf)
        return psf.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, features, base_img):
        """
        features: (B, C, H, W) - Decoder features before final reconstruction
        base_img: (B, out_channels, H, W) - Initial Denoised image or coarse prediction
        Returns combined fused image consistency representation.
        """
        B, C, H, W = features.shape
        device = features.device
        dtype = features.dtype
        out_c = base_img.shape[1]
        
        # 1. Compute gating weights based on semantic features
        # gates: (B, num_psfs, 1, 1)
        gates = self.gate_mlp(features)
        
        # 2. Generate PSFs dynamically and apply to base_img
        psf_outputs = []
        for i in range(self.num_psfs):
            psf = self._generate_gaussian_kernel(self.sigma_axials[i], self.sigma_laterals[i], device, dtype)
            psf = psf.repeat(out_c, 1, 1, 1)
            pad = self.kernel_size // 2
            
            # Apply PSF to base image
            blurred = F.conv2d(base_img, psf, padding=pad, groups=out_c)
            
            # Weight by gate
            weighted_blurred = blurred * gates[:, i:i+1, :, :]
            psf_outputs.append(weighted_blurred)
            
        # Concat original features and PSF-aware reconstructed priors
        concat_features = torch.cat([features] + psf_outputs, dim=1)
        
        # Final output
        reconstruction = self.fusion_conv(concat_features)
        
        return reconstruction
