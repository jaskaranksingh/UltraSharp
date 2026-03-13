import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class PhysicsAwareDegradation(nn.Module):
    """
    Simulates Ultrasound image acquisition pipeline:
    I_LR = S [ (I_HR * h_PSF) * n_speckle ]
    """
    def __init__(self, scale_factor=4, apply_speckle_prob=0.7, kernel_size=15):
        super().__init__()
        self.scale_factor = scale_factor
        self.apply_speckle_prob = apply_speckle_prob
        self.kernel_size = kernel_size
        
        # PSF Bank (sigma_axial, sigma_lateral)
        self.psf_params = [
            (0.5, 1.0),
            (0.8, 1.5),
            (1.1, 2.0),
            (1.4, 2.5)
        ]
        
    def generate_anisotropic_psf(self, sigma_axial, sigma_lateral, device='cpu', dtype=torch.float32):
        """
        Creates an anisotropic 2D Gaussian kernel (PSF).
        Assume axial is height (y), lateral is width (x).
        """
        half_size = self.kernel_size // 2
        y = torch.arange(-half_size, half_size + 1, device=device, dtype=dtype)
        x = torch.arange(-half_size, half_size + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 2D Gaussian
        psf = torch.exp(-0.5 * ((yy / sigma_axial)**2 + (xx / sigma_lateral)**2))
        psf = psf / psf.sum()
        
        # Shape for F.conv2d: (out_channels, in_channels, H, W) -> (1, 1, H, W)
        return psf.view(1, 1, self.kernel_size, self.kernel_size)

    def generate_rayleigh_noise(self, shape, scale=0.5, device='cpu', dtype=torch.float32):
        """
        Generates Rayleigh distributed noise.
        Rayleigh(scale) = scale * sqrt(-2 * ln(U)), where U ~ Uniform(0, 1)
        """
        u = torch.rand(shape, device=device, dtype=dtype)
        # Avoid log(0)
        u = torch.clamp(u, min=1e-8)
        rayleigh = scale * torch.sqrt(-2.0 * torch.log(u))
        return rayleigh

    def forward(self, x):
        """
        x: HR image tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        # 1. Convolution with random anisotropic PSF from bank
        idx = random.randint(0, len(self.psf_params) - 1)
        sigma_axial, sigma_lateral = self.psf_params[idx]
        
        psf = self.generate_anisotropic_psf(sigma_axial, sigma_lateral, device, dtype)
        psf = psf.repeat(C, 1, 1, 1) # Support C channels via groups
        
        pad = self.kernel_size // 2
        # Apply convolution (I_HR * h_PSF)
        blurred = F.conv2d(x, psf, padding=pad, groups=C)
        
        # 2. Multiplicative Speckle Noise
        # n_speckle = 1 + 0.1 * Rayleigh(0.5)
        batch_out = []
        for i in range(B):
            sample = blurred[i:i+1] # shape (1, C, H, W)
            if random.random() < self.apply_speckle_prob:
                noise = self.generate_rayleigh_noise(sample.shape, scale=0.5, device=device, dtype=dtype)
                n_speckle = 1.0 + 0.1 * noise
                sample = sample * n_speckle
            batch_out.append(sample)
                
        noisy_blurred = torch.cat(batch_out, dim=0)
        
        # 3. Downsampling (S)
        # Using bicubic or bilinear to simulate spatial downsampling
        lr_img = F.interpolate(noisy_blurred, scale_factor=1.0/self.scale_factor, mode='bicubic', align_corners=False)
        
        # Clamp to [0, 1] range
        lr_img = torch.clamp(lr_img, 0.0, 1.0)
        
        return lr_img
