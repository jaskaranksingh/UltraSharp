import torch
import torch.nn as nn
import torch.nn.functional as F

class BeltramiLoss(nn.Module):
    """
    Beltrami Regularization Loss.
    L_Bel = mean(sqrt(gxx * dx^2 + gyy * dy^2 + 2 * gxy * dx * dy))
    where [gxx, gxy; gxy, gyy] is the inverse Beltrami metric computed on the LR/HR images.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_hr, gxx, gxy, gyy):
        """
        pred_hr: (B, 1, H, W)
        gxx, gxy, gyy: metric components from utils.structure_tensor (B, 1, H, W)
        """
        # Compute spatial gradients of the predicted HR image using finite differences
        dx = pred_hr - torch.roll(pred_hr, shifts=1, dims=-1)
        dy = pred_hr - torch.roll(pred_hr, shifts=1, dims=-2)
        
        # Eliminate boundary roll effects
        dx[:, :, :, 0] = 0.0
        dy[:, :, 0, :] = 0.0
        
        # Beltrami norm: sqrt(gxx dx^2 + gyy dy^2 + 2 gxy dx dy)
        b_norm_sq = gxx * (dx ** 2) + gyy * (dy ** 2) + 2.0 * gxy * dx * dy
        
        loss = torch.sqrt(torch.clamp(b_norm_sq, min=self.eps))
        return loss.mean()

class SpeckleLoss(nn.Module):
    """
    Speckle Statistics Loss (Rayleigh KL Divergence)
    L_speckle = mean(log(sigma / sigma_hat) + (sigma_hat^2) / (2*sigma^2) - 0.5)
    """
    def __init__(self, patch_size=8, eps=1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps
        
    def forward(self, pred_hr, target_hr):
        """
        Estimate localized Rayleigh scale parameters sigma over small patches
        For Rayleigh distribution, sigma ~ mean / sqrt(pi/2)
        """
        B, C, H, W = pred_hr.shape
        # unfold into patches stringing spatial patches
        # stride=patch_size for non-overlapping
        pred_patches = F.unfold(pred_hr, kernel_size=self.patch_size, stride=self.patch_size)
        target_patches = F.unfold(target_hr, kernel_size=self.patch_size, stride=self.patch_size)
        
        # pred_patches: (B, patch_size*patch_size*C, num_patches)
        # Using Rayleigh mean relation to estimate sigma locally: E[X] = sigma * sqrt(pi/2)
        # Thus sigma_hat = mean / sqrt(pi/2)
        const = torch.sqrt(torch.tensor(torch.pi / 2.0, device=pred_hr.device))
        
        sigma_hat = torch.mean(pred_patches + self.eps, dim=1) / const
        sigma_target = torch.mean(target_patches + self.eps, dim=1) / const
        
        sigma_hat = torch.clamp(sigma_hat, min=self.eps)
        sigma_target = torch.clamp(sigma_target, min=self.eps)
        
        kl_div = torch.log(sigma_target / sigma_hat) + (sigma_hat ** 2) / (2.0 * sigma_target ** 2) - 0.5
        kl_div = torch.relu(kl_div) # KL divergence is non-negative
        
        return kl_div.mean()

class PhysicsLoss(nn.Module):
    """
    Physics Cycle-Consistency Loss
    L_phys = || D(I_HR_hat) - I_LR ||_1
    """
    def __init__(self, degradation_model):
        super().__init__()
        self.degradation = degradation_model
        self.l1 = nn.L1Loss()
        
    def forward(self, pred_hr, target_lr):
        """
        pred_hr: (B, C, H*scale, W*scale) - predicted high-res image
        target_lr: (B, C, H, W) - input low-res image
        """
        # Degrade the predicted HR image back to LR
        # Since degradation has randomness (speckle, blur), ideally we want to
        # use the *same* parameters used to generate LR if possible, or expect statistical consistency
        # Assuming degradation model handles it correctly:
        recon_lr = self.degradation(pred_hr)
        
        return self.l1(recon_lr, target_lr)
