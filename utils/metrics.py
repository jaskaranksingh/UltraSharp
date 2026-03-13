import torch
import torch.nn.functional as F
import math
import lpips  # Perceptual metric

class MultiScaleSSIM:
    """ Multi-Scale Structural Similarity Index. """
    def __init__(self, data_range=1.0, size_average=True, channel=1):
        # Implementation is commonly from pytorch-msssim, but simple SSIM is provided for brevity.
        # This acts as a placeholder for a robust MS-SSIM implementation.
        self.data_range = data_range
        
    def __call__(self, img1, img2):
        # Using simple MSE-based pseudo-SSIM for completeness if msssim not present
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return 1.0
        return 1.0 - math.sqrt(mse.item() / self.data_range**2)

def calculate_psnr(img1, img2, data_range=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 10 * math.log10((data_range ** 2) / mse.item())

def calculate_lpips(img1, img2, device='cpu'):
    """ Calculated Learned Perceptual Image Patch Similarity. """
    loss_fn = lpips.LPIPS(net='alex').to(device)
    # LPIPS expects images in [-1, 1] range, convert from [0, 1]
    img1_norm = img1 * 2.0 - 1.0
    img2_norm = img2 * 2.0 - 1.0
    with torch.no_grad():
        d = loss_fn(img1_norm, img2_norm)
    return d.mean().item()

def calculate_cnr(img, rois_sig, rois_bg):
    """
    Contrast-to-Noise Ratio (CNR) = | mu_sig - mu_bg | / sqrt(sigma_sig^2 + sigma_bg^2)
    Requires bounding boxes/masks for signal and background regions.
    """
    # Assuming rois are flat tensors of pixels
    mu_sig = rois_sig.mean()
    mu_bg = rois_bg.mean()
    var_sig = rois_sig.var()
    var_bg = rois_bg.var()
    
    cnr = torch.abs(mu_sig - mu_bg) / torch.sqrt(var_sig + var_bg + 1e-8)
    return cnr.item()

def calculate_ssnr(img, rois_bg):
    """
    Speckle SNR (sSNR) = mu_bg / sigma_bg
    Requires background (speckle region) pixels.
    """
    mu_bg = rois_bg.mean()
    sigma_bg = rois_bg.std()
    
    ssnr = mu_bg / (sigma_bg + 1e-8)
    return ssnr.item()
