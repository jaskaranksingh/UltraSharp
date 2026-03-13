import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

def random_gamma_jitter(image, gamma_range=(0.8, 1.2)):
    """
    Random gamma correction to simulate different ultrasound dynamic ranges.
    image: tensor in [0, 1]
    """
    gamma = random.uniform(*gamma_range)
    # add eps to avoid invalid values inside power
    return torch.pow(image + 1e-8, gamma)

def elastic_deformation(image, alpha=10, sigma=3, p=0.2):
    """
    Mild elastic deformation (placeholder implementation).
    """
    if random.random() > p:
        return image
    
    # Ideally should use scipy.ndimage.map_coordinates or grid_sample with random flow
    # This serves as a structural placeholder for the pipeline matching the paper
    return image

def add_gaussian_noise(image, std=0.01, p=0.1):
    """
    Low-variance additive Gaussian noise to emulate system electronic noise.
    """
    if random.random() > p:
        return image
        
    noise = torch.randn_like(image) * std
    noisy = image + noise
    return torch.clamp(noisy, 0.0, 1.0)
