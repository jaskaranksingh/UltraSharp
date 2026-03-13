"""
UltraSharp Dataset Utilities
==============================
Oral ISBI 2026 -- "UltraSharp: Beltrami Transformers for Ultrasound Super-Resolution"

NOTE
----
Full dataset loaders for CAMUS, EchoNet-Dynamic, BUSI, and HC18 --
including train/val/test splits, preprocessing, and augmentation pipelines --
will be released alongside pre-trained checkpoints **after ISBI 2026**.

The physics-aware degradation pipeline (data/synthesis.py) is fully available.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Augmentation helpers (data/transforms.py)
# ---------------------------------------------------------------------------
from data.transforms import random_gamma_jitter, elastic_deformation, add_gaussian_noise


class UltrasoundDatasetBase(Dataset):
    """
    Base dataset for loading high-resolution (HR) ultrasound images.

    Supports any directory of grayscale images (PNG / JPG / BMP / TIF).
    Low-resolution (LR) images are synthesised **on-the-fly** at training time
    by the PhysicsAwareDegradation pipeline (``data/synthesis.py``);
    this class returns only the HR images.

    Args:
        data_dir (str): Root directory containing image files.
        img_size (tuple): Spatial size to resize all images to (H, W).
                          Default: (256, 256).
        augment (bool):  Enable physics-aware augmentations at training time.
                         Default: False.

    NOTE
    ----
    Dataset-specific loaders for CAMUS, EchoNet-Dynamic, BUSI, and HC18
    (with ground-truth segmentation masks for CNR / sSNR evaluation)
    will be released after ISBI 2026.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, data_dir: str, img_size=(256, 256), augment: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment  = augment

        self.image_paths = sorted([
            p for p in self.data_dir.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            # Graceful fallback -- generate a tiny synthetic set so scripts
            # can be imported and tested without real data.
            self._synthetic = True
            self._length    = 8
        else:
            self._synthetic = False
            self._length    = len(self.image_paths)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._synthetic:
            # Synthetic placeholder -- replace with real data for experiments
            return torch.rand(1, *self.img_size)

        img = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BICUBIC)
        img = self.to_tensor(img)  # (1, H, W), float in [0, 1]

        if self.augment:
            img = self._apply_augmentations(img)

        return img

    def _apply_augmentations(self, img: torch.Tensor) -> torch.Tensor:
        """Physics-aware augmentation chain (Section 3.3 of the paper)."""
        import random
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-1])           # horizontal flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-2])           # vertical flip
        img = random_gamma_jitter(img, gamma_range=(0.8, 1.2))
        if random.random() > 0.5:
            img = elastic_deformation(img, alpha=34, sigma=4)
        img = add_gaussian_noise(img, std=0.02)
        return img.clamp(0.0, 1.0)
