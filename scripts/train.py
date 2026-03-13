"""
UltraSharp -- Training Entry Point
=====================================
Oral Presentation, IEEE ISBI 2026

"UltraSharp: Beltrami Transformers for Ultrasound Super-Resolution"

NOTE: Full training code, pre-trained checkpoints (model weights), and
evaluation scripts will be released publicly after the ISBI 2026 conference.
The model architecture and loss functions are provided in full.

Please cite our paper if you use this code:
    "UltraSharp: Beltrami Transformers for Ultrasound Super-Resolution"
    Oral, ISBI 2026.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import os
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthesis import PhysicsAwareDegradation
from models.builder import build_ultrasharp
from losses.losses import BeltramiLoss, SpeckleLoss, PhysicsLoss
from utils.structure_tensor import compute_beltrami_metric

# Dataset loaders for CAMUS / EchoNet-Dynamic / BUSI / HC18 will be released
# after the conference together with pre-trained checkpoints.
# For now, supply your own Dataset returning (B, 1, H, W) float32 tensors in [0, 1].

# from data.dataset import UltrasoundDatasetBase   # released post-conference


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    # Full dataset loaders released post-ISBI 2026.
    # Replace the block below with your own DataLoader.
    #
    #   dataset    = UltrasoundDatasetBase(data_dir=args.data_dir,
    #                                      img_size=(256, 256), augment=args.augment)
    #   dataloader = DataLoader(dataset, batch_size=args.batch_size,
    #                           shuffle=True, num_workers=4, pin_memory=True)

    raise NotImplementedError(
        "\n\n  Full training code will be released after ISBI 2026.\n"
        "  Pre-trained weights (.pth) will also be provided for direct inference.\n"
        "\n"
        "  Architecture and losses are fully available in:\n"
        "    models/   losses/   data/synthesis.py\n"
    )

    # Model
    model = build_ultrasharp(args.model, scale=args.scale).to(device)

    # Optimiser and scheduler (Section 4.1 of the paper)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)

    # Loss functions (Equation 6 in the paper)
    l1_loss     = nn.L1Loss()
    ssim_loss   = nn.MSELoss()          # swap for MS-SSIM in full release
    beltrami_fn = BeltramiLoss()
    speckle_fn  = SpeckleLoss()
    degradation = PhysicsAwareDegradation(scale_factor=args.scale).to(device)
    physics_fn  = PhysicsLoss(degradation)

    # Loss weights (lambda values from Table 1)
    lw = dict(l1=1.0, ssim=0.1, bel=0.05, speckle=0.1, physics=0.05)

    # Training loop
    use_amp  = device.type == "cuda"
    scaler   = torch.amp.GradScaler("cuda") if use_amp else None
    best_loss = float("inf")
    patience  = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for hr_imgs in dataloader:
            hr_imgs = hr_imgs.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                lr_imgs = degradation(hr_imgs)
                pred_hr = model(lr_imgs)

                gxx, gxy, gyy, _ = compute_beltrami_metric(pred_hr, alpha=2.0)

                loss = (
                    lw["l1"]       * l1_loss(pred_hr, hr_imgs)
                    + lw["ssim"]   * ssim_loss(pred_hr, hr_imgs)
                    + lw["bel"]    * beltrami_fn(pred_hr, gxx, gxy, gyy)
                    + lw["speckle"] * speckle_fn(pred_hr, hr_imgs)
                    + lw["physics"] * physics_fn(pred_hr, lr_imgs)
                )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{args.epochs}]  loss={avg:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if avg < best_loss - 1e-4:
            best_loss = avg
            patience  = 0
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, "best_model.pth")
                )
        else:
            patience += 1
            if patience >= 20:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UltraSharp training script  |  Oral ISBI 2026"
    )
    parser.add_argument(
        "--model", type=str, default="ultrasharp-b",
        choices=["ultrasharp-t", "ultrasharp-s", "ultrasharp-b", "ultrasharp-l"],
        help="Model variant."
    )
    parser.add_argument("--scale",        type=int,   default=4)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir",     type=str,   default="checkpoints")
    parser.add_argument(
        "--data_dir", type=str, default="./data/train",
        help="Root directory of HR training images."
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable physics-aware augmentations at runtime."
    )
    args = parser.parse_args()
    train(args)
