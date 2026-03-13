"""
UltraSharp Model Builder
========================
Oral Presentation, IEEE ISBI 2026

"UltraSharp: Beltrami Transformers for Ultrasound Super-Resolution"

This module provides a factory function for instantiating UltraSharp models
across four capacity configurations. Pre-trained checkpoints will be released
after ISBI 2026.
"""

import torch
from models.ultrasharp import UltraSharp


# Model capacity configurations.
# All variants share the same Beltrami Transformer Block architecture (BTB + BPE + PCF)
# and differ only in embedding width, number of attention heads, and block depth.

VARIANTS = {
    "ultrasharp-t": dict(dim=32,  num_heads=4,  num_blocks=[1, 1, 1], K_bpe=4),
    "ultrasharp-s": dict(dim=48,  num_heads=6,  num_blocks=[2, 2, 1], K_bpe=6),
    "ultrasharp-b": dict(dim=64,  num_heads=8,  num_blocks=[2, 2, 2], K_bpe=8),
    "ultrasharp-l": dict(dim=96,  num_heads=12, num_blocks=[3, 3, 3], K_bpe=12),
}


def build_ultrasharp(variant: str = "ultrasharp-b",
                     scale: int = 4,
                     pretrained: bool = False,
                     checkpoint: str = None) -> UltraSharp:
    """
    Instantiate an UltraSharp model.

    Args:
        variant    (str):  Model size. One of 'ultrasharp-t', 'ultrasharp-s',
                           'ultrasharp-b' (default), 'ultrasharp-l'.
        scale      (int):  Super-resolution factor. Supported values: 2, 4, 8.
        pretrained (bool): Load official pre-trained weights. Weights will be provided
                           after ISBI 2026.
        checkpoint (str):  Path to a locally saved checkpoint (.pth). Optional.

    Returns:
        UltraSharp: Configured model instance.

    Example:
        from models.builder import build_ultrasharp
        model = build_ultrasharp("ultrasharp-b", scale=4)
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Available options: {list(VARIANTS.keys())}"
        )

    cfg = VARIANTS[variant]
    model = UltraSharp(scale=scale, **cfg)

    if pretrained:
        # Official weights to be released at https://github.com/YourOrg/UltraSharp
        raise NotImplementedError(
            "Pre-trained weights will be released after ISBI 2026. "
            "Please check the repository for the release announcement."
        )

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded checkpoint from: {checkpoint}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UltraSharp [{variant}]  |  {n_params / 1e6:.1f}M parameters  |  scale x{scale}")
    return model


def list_variants():
    """Print a summary table of available model variants."""
    print("\nUltraSharp Model Variants")
    print("=" * 62)
    print(f"{'Variant':<18}{'dim':>6}{'heads':>7}{'blocks':>14}{'Params':>10}")
    print("-" * 62)
    sizes = {
        "ultrasharp-t": "~5M",
        "ultrasharp-s": "~11M",
        "ultrasharp-b": "~22M (paper default)",
        "ultrasharp-l": "~45M",
    }
    for name, cfg in VARIANTS.items():
        print(
            f"{name:<18}{cfg['dim']:>6}{cfg['num_heads']:>7}"
            f"{str(cfg['num_blocks']):>14}{sizes[name]:>10}"
        )
    print()


if __name__ == "__main__":
    list_variants()
    model = build_ultrasharp("ultrasharp-b", scale=4)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {tuple(x.shape)}   Output shape: {tuple(out.shape)}")
