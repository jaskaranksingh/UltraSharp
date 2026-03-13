import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import WindowAttentionAGA


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C) -- note: channels-last convention inside this module
        window_size (int): window size
    Returns:
        windows: (num_windows * B, window_size * window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reconstruct feature map from windows.
    Args:
        windows: (num_windows * B, window_size * window_size, C)
        window_size (int): Window size
        H (int): Height of padded feature map (must be divisible by window_size)
        W (int): Width of padded feature map
    Returns:
        x: (B, H, W, C)
    """
    nH = H // window_size
    nW = W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    """
    Two-layer MLP block: Linear -> GELU -> Linear.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BeltramiTransformerBlock(nn.Module):
    """
    Beltrami Transformer Block (BTB):
        Input (B, C, H, W) -> permute -> LayerNorm -> Window Attention (AGA) -> MLP -> permute -> Output (B, C, H, W)

    The block works internally in channels-last (B, H, W, C) format for attention
    but expects and returns channels-first (B, C, H, W) to be compatible with the
    Conv2d-based encoder/decoder around it.
    """
    def __init__(self, dim, num_heads, window_size=8, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionAGA(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, drop=drop)

    def forward(self, x, bpe_encodings):
        """
        Args:
            x:             (B, C, H, W)  -- channels-first input
            bpe_encodings: (B, K, H, W)  -- BPE positional encodings, same spatial size as x
        Returns:
            x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # --- Convert to channels-last for attention ---
        x_hwc = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        shortcut = x_hwc                         # residual in channels-last

        # LayerNorm (normalized over last dim = C)
        x_hwc = self.norm1(x_hwc)

        # Pad to make H / window_size and W / window_size exact integers
        pad_t = pad_l = 0
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        Hp = H + pad_b
        Wp = W + pad_r

        # x is (B, H, W, C) -- pad dims 1 (H) and 2 (W)
        if pad_b > 0 or pad_r > 0:
            x_hwc = F.pad(x_hwc, (0, 0, 0, pad_r, 0, pad_b))  # pad last 3 dims: C=0,W=pad_r,H=pad_b

        # BPE: (B, K, H, W) -> channels-last (B, H, W, K)
        bpe_hwk = bpe_encodings.permute(0, 2, 3, 1)
        if pad_b > 0 or pad_r > 0:
            bpe_hwk = F.pad(bpe_hwk, (0, 0, 0, pad_r, 0, pad_b))

        # Partition into windows
        x_windows   = window_partition(x_hwc,   self.window_size)   # (B*nW, N, C)
        bpe_windows = window_partition(bpe_hwk,  self.window_size)   # (B*nW, N, K)

        # Window attention with geodesic bias
        attn_windows = self.attn(x_windows, bpe_windows=bpe_windows)  # (B*nW, N, C)

        # Reverse windows
        x_hwc = window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x_hwc = x_hwc[:, :H, :W, :].contiguous()

        # Residual 1
        x_hwc = shortcut + x_hwc

        # MLP with second residual
        x_hwc = x_hwc + self.mlp(self.norm2(x_hwc))

        # Convert back to channels-first (B, C, H, W)
        x = x_hwc.permute(0, 3, 1, 2).contiguous()
        return x
