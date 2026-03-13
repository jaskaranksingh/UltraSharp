import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WindowAttentionAGA(nn.Module):
    """
    Anisotropic Geodesic Attention (AGA) within local windows.
    Based on Swin Transformer window attention but modulated by Beltrami geodesic proximity.
    Attn(Q,K,V) = softmax(QK^T/sqrt(d_k) + log(W_geo))V
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Learnable temperature for geodesic penalty
        self.tau = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x, bpe_windows):
        """
        x: input features within windows (B*num_windows, N, C), N = window_size * window_size
        bpe_windows: BPE embeddings within windows (B*num_windows, N, K)
                     Used to approximate geodesic distance d_B(i,j) ~ ||bpe_i - bpe_j||_2
        """
        B_, N, C = x.shape
        # QKV linear projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B_, num_heads, N, N)
        
        # Approximate Geodesic Distance d_B(i, j) using L2 distance of BPE encodings
        if bpe_windows is not None:
            # bpe_windows: (B_, N, K)
            bpe_i = bpe_windows.unsqueeze(2) # (B_, N, 1, K)
            bpe_j = bpe_windows.unsqueeze(1) # (B_, 1, N, K)
            
            # Use L2 norm for pairwise geodesic distance
            d_B = torch.norm(bpe_i - bpe_j, p=2, dim=-1) # (B_, N, N)
            d_B = d_B.unsqueeze(1) # (B_, 1, N, N)
            
            # W_geo(i,j) = exp(-d_B(i,j) / tau)
            # log(W_geo) = -d_B(i,j) / tau
            log_W_geo = -d_B / torch.clamp(self.tau, min=1e-6)
            
            # Modulate attention map
            attn = attn + log_W_geo
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
