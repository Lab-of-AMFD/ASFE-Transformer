# VitTransformer1D.py
# -*- coding: utf-8 -*-
"""
A standard Transformer (Encoder-only) for 1D sequences.
- Input:  (B, C, T)
- Patch Embedding: Conv1d with stride to create tokens (B, L, D)
- Positional Encoding: Sinusoidal (fixed)
- Encoder: nn.TransformerEncoder (batch_first=True)
- Head:   CLS token or mean pool -> Linear(num_classes)

Author: you :)
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Sinusoidal Positional Encoding
# -----------------------------
class SinusoidalPE1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)          # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))                       # (D/2,)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)              # (1,L,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


# -----------------------------
# Patch Embedding (1D)
# -----------------------------
class PatchEmbed1D(nn.Module):
    """
    (B, C, T) -> (B, L, D)
    Using Conv1d with kernel_size=patch_size and stride=stride to form tokens.
    If overlap>0, stride < patch_size (i.e., overlapping patches).
    """
    def __init__(self,
                 in_chans: int = 1,
                 embed_dim: int = 128,
                 patch_size: int = 32,
                 overlap: float = 0.5,
                 bias: bool = True):
        super().__init__()
        assert 0 <= overlap < 1, "overlap must be in [0,1)."
        stride = max(1, int(round(patch_size * (1 - overlap))))
        padding = 0  # no padding to keep definition clear (can change to 'same' if needed)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=bias)
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, L, D)
        """
        x = self.proj(x)           # (B, D, L)
        x = x.transpose(1, 2)      # (B, L, D)
        return x

    def tokens_length(self, seq_len: int) -> int:
        # L = floor((T - kernel)/stride) + 1  with padding=0
        return max(0, (seq_len - self.patch_size) // self.stride + 1)


# -----------------------------
# Standard Transformer 1D
# -----------------------------
class StandardTransformer1D(nn.Module):
    def __init__(self,
                 in_chans: int = 1,
                 num_classes: int = 5,
                 seq_len: int = 1024,
                 # patch embedding
                 patch_size: int = 32,
                 overlap: float = 0.5,
                 embed_dim: int = 128,
                 # transformer encoder
                 depth: int = 8,
                 n_head: int = 4,
                 mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.1,
                 # head
                 use_cls_token: bool = True,
                 pool_type: str = "mean"  # used when use_cls_token=False
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.use_cls = bool(use_cls_token)
        assert pool_type in ("mean", "max"), "pool_type must be one of {'mean','max'}"
        self.pool_type = pool_type

        # 1) Patch-Embedding
        self.patch_embed = PatchEmbed1D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            overlap=overlap,
            bias=True
        )
        token_len = self.patch_embed.tokens_length(seq_len)
        if token_len <= 0:
            raise ValueError(f"Token length becomes 0. Try smaller patch_size or larger seq_len. "
                             f"(seq_len={seq_len}, patch_size={patch_size}, stride={self.patch_embed.stride})")

        # 2) Optional CLS token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # 3) Positional Encoding (fixed sinusoidal)
        self.pos_embed = SinusoidalPE1D(embed_dim, max_len=token_len + (1 if self.use_cls else 0))

        # 4) Transformer Encoder
        dim_feedforward = int(embed_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False  # <— 显式关闭以消除提示
        )

        # Pre/post norms
        self.pre_ln = nn.LayerNorm(embed_dim)
        self.post_ln = nn.LayerNorm(embed_dim)

        # 5) Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Init
        self.apply(self._init_weights)
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: logits (B, num_classes)
        """
        B = x.size(0)

        # (B, L, D)
        x = self.patch_embed(x)

        # optional CLS
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)     # (B,1,D)
            x = torch.cat([cls, x], dim=1)             # (B, 1+L, D)

        # pre-norm + pe
        x = self.pre_ln(x)
        x = self.pos_embed(x)

        # transformer encoder
        x = self.encoder(x)                            # (B, 1+L or L, D)

        # post-norm
        x = self.post_ln(x)

        # pooling / cls head
        if self.use_cls:
            feat = x[:, 0, :]                          # CLS
        else:
            if self.pool_type == "mean":
                feat = x.mean(dim=1)
            else:
                feat, _ = x.max(dim=1)

        logits = self.head(feat)                       # (B, num_classes)
        return logits


# -----------------------------
# Backward-compat constructor
# -----------------------------
def StandardTransformer1D_small(_unused=None,
                                in_chans: int = 1,
                                num_classes: int = 5,
                                seq_len: int = 1024) -> nn.Module:
    """
    Convenience wrapper to mimic other backbones' constructor signatures:
        Model(None, in_ch, num_cls)
    """
    return StandardTransformer1D(
        in_chans=in_chans,
        num_classes=num_classes,
        seq_len=seq_len,
        patch_size=32,
        overlap=0.5,
        embed_dim=128,
        depth=8,
        n_head=4,
        mlp_ratio=4.0,
        dropout=0.1,
        use_cls_token=True
    )
