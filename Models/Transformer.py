# Transformer.py
# -*- coding: utf-8 -*-
"""
Grid-reshape Transformer for 1D acoustic sequences.
- Input:  (B, C, T) or (B, T)
- Reshape: 1×1024 -> 32×32  (sequence_len=32, feature_dim=32)
- Embedding: Linear(feature_dim -> d_model)
- Encoder: nn.TransformerEncoder (batch_first=True, Pre-LN)
- Head:   LayerNorm -> AvgPool over sequence -> Linear(num_classes)

This mirrors the structure used in Transformer信号分类.ipynb:
    view(B, 32, 32) -> Transformer -> LayerNorm -> AvgPool -> FC
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn


class GridTransformer1D(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 total_len: int = 1024,    # 原始时长 T，应当能被 (seq_len * feature_dim) 整除
                 seq_len: int = 32,        # 重排后的序列长度（步数）
                 feature_dim: int = 32,    # 重排后的每步特征维度
                 d_model: int = 128,       # 注意力维度
                 num_layers: int = 4,
                 nhead: int = 8,
                 dropout: float = 0.5,
                 norm_first: bool = True,  # Pre-LN，与 notebook 文本一致
                 use_positional_encoding: bool = False  # notebook 里没有显式 PE，因此默认 False
                 ):
        super().__init__()
        assert seq_len * feature_dim <= total_len, \
            "seq_len * feature_dim must be <= total_len; usually == 1024 for 32×32."

        self.num_classes = num_classes
        self.total_len = total_len
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model

        # 1) 线性投影：将 feature_dim -> d_model
        self.embed = nn.Linear(feature_dim, d_model)

        # 2) （可选）位置编码——notebook 结构里没有；默认关闭
        if use_positional_encoding:
            self.pos_embed = SinusoidalPE1D(d_model, max_len=max(seq_len, 8192))
        else:
            self.pos_embed = None

        # 3) Transformer Encoder（与 notebook 一致，batch_first=True）
        ff_dim = d_model * 4
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=norm_first
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 4) 编码后归一化 + 沿序列方向的全局平均池化
        self.norm = nn.LayerNorm(d_model)
        self.avgpool = nn.AvgPool1d(kernel_size=seq_len)  # 池化到 1

        # 5) 分类头
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _reshape_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        支持 (B, C, T) 或 (B, T) 两种输入，重排为 (B, seq_len, feature_dim)。
        - 典型：C=1, T=1024, seq_len=32, feature_dim=32
        """
        if x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, -1)  # (B, C*T)
        elif x.dim() == 2:
            B, T = x.shape
        else:
            raise ValueError("Input must be (B,C,T) or (B,T).")

        need = self.seq_len * self.feature_dim
        if T < need:
            # 若 T 比 32×32 短，右侧用零填充
            pad = need - T
            x = torch.nn.functional.pad(x, (0, pad))
            T = need

        # 裁剪到整块并重排
        x = x[:, :need]                       # (B, need)
        x = x.view(-1, self.seq_len, self.feature_dim)  # (B, L=32, F=32)
        return x

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        x: (B, C, T) 或 (B, T)
        return: (B, num_classes)
        """
        # 1) 重排到 (B, L, F)
        x = self._reshape_to_grid(x)

        # 2) Linear 嵌入到 d_model
        x = self.embed(x)  # (B, L, d_model)

        # 3) 可选位置编码
        if self.pos_embed is not None:
            x = self.pos_embed(x)

        # 4) Transformer 编码
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # 5) LayerNorm + AvgPool1d(沿 L)
        x = self.norm(x)                    # (B, L, d_model)
        x = self.avgpool(x.transpose(1, 2)) # (B, d_model, 1)
        x = x.view(x.size(0), -1)           # (B, d_model)

        # 6) 分类
        logits = self.head(x)               # (B, num_classes)
        return logits


# （可选）正弦位置编码：默认关闭以贴合 notebook
class SinusoidalPE1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 65536):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)     # (1,L,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# 为旧脚本提供稳定别名（与之前的 Transformer1D_small 一致的“体量”）
def Transformer1D_small(_unused=None,
                        num_classes: int = 10,
                        total_len: int = 1024) -> nn.Module:
    return GridTransformer1D(
        num_classes=num_classes,
        total_len=total_len,
        seq_len=32,
        feature_dim=32,
        d_model=128,
        num_layers=4,
        nhead=8,
        dropout=0.5,
        norm_first=True,
        use_positional_encoding=False
    )


# Minimal self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T = 4, 1, 1024
    num_classes = 10
    model = Transformer1D_small(None, num_classes=num_classes, total_len=T)
    x = torch.randn(B, C, T)
    y = model(x)
    print("logits:", y.shape)  # (B, num_classes)
