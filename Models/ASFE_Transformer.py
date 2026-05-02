from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

try:
    from torch.nn.init import trunc_normal_
except Exception:
    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(a, b)


def hilbert_transform_fft(x: torch.Tensor) -> torch.Tensor:
    Xf = torch.fft.fft(x, dim=-1)
    T = x.size(-1)
    h = torch.zeros(T, dtype=Xf.dtype, device=Xf.device)
    if T % 2 == 0:
        h[0] = 1.0
        h[T // 2] = 1.0
        h[1:T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(T + 1) // 2] = 2.0
    return torch.fft.ifft(Xf * h, dim=-1).imag


class AnalyticAmplitudePhaseRepresentationModule(nn.Module):
    def __init__(self, feature_mode: str = "IQ"):
        super().__init__()
        self.feature_mode = feature_mode.upper()
        if self.feature_mode not in {"IQ", "AP"}:
            raise ValueError("feature_mode must be 'IQ' or 'AP'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"AAPR expects (B,T) or (B,C,T), got {tuple(x.shape)}")

        x_quad = hilbert_transform_fft(x)
        if self.feature_mode == "IQ":
            pair = torch.stack([x, x_quad], dim=-1)
        else:
            analytic = torch.complex(x, x_quad)
            pair = torch.stack([torch.angle(analytic), torch.abs(analytic)], dim=-1)

        B, C, T, _ = pair.shape
        return pair.permute(0, 2, 1, 3).reshape(B, T, 2 * C)


class AcousticFrameEmbeddingModule(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        input_length: int = 1024,
        frame_length: int = 16,
        overlap: float = 0.5,
        hop_size: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channel = int(in_channel)
        self.input_length = int(input_length)
        self.frame_length = int(frame_length)
        self.hop_size = int(hop_size) if hop_size is not None else int(round(self.frame_length * (1.0 - overlap)))

        if self.in_channel < 1:
            raise ValueError("in_channel must be >= 1.")
        if self.frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if self.hop_size <= 0:
            raise ValueError("hop_size must be positive.")
        if self.input_length < self.frame_length:
            raise ValueError("input_length must be >= frame_length.")

        self.num_frame_tokens = (self.input_length - self.frame_length) // self.hop_size + 1
        self.token_dim = self.frame_length * 2 * self.in_channel
        self.frame_projection = nn.Linear(self.token_dim, self.token_dim, bias=bias)

    def forward(self, analytic_features: torch.Tensor) -> torch.Tensor:
        if analytic_features.dim() != 3:
            raise ValueError(f"AFEM expects (B,T,2C), got {tuple(analytic_features.shape)}")

        B, T, feature_dim = analytic_features.shape
        expected_feature_dim = 2 * self.in_channel
        if feature_dim != expected_feature_dim:
            raise ValueError(f"Expected feature dimension {expected_feature_dim}, got {feature_dim}.")
        if T < self.frame_length:
            raise ValueError("Input sequence is shorter than frame_length.")

        frames = analytic_features.unfold(dimension=1, size=self.frame_length, step=self.hop_size)
        frames = frames.contiguous().view(B, frames.size(1), feature_dim * self.frame_length)
        frames = frames[:, :self.num_frame_tokens, :]
        return self.frame_projection(frames)


class LowDimensionalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        d_fix_qk: int = 16,
        d_fix_v: int = 16,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
        talking: bool = False,
        attn_res: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_fix_qk = int(d_fix_qk)
        self.d_fix_v = int(d_fix_v)
        self.num_heads = int(num_heads)
        self.attn_res = bool(attn_res)
        self.scale = math.sqrt(float(self.d_fix_qk))

        self.to_q = nn.Linear(self.d_model, self.d_fix_qk * self.num_heads, bias=bias)
        self.to_k = nn.Linear(self.d_model, self.d_fix_qk * self.num_heads, bias=bias)
        self.to_v = nn.Linear(self.d_model, self.d_fix_v * self.num_heads, bias=bias)
        self.head_mixer = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, bias=False) if talking else nn.Identity()
        self.proj_out = nn.Linear(self.d_fix_v * self.num_heads, self.d_model, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = x.shape
        q = self.to_q(x).reshape(B, N, self.num_heads, self.d_fix_qk).permute(0, 2, 1, 3) / self.scale
        k = self.to_k(x).reshape(B, N, self.num_heads, self.d_fix_qk).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, N, self.num_heads, self.d_fix_v).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.head_mixer(attn)
        if self.attn_res and attn_prev is not None:
            attn = attn + attn_prev
        attn = self.softmax(attn)
        attn = self.drop_attn(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.d_fix_v)
        out = self.drop_out(self.proj_out(out))
        return out, attn


class SwishDepthwiseGatedLinearUnit(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        dim_feedforward: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_feedforward % 2 != 0:
            raise ValueError("dim_feedforward must be even for SD-GLU.")
        self.hidden_dim = dim_feedforward // 2
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.dwconv = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=self.hidden_dim,
            bias=True,
        )
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        value, gate = h[..., :self.hidden_dim], h[..., self.hidden_dim:]
        value = self.dwconv(value.transpose(1, 2)).transpose(1, 2)
        y = value * self.swish(gate)
        return self.fc2(self.dropout(y))


class ASFETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        d_fix_qk: int = 16,
        d_fix_v: int = 16,
        num_heads: int = 4,
        dim_feedforward: int = 64,
        sdglu_kernel_size: int = 5,
        dropout: float = 0.0,
        bias: bool = False,
        talking: bool = False,
        real_former: bool = False,
    ):
        super().__init__()
        self.real_former = bool(real_former)
        self.lmhsa = LowDimensionalMultiHeadSelfAttention(
            d_model=d_model,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            talking=talking,
            attn_res=real_former,
        )
        self.sdglu = SwishDepthwiseGatedLinearUnit(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            kernel_size=sdglu_kernel_size,
            dropout=dropout,
        )
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, attn_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_attn, attn = self.lmhsa(x, attn_prev if self.real_former else None)
        x = self.norm_mhsa(x + x_attn)
        x = self.norm_ffn(x + self.sdglu(x))
        return x, attn


class ASFETransformerEncoder(nn.Module):
    def __init__(self, num_layers: int = 8, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([ASFETransformerEncoderLayer(**layer_kwargs) for _ in range(int(num_layers))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = None
        for layer in self.layers:
            x, attn = layer(x, attn)
        return x


class ASFE_Transformer(nn.Module):
    model_name = "ASFE-Transformer"

    def __init__(
        self,
        _unused=None,
        in_channel: int = 1,
        out_channel: int = 5,
        input_length: int = 1024,
        frame_length: int = 16,
        overlap: float = 0.5,
        hop_size: Optional[int] = None,
        layer_num: int = 8,
        d_fix_qk: int = 16,
        d_fix_v: int = 16,
        num_heads: int = 4,
        hidden_features: Optional[int] = None,
        sdglu_kernel_size: int = 5,
        dropout: float = 0.2,
        pos_emb: bool = True,
        bias: bool = False,
        talking: bool = False,
        real_former: bool = False,
        feature_mode: str = "IQ",
    ):
        super().__init__()
        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        self.input_length = int(input_length)
        self.frame_length = int(frame_length)
        self.hop_size = int(hop_size) if hop_size is not None else int(round(self.frame_length * (1.0 - overlap)))
        self.token_dim = self.frame_length * 2 * self.in_channel
        self.pos_emb = bool(pos_emb)

        self.AAPR = AnalyticAmplitudePhaseRepresentationModule(feature_mode=feature_mode)
        self.AFEM = AcousticFrameEmbeddingModule(
            in_channel=self.in_channel,
            input_length=self.input_length,
            frame_length=self.frame_length,
            overlap=overlap,
            hop_size=self.hop_size,
            bias=bias,
        )
        self.num_frame_tokens = self.AFEM.num_frame_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_tokens + 1, self.token_dim))

        dim_feedforward = int(hidden_features) if hidden_features is not None else 4 * self.frame_length
        self.encoder = ASFETransformerEncoder(
            num_layers=layer_num,
            d_model=self.token_dim,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            sdglu_kernel_size=sdglu_kernel_size,
            dropout=dropout,
            bias=bias,
            talking=talking,
            real_former=real_former,
        )
        self.classifier = nn.Linear(self.token_dim, self.out_channel)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.AAPR(x)
        x = self.AFEM(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, cls_token], dim=1)
        if self.pos_emb:
            x = x + self.pos_embed
        x = self.encoder(x)
        return self.classifier(x[:, -1])
