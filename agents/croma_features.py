"""Frozen CROMA radar features, not a vessel detector or HH/HV adapter.

Radar-only architecture adapted from Anthony Fuller's MIT-licensed CROMA
use_croma.py at 59505a6bcadbf36ba20767270154bf9f3067c5e7.
See third_party/CROMA_LICENSE.txt. Checkpoint keys remain upstream-compatible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from agents.artifacts import sha256_file

WEIGHTS_SHA256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"
WEIGHTS_REVISION = "0dd28e3d633bd6715856ae9890e8c49360040598"
WEIGHTS_URL = (
    "https://huggingface.co/antofuller/CROMA/resolve/"
    f"{WEIGHTS_REVISION}/CROMA_base.pt"
)
SOURCE_REVISION = "59505a6bcadbf36ba20767270154bf9f3067c5e7"
WINDOW_SIZE = 128


def normalize_window(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Independent per-window VV/VH mean +/- 2 sample-SD stretch to [0,1].

    Upstream README's float normalization, with batch size one, no 8-bit
    quantization. No statistics are fitted across training/validation scenes.
    Constant channels are rejected rather than silently producing NaN features.
    """
    x = np.asarray(values, dtype=np.float32)
    if x.shape != (2, WINDOW_SIZE, WINDOW_SIZE) or not np.isfinite(x).all():
        raise ValueError("Finite two-band 128px VV/VH window required")
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), ddof=1, keepdims=True)
    if np.any(std <= 1e-6):
        raise ValueError("Constant SAR channel cannot be normalized")
    return np.asarray(np.clip((x - (mean - 2 * std)) / (4 * std), 0, 1))


def build_encoder(*, dim: int = 768, depth: int = 6) -> Any:
    """Build upstream-compatible SAR encoder without optical/cross encoders."""
    import torch
    from torch import nn

    if dim % 16 or depth < 1:
        raise ValueError("CROMA requires 16 attention heads and positive depth")

    class Attention(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Linear(dim, dim)
            self.input_norm = nn.LayerNorm(dim)

        def forward(self, x: Any, bias: Any) -> Any:
            batch, tokens, _ = x.shape
            qkv = self.to_qkv(self.input_norm(x)).chunk(3, dim=-1)
            q, k, v = (
                a.reshape(batch, tokens, 16, dim // 16).transpose(1, 2) for a in qkv
            )
            scores = (q @ k.transpose(-1, -2)) * ((dim // 16) ** -0.5) + bias
            out = (scores.softmax(dim=-1) @ v).transpose(1, 2)
            return self.to_out(out.reshape(batch, tokens, dim))

    class FeedForward(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0),
                nn.Linear(dim * 4, dim),
            )
            self.input_norm = nn.LayerNorm(dim)

        def forward(self, x: Any) -> Any:
            return self.net(self.input_norm(x))

    class Transformer(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.ModuleList([Attention(), FeedForward()]) for _ in range(depth)]
            )
            self.norm_out = nn.LayerNorm(dim)

        def forward(self, x: Any, bias: Any) -> Any:
            for attention, ffn in self.layers:
                x = attention(x, bias) + x
                x = ffn(x) + x
            return self.norm_out(x)

    class RadarViT(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear_input = nn.Linear(2 * 8 * 8, dim)
            self.transformer = Transformer()

        def forward(self, x: Any, bias: Any) -> Any:
            batch, _, height, width = x.shape
            x = x.reshape(batch, 2, height // 8, 8, width // 8, 8)
            x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, 128)
            return self.transformer(self.linear_input(x), bias)

    class RadarEncoder(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.s1_encoder = RadarViT()
            self.GAP_FFN_s1 = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim),
            )
            axis = torch.arange(WINDOW_SIZE // 8, dtype=torch.float32)
            points = torch.cartesian_prod(axis, axis)
            distances = torch.cdist(points, points)
            slopes = torch.tensor([2 ** (-0.5 * (i + 1)) for i in range(16)])
            self.register_buffer("bias", -slopes[None, :, None, None] * distances)

        def forward(self, x: Any) -> dict[str, Any]:
            tokens = self.s1_encoder(x, self.bias)
            return {
                "SAR_encodings": tokens,
                "SAR_GAP": self.GAP_FFN_s1(tokens.mean(dim=1)),
            }

    return RadarEncoder()


def parameter_digest(model: Any) -> str:
    """Detect parameter changes in the actual model, not only checkpoint bytes."""
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def load_encoder(path: Path, device: str) -> Any:
    import torch

    if sha256_file(path) != WEIGHTS_SHA256:
        raise ValueError("CROMA checkpoint does not match pinned public weights")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = build_encoder()
    model.s1_encoder.load_state_dict(state["s1_encoder"], strict=True)
    model.GAP_FFN_s1.load_state_dict(state["s1_GAP_FFN"], strict=True)
    del state
    model.requires_grad_(False)
    return model.eval().to(device)
