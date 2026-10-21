"""Transformer encoder pequeno da bIA; atenção e camadas implementadas no projeto.

PyTorch fornece tensores, autograd e kernels, mas NÃO pesos pré-treinados.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

LABELS = ("O", "B-MATERIAL", "I-MATERIAL", "B-PROCESS", "I-PROCESS", "B-PROPERTY", "I-PROPERTY", "B-QUANTITY", "I-QUANTITY", "B-UNIT", "I-UNIT", "B-CONDITION", "I-CONDITION", "B-APPLICATION", "I-APPLICATION")
ENTITY_LABELS = tuple(item[2:] for item in LABELS if item.startswith("B-"))


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    dim: int = 64
    heads: int = 4
    layers: int = 2
    max_length: int = 256

    def validate(self):
        if self.dim % self.heads or min(self.dim, self.heads, self.layers, self.max_length, self.vocab_size) <= 0:
            raise ValueError("Configuração de modelo inválida")


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, time, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, time, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().reshape(batch, time, dim)
        return self.proj(out) * mask.unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = Attention(config.dim, config.heads)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(nn.Linear(config.dim, config.dim * 4), nn.GELU(), nn.Linear(config.dim * 4, config.dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        return (x + self.mlp(self.norm2(x))) * mask.unsqueeze(-1)


class BiaTagger(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.positions = nn.Embedding(config.max_length, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.classifier = nn.Linear(config.dim, len(LABELS))

    def forward(self, ids: torch.Tensor):
        batch, seq = ids.shape
        if seq > self.config.max_length:
            raise ValueError("Sequência excede max_length")
        mask = ids != 0
        x = self.tokens(ids) + self.positions(torch.arange(seq, device=ids.device))[None]
        x = x * mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, mask)
        return self.classifier(self.norm(x))
