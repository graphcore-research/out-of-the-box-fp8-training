"""A non-unit-scaled model that behaves like NanoGPT"""

from argparse import Namespace
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = torch.split(self.c_attn(x), self.cfg.n_embd, dim=2)
        q = q.view(B, T, self.cfg.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_head, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.cfg.dropout, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + F.dropout(
            self.attn(F.layer_norm(x, (self.cfg.n_embd,))), self.cfg.dropout
        )
        return x + F.dropout(
            self.mlp(F.layer_norm(x, (self.cfg.n_embd,))), self.cfg.dropout
        )


class Transformer(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.Sequential(*[TransformerLayer(cfg) for _ in range(cfg.n_layer)])

    def forward(self, idx: Tensor) -> Tensor:
        h = self.wte(idx) + self.wpe(torch.arange(0, idx.shape[1]))
        h = F.dropout(h, self.cfg.dropout)
        h = self.h(h)
        return F.layer_norm(h, (self.cfg.n_embd,))


class Model(nn.Module):
    def __init__(self, **args: Any):
        super().__init__()
        self.config = cfg = Namespace(**args)
        self.transformer = Transformer(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        for p in self.parameters():
            if p.ndim == 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.lm_head(self.transformer(idx))
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss
