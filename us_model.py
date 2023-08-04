"""A unit-scaled model that behaves like NanoGPT"""

from argparse import Namespace
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import unit_scaling as uu
import unit_scaling.functional as U


class Attention(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.cfg = cfg
        self.c_attn = uu.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = uu.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = torch.split(self.c_attn(x), self.cfg.n_embd, dim=2)
        q = q.view(B, T, self.cfg.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_head, -1).transpose(1, 2)
        y = U.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.cfg.dropout, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.c_fc = uu.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = uu.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(U.gelu(self.c_fc(x)))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        for block, tau in [(self.attn, 0.005), (self.mlp, 0.5)]:
            b, x = U.residual_split(x, tau=tau)
            b = U.dropout(block(U.layer_norm(b, (self.cfg.n_embd,))), self.cfg.dropout)
            x = U.residual_add(b, x, tau=tau)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.cfg = cfg
        self.wte = uu.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = uu.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.Sequential(*[TransformerLayer(cfg) for _ in range(cfg.n_layer)])

    def forward(self, idx: Tensor) -> Tensor:
        h = U.add(self.wte(idx), self.wpe(torch.arange(0, idx.shape[1])))
        h = U.dropout(h, self.cfg.dropout)
        h = self.h(h)
        return U.layer_norm(h, (self.cfg.n_embd,))


class Model(nn.Module):
    def __init__(self, **args: Any):
        super().__init__()
        self.config = cfg = Namespace(**args)
        self.transformer = Transformer(cfg)
        self.lm_head = uu.Linear(
            cfg.n_embd, cfg.vocab_size, bias=False, constraint=None
        )

    def forward(self, idx: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.lm_head(self.transformer(idx))
        loss = U.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss
