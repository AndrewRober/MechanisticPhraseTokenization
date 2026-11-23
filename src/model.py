import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelSpec


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps).to(x.dtype)
        return x_normed * self.weight


def precompute_rope_cache(
    seq_len: int, head_dim: int, theta: float = 10_000.0, device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    T = x.size(2)
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    out_even = x_even * cos_t - x_odd * sin_t
    out_odd = x_even * sin_t + x_odd * cos_t
    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.n_heads = spec.n_heads
        self.head_dim = spec.head_dim
        self.qkv = nn.Linear(spec.d_model, 3 * spec.d_model, bias=False)
        self.out_proj = nn.Linear(spec.d_model, spec.d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class FeedForward(nn.Module):
    """Bias-free 2-matrix GELU MLP matching the exact 30.42M parameter specification."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.up_proj = nn.Linear(spec.d_model, spec.d_ff, bias=False)
        self.down_proj = nn.Linear(spec.d_ff, spec.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


class DecoderBlock(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.attn_norm = RMSNorm(spec.d_model, eps=spec.rms_norm_eps)
        self.attn = CausalSelfAttention(spec)
        self.ffn_norm = RMSNorm(spec.d_model, eps=spec.rms_norm_eps)
        self.ffn = FeedForward(spec)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ResearchTransformerLM(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.tok_emb = nn.Embedding(spec.vocab_size, spec.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(spec) for _ in range(spec.n_layers)])
        self.final_norm = RMSNorm(spec.d_model, eps=spec.rms_norm_eps)

        if spec.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)

        cos, sin = precompute_rope_cache(spec.max_seq_len, spec.head_dim, spec.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        residual_std = std / math.sqrt(2 * self.spec.n_layers)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, mean=0.0, std=std)
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.up_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

    def forward_from_embeddings(self, h: torch.Tensor, active_output_vocab: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass starting from input embeddings h [B, T, D].
        Supports:
          - C1 (ConflictMerge): attach hook directly to h to read pure input-activation gradients
            without output-head gradient contamination from tied weights.
          - C3/C5 (Spillover/Contrast probes): splice candidate phrase embeddings into h while
            keeping active_output_vocab=14_336 so softmax normalization is strictly identical.
        """
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin)
        h = self.final_norm(h)
        if self.lm_head is not None:
            weight = self.lm_head.weight
        else:
            weight = self.tok_emb.weight
        if active_output_vocab is not None:
            weight = weight[:active_output_vocab]
        return F.linear(h, weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        active_output_vocab: Optional[int] = None,
        loss_reduction: str = "mean",
        return_logits: bool = True,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = self.tok_emb(input_ids)
        logits = self.forward_from_embeddings(h, active_output_vocab=active_output_vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction=loss_reduction,
            )
        return (logits if return_logits else None), loss

    def parameter_breakdown(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        return {
            "total": total,
            "tied_embedding": emb,
            "non_embedding_backbone": total - emb,
        }
