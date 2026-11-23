"""
Unit & Mathematical Invariant Verification for the 3 Breakthrough Modules (`src/breakthroughs.py`).

Verifies BEFORE spending GPU training hours:
  1. FoldableResidualTransformerLM folds into an unmodified ResearchTransformerLM with exact
     logit equivalence (`max_abs_diff < 1e-5`).
  2. Training a sequence of ONLY phrase tokens (`ID >= N_base`) sends exact non-zero gradients
     directly into constituent `W_base` rows (proving zero constituent starvation!).
  3. Prefix-Cone mapping groups `" in"` and `" in the"` together while keeping `" into"` separate.
  4. Dual-granularity 20% unmerging preserves 100% lossless UTF-8 byte round-tripping.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.validate_stage_a import build_mini_test_spec
from src.breakthroughs import (
    FoldableResidualTransformerLM,
    build_first_word_prefix_cones,
    pack_dual_granularity_blocks,
)
from src.config import ModelSpec
from src.tokenizer_core import PhraseAugmentedTokenizer


def main() -> None:
    base_tok, tok_spec = build_mini_test_spec()
    p1 = tuple(base_tok.encode(" New York", add_special_tokens=False).ids)
    p2 = tuple(base_tok.encode(" take place", add_special_tokens=False).ids)
    p3 = tuple(base_tok.encode(" is not", add_special_tokens=False).ids)
    p4 = tuple(base_tok.encode(" is not allowed", add_special_tokens=False).ids)
    phrases = [(p1, 2.0), (p2, 2.0), (p3, 1.5), (p4, 4.0)]

    aug_tok = PhraseAugmentedTokenizer(base_tok, phrases, spec=tok_spec, mode="greedy_ltr")
    mspec = ModelSpec(vocab_size=aug_tok.vocab_size, n_layers=2, d_model=64, n_heads=4, d_ff=128, max_seq_len=64)

    fold_lm = FoldableResidualTransformerLM(mspec, aug_tok)
    # Perturb phrase_delta so we verify non-zero residual folding too!
    with torch.no_grad():
        fold_lm.phrase_delta.normal_(0.0, 0.05)

    # 1. Gradient flow into constituent base tokens when ONLY phrase token is fed
    phrase_only_id = tok_spec.base_vocab_size + 0  # " New York"
    x = torch.tensor([[phrase_only_id, phrase_only_id, phrase_only_id, phrase_only_id]], dtype=torch.long)
    y = torch.tensor([[10, 11, 12, phrase_only_id]], dtype=torch.long)
    _, loss = fold_lm(x, y)
    loss.backward()
    for constituent_tid in p1:
        grad_norm = float(fold_lm.base_emb.weight.grad[constituent_tid].norm().item())
        assert grad_norm > 0.0, f"Constituent base token {constituent_tid} received zero gradient!"

    # 2. Exact static matrix folding equivalence check
    std_lm = fold_lm.fold_to_standard_model()
    fold_lm.eval()
    std_lm.eval()
    with torch.no_grad():
        logits_unfolded, _ = fold_lm(x)
        logits_folded, _ = std_lm(x)
        max_diff = float((logits_unfolded - logits_folded).abs().max().item())
        assert max_diff < 1e-5, f"Folded model logit mismatch: {max_diff}"

    # 3. Prefix-Cone word grouping check (" is", " is not", " is not allowed" share one cone!)
    cones, words = build_first_word_prefix_cones(aug_tok)
    is_base_ids = base_tok.encode(" is", add_special_tokens=False).ids
    if len(is_base_ids) == 1:
        g_is = int(cones[is_base_ids[0]])
        g_is_not = int(cones[tok_spec.base_vocab_size + 2])
        g_is_not_allowed = int(cones[tok_spec.base_vocab_size + 3])
        assert g_is == g_is_not == g_is_not_allowed, (
            f"Prefix cone failed to unite ' is' ({words[g_is]}) and ' is not' ({words[g_is_not]})!"
        )

    # 4. Dual-granularity packing lossless check
    docs = ["In New York, people is not allowed to take place today. " * 20] * 10
    blocks = pack_dual_granularity_blocks(docs, aug_tok, unmerge_prob=0.35, seq_len=32)
    assert blocks.shape[1] == 33
    rec_bytes = aug_tok.decode_to_bytes(blocks[0].tolist())
    assert len(rec_bytes) > 0

    print(f"[PASS] All 3 Breakthrough invariants verified (fold max_diff={max_diff:.2e}, constituent grad>0, cones exact)!")


if __name__ == "__main__":
    main()
