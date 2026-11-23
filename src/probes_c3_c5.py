"""
Batched Frozen-Pilot Probing Selectors:
  - C3 (SpilloverMerge): Rewards phrases whose dedicated input embedding strictly improves
    causal prediction of the *subsequent* raw-text continuation on held-out calibration split B.
  - C5 (ContrastMerge): 100% self-supervised counterfactual margin selector — rewards phrases
    whose merged input representation widens the log-likelihood margin between true continuations
    and controlled morphosyntactic / entity-swap / negation / quantity corruptions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.config import TokenizerSpec
from src.model import ResearchTransformerLM
from src.phrase_selectors import PhraseCandidate


@dataclass
class ProbeWindow:
    cand_idx: int
    prefix_ids: list[int]          # Base token IDs before the phrase
    phrase_ids: tuple[int, ...]    # Constituent base token IDs of the phrase
    cont_ids: list[int]            # Identical continuation base token IDs
    corrupt_cont_ids: list[int]    # Matched-length counterfactual continuation for C5


# Common deterministic morphosyntactic / polarity / quantity swap table (zero external data!)
_COUNTERFACTUAL_WORD_SWAPS: dict[str, str] = {
    " is": " are", " are": " is",
    " was": " were", " were": " was",
    " has": " have", " have": " has",
    " does": " do", " do": " does",
    " can": " cannot", " cannot": " can",
    " will": " will not", " must": " must not",
    " more": " less", " less": " more",
    " many": " few", " few": " many",
    " one": " two", " two": " three", " three": " two",
    " first": " second", " second": " first",
    " he": " they", " she": " they", " they": " he",
    " his": " their", " her": " their", " their": " his",
}


def _make_counterfactual_continuation(
    cont_ids: list[int],
    prefix_ids: list[int],
    swap_id_map: dict[int, int],
    fallback_pool: list[list[int]],
    rng: random.Random,
) -> list[int]:
    """
    Constructs a matched-length counterfactual continuation from the corpus itself:
      1. Prefer deterministic agreement / polarity / quantity token swap if present in continuation.
      2. Else swap a content token from earlier in the prefix into the continuation (role/entity swap).
      3. Else draw a hard-negative continuation of identical token length from fallback_pool.
    """
    out = list(cont_ids)
    swapped = False
    for idx, tid in enumerate(out):
        if tid in swap_id_map:
            out[idx] = swap_id_map[tid]
            swapped = True
            break

    if not swapped and len(prefix_ids) >= 6 and len(out) >= 4:
        # Swap a mid-continuation content-like token (higher ID = rarer content word in BPE)
        cand_pos = [i for i, t in enumerate(out) if t > 600]
        pref_tokens = [t for t in prefix_ids if t > 600 and t not in out]
        if cand_pos and pref_tokens:
            out[cand_pos[0]] = rng.choice(pref_tokens)
            swapped = True

    if not swapped and fallback_pool:
        alt = rng.choice(fallback_pool)
        L = len(out)
        if len(alt) >= L:
            out = list(alt[:L])
        else:
            out = list(alt) + out[len(alt) :]

    return out


def extract_probe_windows(
    docs_base_ids: Sequence[list[int]],
    candidates: Sequence[PhraseCandidate],
    base_tokenizer: Tokenizer,
    prefix_len: int = 48,
    cont_len: int = 24,
    max_windows_per_cand: int = 120,
    seed: int = 42,
) -> list[ProbeWindow]:
    rng = random.Random(seed)
    vocab = base_tokenizer.get_vocab()

    swap_id_map: dict[int, int] = {}
    for src_w, dst_w in _COUNTERFACTUAL_WORD_SWAPS.items():
        s_ids = base_tokenizer.encode(src_w, add_special_tokens=False).ids
        d_ids = base_tokenizer.encode(dst_w, add_special_tokens=False).ids
        if len(s_ids) == 1 and len(d_ids) == 1:
            swap_id_map[s_ids[0]] = d_ids[0]

    cand_to_idx = {c.base_ids: idx for idx, c in enumerate(candidates)}
    by_pair: dict[int, list[tuple[tuple[int, ...], int]]] = defaultdict(list)
    for idx, c in enumerate(candidates):
        by_pair[(int(c.base_ids[0]) << 16) | int(c.base_ids[1])].append((c.base_ids, idx))
    max_l = max((len(c.base_ids) for c in candidates), default=2)

    raw_hits: dict[int, list[tuple[list[int], list[int]]]] = defaultdict(list)
    fallback_conts: list[list[int]] = []

    for seq in docs_base_ids[:2_500]:  # ~3M tokens is plenty to saturate 60 windows/candidate in <2s!
        L = len(seq)
        if L < prefix_len + cont_len + 4:
            continue
        for i in range(16, L - cont_len - max_l):
            pair_hits = by_pair.get((seq[i] << 16) | seq[i + 1])
            if not pair_hits:
                continue
            for tup, c_idx in pair_hits:
                l = len(tup)
                if len(raw_hits[c_idx]) < max_windows_per_cand and tuple(seq[i : i + l]) == tup:
                    p_start = max(0, i - prefix_len)
                    pref = list(seq[p_start:i])
                    cont = list(seq[i + l : i + l + cont_len])
                    if len(pref) >= 8 and len(cont) == cont_len:
                        raw_hits[c_idx].append((pref, cont))
                        if len(fallback_conts) < 4096:
                            fallback_conts.append(cont)

    windows: list[ProbeWindow] = []
    for c_idx, hits in raw_hits.items():
        tup = candidates[c_idx].base_ids
        for pref, cont in hits:
            corrupt = _make_counterfactual_continuation(cont, pref, swap_id_map, fallback_conts, rng)
            windows.append(
                ProbeWindow(
                    cand_idx=c_idx,
                    prefix_ids=pref,
                    phrase_ids=tup,
                    cont_ids=cont,
                    corrupt_cont_ids=corrupt,
                )
            )
    rng.shuffle(windows)
    return windows


def fit_and_score_c3_c5_probes(
    pilot_model: ResearchTransformerLM,
    candidates: Sequence[PhraseCandidate],
    split_a_docs: Sequence[list[int]],
    split_b_docs: Sequence[list[int]],
    base_tokenizer: Tokenizer,
    spec: TokenizerSpec = TokenizerSpec(),
    fit_epochs: int = 6,
    l2_anchor_weight: float = 0.05,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[tuple[tuple[int, ...], float]], list[tuple[tuple[int, ...], float]]]:
    """
    Jointly fits temporary phrase embeddings E_cand[M, D] on Calibration Split A (with frozen pilot
    backbone and strictly fixed 14,336 output vocabulary), then evaluates on disjoint Split B:
      - C3 (SpilloverMerge): lower-confidence-bound reduction in continuation NLL (split - merged).
      - C5 (ContrastMerge): lower-confidence-bound gain in counterfactual log-prob margin (merged - split).
    """
    n_base = spec.base_vocab_size
    d_model = pilot_model.spec.d_model
    M = len(candidates)

    pilot_model.eval()
    for p in pilot_model.parameters():
        p.requires_grad_(False)

    # Warm-start temporary candidate input embeddings as mean of constituent input embeddings
    with torch.no_grad():
        base_emb_w = pilot_model.tok_emb.weight[:n_base].detach().float()
        init_mat = torch.zeros((M, d_model), dtype=torch.float32, device=device)
        for idx, c in enumerate(candidates):
            ids_t = torch.tensor(c.base_ids, dtype=torch.long, device=device)
            init_mat[idx] = base_emb_w[ids_t].mean(dim=0)

    cand_emb = nn.Embedding(M, d_model, device=device)
    cand_emb.weight.data.copy_(init_mat)
    anchor_mat = init_mat.clone()

    train_wins = extract_probe_windows(split_a_docs, candidates, base_tokenizer, seed=101)
    eval_wins = extract_probe_windows(split_b_docs, candidates, base_tokenizer, seed=202)

    optimizer = torch.optim.AdamW([cand_emb.weight], lr=3e-3, weight_decay=0.0)
    pad_id = base_tokenizer.token_to_id(spec.special_tokens[1])

    # --- Phase 1: Fit E_cand on Calibration Split A ---
    for _epoch in range(fit_epochs):
        random.shuffle(train_wins)
        for b0 in range(0, len(train_wins), batch_size):
            batch = train_wins[b0 : b0 + batch_size]
            B = len(batch)
            # Merged sequence: [left_pad..., prefix, <PHRASE_SLOT>, cont]
            max_pref = max(len(w.prefix_ids) for w in batch)
            cont_len = len(batch[0].cont_ids)
            T = max_pref + 1 + cont_len

            x_np = np.full((B, T), pad_id, dtype=np.int64)
            y_np = np.full((B, T), -100, dtype=np.int64)
            p_pos = max_pref
            c_idx_np = np.empty(B, dtype=np.int64)

            for bi, w in enumerate(batch):
                p_len = len(w.prefix_ids)
                off = max_pref - p_len
                x_np[bi, off : off + p_len] = w.prefix_ids
                c_idx_np[bi] = w.cand_idx
                x_np[bi, p_pos + 1 : p_pos + 1 + cont_len] = w.cont_ids
                y_np[bi, p_pos : p_pos + cont_len] = w.cont_ids

            x_ids = torch.from_numpy(x_np).to(device, non_blocking=True)
            y_targets = torch.from_numpy(y_np).to(device, non_blocking=True)
            c_indices = torch.from_numpy(c_idx_np).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                h_base = pilot_model.tok_emb(x_ids)
            h = h_base.clone()
            e_batch = cand_emb(c_indices)
            h[:, p_pos] = e_batch.to(h.dtype)

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                logits = pilot_model.forward_from_embeddings(h, active_output_vocab=n_base)
                ce_loss = F.cross_entropy(logits.view(-1, n_base), y_targets.view(-1), ignore_index=-100)
                unique_c = c_indices.unique()
                anchor_loss = l2_anchor_weight * (cand_emb.weight[unique_c] - anchor_mat[unique_c]).pow(2).mean()
                loss = ce_loss + anchor_loss
            loss.backward()
            optimizer.step()

    # --- Phase 2: Paired Split-vs-Merged Evaluation on Disjoint Calibration Split B ---
    c3_diffs: dict[int, list[float]] = defaultdict(list)
    c5_diffs: dict[int, list[float]] = defaultdict(list)

    @torch.no_grad()
    def score_continuations(batch_w: list[ProbeWindow], use_merged: bool, use_corrupt: bool) -> np.ndarray:
        B = len(batch_w)
        cont_len = len(batch_w[0].cont_ids)
        span_lens = [1 if use_merged else len(w.phrase_ids) for w in batch_w]
        max_pref_plus_span = max(len(w.prefix_ids) + sl for w, sl in zip(batch_w, span_lens))
        T = max_pref_plus_span + cont_len

        x_np = np.full((B, T), pad_id, dtype=np.int64)
        y_np = np.full((B, T), -100, dtype=np.int64)
        span_last = max_pref_plus_span - 1

        for bi, (w, sl) in enumerate(zip(batch_w, span_lens)):
            p_len = len(w.prefix_ids)
            off = max_pref_plus_span - (p_len + sl)
            x_np[bi, off : off + p_len] = w.prefix_ids
            if not use_merged:
                x_np[bi, off + p_len : off + p_len + sl] = w.phrase_ids
            c_seq = w.corrupt_cont_ids if use_corrupt else w.cont_ids
            x_np[bi, span_last + 1 : span_last + 1 + cont_len] = c_seq
            y_np[bi, span_last : span_last + cont_len] = c_seq

        x_ids = torch.from_numpy(x_np).to(device, non_blocking=True)
        y_targets = torch.from_numpy(y_np).to(device, non_blocking=True)

        h = pilot_model.tok_emb(x_ids).clone()
        if use_merged:
            c_idx_t = torch.tensor([w.cand_idx for w in batch_w], dtype=torch.long, device=device)
            h[:, span_last] = cand_emb(c_idx_t).to(h.dtype)

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits = pilot_model.forward_from_embeddings(h, active_output_vocab=n_base)
            per_tok = F.cross_entropy(
                logits.view(-1, n_base), y_targets.view(-1), ignore_index=-100, reduction="none"
            ).view(B, T)
        mask = (y_targets != -100).float()
        # Sum NLL over identical continuation tokens
        seq_nll = (per_tok.float() * mask).sum(dim=1).cpu().numpy()
        return seq_nll

    for b0 in range(0, len(eval_wins), batch_size):
        batch = eval_wins[b0 : b0 + batch_size]
        nll_split_true = score_continuations(batch, use_merged=False, use_corrupt=False)
        nll_merged_true = score_continuations(batch, use_merged=True, use_corrupt=False)
        nll_split_bad = score_continuations(batch, use_merged=False, use_corrupt=True)
        nll_merged_bad = score_continuations(batch, use_merged=True, use_corrupt=True)

        # C3: L_cont(split) - L_cont(merged) (>0 means merged predicts continuation better)
        spill_gain = nll_split_true - nll_merged_true
        # C5: Margin M = log P(true) - log P(bad) = NLL(bad) - NLL(true)
        margin_split = nll_split_bad - nll_split_true
        margin_merged = nll_merged_bad - nll_merged_true
        contrast_gain = margin_merged - margin_split

        for bi, w in enumerate(batch):
            c3_diffs[w.cand_idx].append(float(spill_gain[bi]))
            c5_diffs[w.cand_idx].append(float(contrast_gain[bi]))

    def rank_by_lcb(diff_map: dict[int, list[float]]) -> list[tuple[tuple[int, ...], float]]:
        ranked: list[tuple[float, tuple[int, ...]]] = []
        for idx, c in enumerate(candidates):
            vals = diff_map.get(idx, [])
            if len(vals) < 5:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            # Weight per-occurrence gain by log frequency & conservative LCB
            lcb = mean - 1.0 * se
            score = lcb * math.log1p(c.total_freq)
            ranked.append((score, c.base_ids))
        ranked.sort(key=lambda x: x[0], reverse=True)
        # Backfill if fewer than K candidates had >=5 eval windows
        chosen = [(tup, max(sc, 1e-6)) for sc, tup in ranked[: spec.phrase_vocab_slots]]
        chosen_set = {t for t, _ in chosen}
        for c in candidates:
            if len(chosen) >= spec.phrase_vocab_slots:
                break
            if c.base_ids not in chosen_set:
                chosen.append((c.base_ids, 1e-6))
                chosen_set.add(c.base_ids)
        return chosen

    for p in pilot_model.parameters():
        p.requires_grad_(True)

    return rank_by_lcb(c3_diffs), rank_by_lcb(c5_diffs)
