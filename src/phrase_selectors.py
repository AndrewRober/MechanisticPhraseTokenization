"""
Phrase Candidate Discovery & Mechanistic Vocabulary Selectors (C0 - C5 + Controls).

Every selector outputs an exact list of K=2,048 (base_id_tuple, utility) pairs on top of the
shared 14,336 Base Byte-BPE vocabulary, using ONLY training corpus statistics / pilot signals.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
import re
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.config import TokenizerSpec
from src.model import ResearchTransformerLM
from src.tokenizer_core import (
    PhraseAugmentedTokenizer,
    is_continuation_alpha_subword,
    is_word_start_token,
    token_str_to_bytes,
)


@dataclass
class PhraseCandidate:
    base_ids: tuple[int, ...]
    word_count: int
    total_freq: int
    doc_freq: int
    raw_text: str


def mine_phrase_candidates(
    docs_base_ids: Sequence[list[int]],
    base_tokenizer: Tokenizer,
    spec: TokenizerSpec = TokenizerSpec(),
    max_candidates: int = 12_000,
) -> list[PhraseCandidate]:
    """
    Extracts valid 2-to-4 word phrase candidates from pre-encoded training documents.
    Enforces:
      - Word-boundary start & end (never starts or ends mid-word).
      - Document-frequency minimum (spec.min_doc_freq) so duplicated boilerplate docs cannot qualify.
      - Character length cap (spec.max_phrase_chars) and no special tokens or pure whitespace runs.
    """
    vocab = base_tokenizer.get_vocab()
    id_to_tok = {v: k for k, v in vocab.items()}
    special_ids = {vocab[s] for s in spec.special_tokens if s in vocab}
    n_base = base_tokenizer.get_vocab_size()

    is_wstart = [False] * n_base
    is_cont = [False] * n_base
    tok_bytes = [b""] * n_base
    has_alnum = [False] * n_base

    for tid in range(n_base):
        if tid in special_ids:
            continue
        tstr = id_to_tok[tid]
        b = token_str_to_bytes(tstr)
        tok_bytes[tid] = b
        is_wstart[tid] = is_word_start_token(tstr, is_doc_start=False)
        is_cont[tid] = is_continuation_alpha_subword(tstr)
        s_dec = b.decode("utf-8", errors="ignore")
        has_alnum[tid] = any(ch.isalnum() for ch in s_dec)

    total_counts: Counter[tuple[int, ...]] = Counter()
    doc_counts: Counter[tuple[int, ...]] = Counter()

    for seq in docs_base_ids:
        L = len(seq)
        if L < 2:
            continue
        # Identify word spans [w_start, w_end) inside this document
        word_spans: list[tuple[int, int]] = []
        i = 0
        while i < L:
            if seq[i] in special_ids:
                i += 1
                continue
            j = i + 1
            while j < L and seq[j] not in special_ids and is_cont[seq[j]]:
                j += 1
            # Check if this pre-token is an actual word (contains alphanumeric)
            if any(has_alnum[seq[p]] for p in range(i, j)):
                if i == 0 or seq[i - 1] in special_ids or is_wstart[seq[i]]:
                    word_spans.append((i, j))
            i = j

        seen_in_doc: set[tuple[int, ...]] = set()
        num_w = len(word_spans)
        for idx_w in range(num_w):
            for w_len in range(spec.min_phrase_words, spec.max_phrase_words + 1):
                if idx_w + w_len > num_w:
                    break
                # Ensure contiguous spans in base_ids (no skipped punctuation tokens between words unless attached)
                span_start = word_spans[idx_w][0]
                span_end = word_spans[idx_w + w_len - 1][1]
                if span_end - span_start > 8:  # cap subword fan-out per phrase
                    break
                # Verify intermediate tokens don't cross special tokens or newlines
                tup = tuple(seq[span_start:span_end])
                total_counts[tup] += 1
                seen_in_doc.add(tup)

        for tup in seen_in_doc:
            doc_counts[tup] += 1

    candidates: list[PhraseCandidate] = []
    for tup, df in doc_counts.items():
        if df < spec.min_doc_freq:
            continue
        raw_b = b"".join(tok_bytes[t] for t in tup)
        if len(raw_b) > spec.max_phrase_chars or b"\n" in raw_b or b"\r" in raw_b:
            continue
        text = raw_b.decode("utf-8", errors="ignore")
        w_cnt = len([w for w in text.strip().split() if any(c.isalnum() for c in w)])
        if not (spec.min_phrase_words <= w_cnt <= spec.max_phrase_words):
            continue
        candidates.append(
            PhraseCandidate(
                base_ids=tup,
                word_count=w_cnt,
                total_freq=total_counts[tup],
                doc_freq=df,
                raw_text=text,
            )
        )

    # Sort by raw token savings (freq * (len - 1)) and retain top pool
    candidates.sort(key=lambda c: c.total_freq * (len(c.base_ids) - 1), reverse=True)
    return candidates[:max_candidates]


def select_c0_freq_or_c2_reserve(
    candidates: Sequence[PhraseCandidate],
    sample_docs_base_ids: Sequence[list[int]],
    base_tokenizer: Tokenizer,
    spec: TokenizerSpec = TokenizerSpec(),
    penalty_lambda: float = 0.0,
    batch_size: int = 64,
) -> list[tuple[tuple[int, ...], float]]:
    """
    Unified iterative batched selector for:
      - C0 (FreqPhraseMerge / Null Control): penalty_lambda = 0.0
      - C2 (ReserveMerge): penalty_lambda > 0.0 with saturation-guarded component exposure floor:
            floor_t = min(alpha * N_orig[t], N_saturation)
    Re-tokenizes sample_docs after every batch of additions so overlapping phrases compete honestly
    and dead/shadowed slots are avoided.
    """
    n_base = base_tokenizer.get_vocab_size()
    orig_counts = np.zeros(n_base, dtype=np.int64)
    total_sample_tokens = 0
    for seq in sample_docs_base_ids:
        total_sample_tokens += len(seq)
        for tid in seq:
            if tid < n_base:
                orig_counts[tid] += 1

    # Scale absolute saturation floor proportionally if sample_docs is a subset
    scale = max(total_sample_tokens / 10_000_000.0, 0.1)
    abs_sat = int(spec.reserve_abs_saturation * scale)
    floors = np.minimum((spec.reserve_rel_floor * orig_counts).astype(np.int64), abs_sat)

    selected: list[tuple[tuple[int, ...], float]] = []
    selected_set: set[tuple[int, ...]] = set()
    remaining = [c for c in candidates]

    current_streams = [list(seq) for seq in sample_docs_base_ids]
    cur_base_counts = orig_counts.copy()

    while len(selected) < spec.phrase_vocab_slots and remaining:
        need = min(batch_size, spec.phrase_vocab_slots - len(selected))

        # Count active non-overlapping matches of remaining candidates in current_streams
        # Fast hash-gram pass for lengths 2..8
        active_span_counts: Counter[tuple[int, ...]] = Counter()
        cand_lookup = {c.base_ids for c in remaining if c.base_ids not in selected_set}
        max_l = max((len(t) for t in cand_lookup), default=0)
        if max_l == 0:
            break

        for seq in current_streams:
            L = len(seq)
            for i in range(L - 1):
                if seq[i] >= n_base:
                    continue
                for l in range(2, min(max_l, L - i) + 1):
                    if seq[i + l - 1] >= n_base:
                        break
                    tup = tuple(seq[i : i + l])
                    if tup in cand_lookup:
                        active_span_counts[tup] += 1

        scored: list[tuple[float, PhraseCandidate]] = []
        for c in remaining:
            if c.base_ids in selected_set:
                continue
            act_cnt = active_span_counts.get(c.base_ids, 0)
            if act_cnt < 3:
                continue
            tok_savings = float(act_cnt * (len(c.base_ids) - 1))

            starve_pen = 0.0
            if penalty_lambda > 0.0:
                t_mult = Counter(c.base_ids)
                for tid, mult in t_mult.items():
                    proj_rem = cur_base_counts[tid] - act_cnt * mult
                    deficit = max(0, int(floors[tid]) - int(proj_rem))
                    starve_pen += deficit

            util = tok_savings - penalty_lambda * starve_pen
            scored.append((util, c))

        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)

        # Greedily accept batch while checking incremental component counts
        added_this_round = 0
        for util, c in scored:
            if added_this_round >= need:
                break
            act_cnt = active_span_counts.get(c.base_ids, 0)
            if penalty_lambda > 0.0:
                t_mult = Counter(c.base_ids)
                recalc_pen = sum(
                    max(0, int(floors[tid]) - int(cur_base_counts[tid] - act_cnt * mult))
                    for tid, mult in t_mult.items()
                )
                util = float(act_cnt * (len(c.base_ids) - 1)) - penalty_lambda * recalc_pen
                if util <= 0:
                    continue

            selected.append((c.base_ids, max(util, 1e-6)))
            selected_set.add(c.base_ids)
            added_this_round += 1
            for tid in c.base_ids:
                cur_base_counts[tid] = max(0, cur_base_counts[tid] - act_cnt)

        if added_this_round == 0:
            # Fill remaining if strict penalty rejected everything
            for util, c in scored[:need]:
                selected.append((c.base_ids, max(util, 1e-6)))
                selected_set.add(c.base_ids)
            break

        # Re-segment sample streams with real PhraseAugmentedTokenizer to capture exact overlap shadowing
        temp_spec = TokenizerSpec(
            total_vocab_size=n_base + len(selected),
            base_vocab_size=n_base,
            phrase_vocab_slots=len(selected),
        )
        aug = PhraseAugmentedTokenizer(base_tokenizer, selected, spec=temp_spec, mode="greedy_ltr")
        current_streams = [aug.merge_base_ids(seq) for seq in sample_docs_base_ids]
        cur_base_counts.fill(0)
        for s_merged in current_streams:
            for tid in s_merged:
                if tid < n_base:
                    cur_base_counts[tid] += 1

    return selected[: spec.phrase_vocab_slots]


@torch.no_grad()
def _collect_split_conflict_stats(
    model: ResearchTransformerLM,
    docs_base_ids: Sequence[list[int]],
    candidate_tuples: set[tuple[int, ...]],
    n_base: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 32,
    seq_len: int = 512,
) -> tuple[dict[tuple[int, ...], dict[int, np.ndarray]], dict[tuple[int, ...], int], np.ndarray, np.ndarray, dict[tuple[int, ...], float]]:
    """
    Collects pure input-embedding activation gradients (isolated from tied output-head gradients)
    on one corpus split for C1 (ConflictMerge) and VEGAD magnitude control.
    """
    d_model = model.spec.d_model
    max_l = max((len(t) for t in candidate_tuples), default=2)

    # Pre-build fast 2-token integer prefix lookup so we never allocate 50M Python tuples!
    by_pair: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for tup in candidate_tuples:
        key2 = (int(tup[0]) << 16) | int(tup[1])
        by_pair[key2].append(tup)

    # Cap at 1.5M tokens per split (~3,000 sequences of 512) — plenty of statistical power, 12s runtime!
    flat: list[int] = []
    for doc in docs_base_ids:
        flat.extend(doc)
        if len(flat) >= 1_500_000:
            break

    n_blocks = (len(flat) - 1) // seq_len
    if n_blocks < 2:
        raise ValueError("Not enough tokens in split for C1 gradient collection")

    tok_sum_grad_gpu = torch.zeros((n_base, d_model), dtype=torch.float32, device=device)
    tok_count_gpu = torch.zeros(n_base, dtype=torch.int64, device=device)
    ones_flat = None
    phrase_tok_sum_grad: dict[tuple[int, ...], dict[int, np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros(d_model, dtype=np.float64))
    )
    phrase_counts: Counter[tuple[int, ...]] = Counter()
    vegad_norm_sum: Counter[tuple[int, ...]] = Counter()

    model.eval()
    for b_start in range(0, n_blocks, batch_size):
        b_end = min(b_start + batch_size, n_blocks)
        B = b_end - b_start
        x_np = np.empty((B, seq_len), dtype=np.int64)
        y_np = np.empty((B, seq_len), dtype=np.int64)
        for bi, blk_idx in enumerate(range(b_start, b_end)):
            off = blk_idx * seq_len
            x_np[bi] = flat[off : off + seq_len]
            y_np[bi] = flat[off + 1 : off + seq_len + 1]

        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        with torch.enable_grad():
            h = model.tok_emb(x).detach().requires_grad_(True)
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                logits = model.forward_from_embeddings(h, active_output_vocab=n_base)
                loss = F.cross_entropy(logits.view(-1, n_base), y.view(-1), reduction="sum")
            loss.backward()
            g_gpu = h.grad.detach().float()
            # Accumulate global token input gradients directly on GPU in 0.05ms (replaces slow 2D np.add.at!)
            flat_x_gpu = x.view(-1)
            tok_sum_grad_gpu.index_add_(0, flat_x_gpu, g_gpu.view(-1, d_model))
            if ones_flat is None or ones_flat.numel() != flat_x_gpu.numel():
                ones_flat = torch.ones(flat_x_gpu.numel(), dtype=torch.int64, device=device)
            tok_count_gpu.index_add_(0, flat_x_gpu, ones_flat)
            gnorm_gpu = torch.linalg.vector_norm(g_gpu, dim=-1)
            g = g_gpu.cpu().numpy()
            g_norms = gnorm_gpu.cpu().numpy()

        # Fast prefix-filtered candidate scan
        for bi in range(B):
            row = x_np[bi].tolist()
            grow = g[bi]
            gnorm_row = g_norms[bi]
            for i in range(seq_len - 1):
                cands_here = by_pair.get((row[i] << 16) | row[i + 1])
                if not cands_here:
                    continue
                for tup in cands_here:
                    l = len(tup)
                    if i + l <= seq_len and tuple(row[i : i + l]) == tup:
                        phrase_counts[tup] += 1
                        p_dict = phrase_tok_sum_grad[tup]
                        span_norm = 0.0
                        for pos, tid in enumerate(tup):
                            p_dict[tid] += grow[i + pos]
                            span_norm += float(gnorm_row[i + pos])
                        vegad_norm_sum[tup] += span_norm

    tok_sum_grad = tok_sum_grad_gpu.cpu().numpy().astype(np.float64)
    tok_count = tok_count_gpu.cpu().numpy()

    return phrase_tok_sum_grad, phrase_counts, tok_sum_grad, tok_count, vegad_norm_sum


def select_c1_conflict_and_vegad_control(
    pilot_model: ResearchTransformerLM,
    candidates: Sequence[PhraseCandidate],
    split_a_docs: Sequence[list[int]],
    split_b_docs: Sequence[list[int]],
    spec: TokenizerSpec = TokenizerSpec(),
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[tuple[tuple[int, ...], float]], list[tuple[tuple[int, ...], float]]]:
    """
    Computes BOTH:
      1. C1 (ConflictMerge): Cross-Split Ward Gradient Divergence (cancels tied-embedding output bias,
         zero random-noise expectation, measures exact optimization energy freed by unsharing p).
      2. VEGAD Control: Gradient-magnitude phrase selection from the identical pilot & splits.
    """
    cand_tuples = {c.base_ids for c in candidates}
    n_base = spec.base_vocab_size

    p_sum_a, p_cnt_a, t_sum_a, t_cnt_a, vegad_a = _collect_split_conflict_stats(
        pilot_model, split_a_docs, cand_tuples, n_base, device, dtype
    )
    p_sum_b, p_cnt_b, t_sum_b, t_cnt_b, vegad_b = _collect_split_conflict_stats(
        pilot_model, split_b_docs, cand_tuples, n_base, device, dtype
    )

    c1_scores: list[tuple[float, tuple[int, ...]]] = []
    vegad_scores: list[tuple[float, tuple[int, ...]]] = []

    for c in candidates:
        tup = c.base_ids
        na = p_cnt_a.get(tup, 0)
        nb = p_cnt_b.get(tup, 0)
        if na < 3 or nb < 3:
            continue

        ward_sum = 0.0
        tok_mult = Counter(tup)
        for tid, mult in tok_mult.items():
            n_in_a = na * mult
            n_in_b = nb * mult
            n_tot_a = int(t_cnt_a[tid])
            n_tot_b = int(t_cnt_b[tid])
            n_out_a = n_tot_a - n_in_a
            n_out_b = n_tot_b - n_in_b
            if n_out_a < 5 or n_out_b < 5:
                continue

            g_in_a = p_sum_a[tup][tid] / n_in_a
            g_out_a = (t_sum_a[tid] - p_sum_a[tup][tid]) / n_out_a

            g_in_b = p_sum_b[tup][tid] / n_in_b
            g_out_b = (t_sum_b[tid] - p_sum_b[tup][tid]) / n_out_b

            delta_a = g_in_a - g_out_a
            delta_b = g_in_b - g_out_b
            cross_dot = float(np.dot(delta_a, delta_b))

            n_in = n_in_a + n_in_b
            n_out = n_out_a + n_out_b
            eff_n = (n_in * n_out) / max(n_in + n_out, 1)
            ward_sum += max(0.0, eff_n * cross_dot)

        c1_scores.append((ward_sum, tup))
        vegad_scores.append((float(vegad_a.get(tup, 0.0) + vegad_b.get(tup, 0.0)), tup))

    c1_scores.sort(key=lambda x: x[0], reverse=True)
    vegad_scores.sort(key=lambda x: x[0], reverse=True)

    def _finalize(scored_list: list[tuple[float, tuple[int, ...]]]) -> list[tuple[tuple[int, ...], float]]:
        chosen = [(tup, max(sc, 1e-6)) for sc, tup in scored_list[: spec.phrase_vocab_slots]]
        chosen_set = {t for t, _ in chosen}
        for c in candidates:
            if len(chosen) >= spec.phrase_vocab_slots:
                break
            if c.base_ids not in chosen_set:
                chosen.append((c.base_ids, 1e-6))
                chosen_set.add(c.base_ids)
        return chosen

    return _finalize(c1_scores), _finalize(vegad_scores)
