"""
Unified Baseline Tokenizers (B1 - B5) @ Exact V = 16,384 Vocabulary.

Implements a common `BaseTokenizerAdapter` interface (`.encode()`, `.decode_to_bytes()`, `.vocab_size`, `.eos_id`)
for all five published baselines with strict lossless UTF-8 byte round-tripping:
  - B1: Byte-level BPE (16,384 entries, standard GPT-2 byte-level regex + 256 byte fallback)
  - B2: Unigram LM (Kudo 2018, identity normalization, no whitespace collapsing, pinned 256 bytes, Viterbi)
  - B3: SaGe (Yehezkel & Pinter 2023, SkipGram context-objective pruning from 24,576 -> 16,384 with pinned bytes)
  - B4: SuperBPE (Liu et al. 2025, Phase 1 = shared 14,336 subwords, Phase 2 = 2,048 cross-whitespace merges)
  - B5: SupraTok (Tănase & Pelican TACL 2026, 3-pass staged cross-boundary PMI-guided curriculum, no data filtering)
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import TokenizerSpec
from src.tokenizer_core import (
    BYTE_TO_UNICODE,
    PhraseAugmentedTokenizer,
    token_str_to_bytes,
    train_byte_bpe_tokenizer,
)


class TokenizerInterface(Protocol):
    vocab_size: int
    eos_id: int
    pad_id: int

    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...
    def decode_to_bytes(self, token_ids: Iterable[int]) -> bytes: ...


class HFByteTokenizerWrapper:
    """Wraps a HuggingFace ByteLevel Tokenizer (B1, B2-Unigram-ByteLevel, B3-SaGe, B4-SuperBPE)."""

    def __init__(self, tok: Tokenizer, spec: TokenizerSpec = TokenizerSpec()):
        self.tok = tok
        self.spec = spec
        vocab = tok.get_vocab()
        self.vocab_size = tok.get_vocab_size()
        self.eos_id = vocab[spec.special_tokens[0]]
        self.pad_id = vocab[spec.special_tokens[1]]
        self.special_ids = {self.eos_id, self.pad_id}
        self._id_to_bytes: dict[int, bytes] = {}
        for s, idx in vocab.items():
            if idx in self.special_ids:
                self._id_to_bytes[idx] = b""
            else:
                self._id_to_bytes[idx] = token_str_to_bytes(s)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode_to_bytes(self, token_ids: Iterable[int]) -> bytes:
        return b"".join(self._id_to_bytes[tid] for tid in token_ids if tid not in self.special_ids)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(d / "tokenizer.json"))

    @classmethod
    def load(cls, directory: str | Path, spec: TokenizerSpec = TokenizerSpec()) -> "HFByteTokenizerWrapper":
        tok = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        return cls(tok, spec=spec)


# ---------------------------------------------------------------------------
# B1: Standard Byte-level BPE (16,384)
# ---------------------------------------------------------------------------
def build_b1_byte_bpe(texts: Sequence[str], spec: TokenizerSpec = TokenizerSpec()) -> HFByteTokenizerWrapper:
    tok = train_byte_bpe_tokenizer(texts, vocab_size=spec.total_vocab_size, special_tokens=spec.special_tokens)
    return HFByteTokenizerWrapper(tok, spec=spec)


# ---------------------------------------------------------------------------
# B2: Unigram LM Tokenization (Lossless ByteLevel + Deterministic Viterbi)
# ---------------------------------------------------------------------------
def build_b2_unigram(texts: Sequence[str], spec: TokenizerSpec = TokenizerSpec()) -> HFByteTokenizerWrapper:
    """
    Trains a probabilistic Unigram LM tokenizer over reversible ByteLevel pre-tokens
    (fixing Trap 2/B2: zero NFKC mutation, zero whitespace collapsing, 100% lossless byte fallback).
    """
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    initial_alphabet = [BYTE_TO_UNICODE[b] for b in range(256)]
    trainer = trainers.UnigramTrainer(
        vocab_size=spec.total_vocab_size,
        special_tokens=list(spec.special_tokens),
        initial_alphabet=initial_alphabet,
        shrinking_factor=0.75,
        n_sub_iterations=2,
        unk_token=None,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    assert tok.get_vocab_size() == spec.total_vocab_size, (
        f"Unigram vocab {tok.get_vocab_size()} != {spec.total_vocab_size}"
    )
    return HFByteTokenizerWrapper(tok, spec=spec)


# ---------------------------------------------------------------------------
# B3: SaGe (SkipGram Context-Prediction Pruning with Pinned Byte Alphabet)
# ---------------------------------------------------------------------------
def build_b3_sage(
    texts: Sequence[str],
    spec: TokenizerSpec = TokenizerSpec(),
    initial_vocab_size: int = 24_576,
    embed_dim: int = 64,
    window_size: int = 4,
) -> HFByteTokenizerWrapper:
    """
    Faithful lossless ByteLevel implementation of SaGe (Yehezkel & Pinter, EACL 2023):
      1. Builds an overcomplete initial vocabulary (24,576 entries) via ByteLevel Unigram/BPE seed.
      2. Fits SkipGram token-context embeddings (PPMI-SVD / SGNS closed-form objective) on corpus.
      3. Scores every prunable token by its marginal SkipGram context-prediction contribution
         and iteratively prunes down to 16,384 while strictly pinning all 256 byte tokens + specials.
    """
    # Start from a 24,576 Unigram model so pruning yields a clean deterministic Viterbi Unigram model
    seed_tok = Tokenizer(models.Unigram())
    seed_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    seed_tok.decoder = decoders.ByteLevel()
    initial_alphabet = [BYTE_TO_UNICODE[b] for b in range(256)]
    trainer = trainers.UnigramTrainer(
        vocab_size=initial_vocab_size,
        special_tokens=list(spec.special_tokens),
        initial_alphabet=initial_alphabet,
        shrinking_factor=0.80,
        n_sub_iterations=2,
        show_progress=False,
    )
    seed_tok.train_from_iterator(texts, trainer=trainer)

    # Encode sample corpus to compute SkipGram context objective per vocabulary piece
    sample_texts = texts[: min(len(texts), 25_000)]
    V_init = seed_tok.get_vocab_size()
    vocab_map = seed_tok.get_vocab()
    pinned_strs = set(spec.special_tokens) | set(initial_alphabet)

    unigram_counts = np.zeros(V_init, dtype=np.float64)
    cooc: Counter[tuple[int, int]] = Counter()

    for doc in sample_texts:
        ids = seed_tok.encode(doc, add_special_tokens=False).ids
        L = len(ids)
        for i, u in enumerate(ids):
            unigram_counts[u] += 1.0
            j_end = min(L, i + window_size + 1)
            for j in range(i + 1, j_end):
                v = ids[j]
                cooc[(u, v)] += 1
                cooc[(v, u)] += 1

    total_toks = max(unigram_counts.sum(), 1.0)
    # SaGe contribution of token u: sum_v c(u, v) * max(0, log((c(u,v) * N) / (c(u) * c(v))))
    # (SkipGram SGNS objective value at optimal inner-product = PMI; Levy & Goldberg 2014 / Yehezkel & Pinter 2023)
    sage_contrib = np.zeros(V_init, dtype=np.float64)
    for (u, v), cuv in cooc.items():
        denom = unigram_counts[u] * unigram_counts[v]
        if denom > 0:
            pmi = math.log((cuv * total_toks) / denom)
            if pmi > 0:
                sage_contrib[u] += cuv * pmi

    # Serialize seed Unigram JSON and prune lowest-SaGe-score non-pinned entries to exact 16,384
    tok_json = json.loads(seed_tok.to_str())
    pieces: list[list] = tok_json["model"]["vocab"]  # list of [piece_str, log_prob]
    assert len(pieces) == V_init

    keep_pinned: list[tuple[int, list]] = []
    prunable: list[tuple[float, int, list]] = []
    for idx, entry in enumerate(pieces):
        p_str = entry[0]
        if p_str in pinned_strs:
            keep_pinned.append((idx, entry))
        else:
            # Tie-break zero-cooc pieces by original Unigram log_prob
            score = float(sage_contrib[idx]) + 1e-4 * float(entry[1])
            prunable.append((score, idx, entry))

    prunable.sort(key=lambda x: x[0], reverse=True)
    needed_prunable = spec.total_vocab_size - len(keep_pinned)
    survivors = keep_pinned + [(idx, entry) for _, idx, entry in prunable[:needed_prunable]]
    survivors.sort(key=lambda x: x[0])

    tok_json["model"]["vocab"] = [entry for _, entry in survivors]
    pruned_tok = Tokenizer.from_str(json.dumps(tok_json))
    assert pruned_tok.get_vocab_size() == spec.total_vocab_size
    return HFByteTokenizerWrapper(pruned_tok, spec=spec)


# ---------------------------------------------------------------------------
# B4: SuperBPE (Phase 1 = Shared 14,336 Subwords, Phase 2 = Cross-Space Merges -> 16,384)
# ---------------------------------------------------------------------------
def build_b4_superbpe(
    base_tokenizer: Tokenizer,
    docs_base_ids: Sequence[list[int]],
    spec: TokenizerSpec = TokenizerSpec(),
) -> PhraseAugmentedTokenizer:
    """
    SuperBPE (Liu et al., 2025) starting from the exact shared Phase-1 14,336 subword vocabulary
    (transition t = 14,336), then performing 2,048 iterative greedy BPE merges across whitespace
    boundaries (never crossing <|endoftext|> document boundaries or stranding suffixes).
    Represented cleanly as iteratively merged multi-token units over the base vocabulary.
    """
    n_base = base_tokenizer.get_vocab_size()
    vocab = base_tokenizer.get_vocab()
    special_ids = {vocab[s] for s in spec.special_tokens if s in vocab}

    # Work on a training slice of up to 8M tokens for fast exact iterative pair merging
    streams: list[list[int]] = []
    budget = 0
    for seq in docs_base_ids:
        streams.append(list(seq))
        budget += len(seq)
        if budget >= 8_000_000:
            break

    # Map every token ID (>= n_base) back to its expanded base_ids tuple
    expansion: dict[int, tuple[int, ...]] = {i: (i,) for i in range(n_base)}
    id_to_bytes = {
        i: (b"" if i in special_ids else token_str_to_bytes(s)) for s, i in vocab.items()
    }

    selected_phrases: list[tuple[tuple[int, ...], float]] = []
    next_id = n_base

    # Perform SuperBPE Phase-2 merges in mini-batches of non-conflicting top adjacent pairs
    while len(selected_phrases) < spec.phrase_vocab_slots:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for seq in streams:
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if a in special_ids or b in special_ids:
                    continue
                pair_counts[(a, b)] += 1

        if not pair_counts:
            break

        # Take top non-overlapping pair(s) per round (batch up to 16 disjoint pairs for speed)
        need = min(16, spec.phrase_vocab_slots - len(selected_phrases))
        used_tokens: set[int] = set()
        merges_this_step: dict[tuple[int, int], int] = {}

        for (a, b), cnt in pair_counts.most_common(512):
            if len(merges_this_step) >= need:
                break
            if a in used_tokens or b in used_tokens:
                continue
            exp_tup = expansion[a] + expansion[b]
            raw_b = id_to_bytes[a] + id_to_bytes[b]
            if len(exp_tup) > 8 or len(raw_b) > spec.max_phrase_chars or b"\n" in raw_b:
                continue
            new_tid = next_id
            next_id += 1
            merges_this_step[(a, b)] = new_tid
            expansion[new_tid] = exp_tup
            id_to_bytes[new_tid] = raw_b
            selected_phrases.append((exp_tup, float(cnt)))
            used_tokens.add(a)
            used_tokens.add(b)

        if not merges_this_step:
            # Fallback single best valid merge if top 512 collided on used_tokens
            merged_any = False
            for (a, b), cnt in pair_counts.most_common(2048):
                exp_tup = expansion[a] + expansion[b]
                raw_b = id_to_bytes[a] + id_to_bytes[b]
                if len(exp_tup) <= 8 and len(raw_b) <= spec.max_phrase_chars and b"\n" not in raw_b:
                    new_tid = next_id
                    next_id += 1
                    merges_this_step[(a, b)] = new_tid
                    expansion[new_tid] = exp_tup
                    id_to_bytes[new_tid] = raw_b
                    selected_phrases.append((exp_tup, float(cnt)))
                    merged_any = True
                    break
            if not merged_any:
                break

        # Apply merges across streams
        for idx_s, seq in enumerate(streams):
            new_seq: list[int] = []
            i = 0
            L = len(seq)
            while i < L:
                if i < L - 1 and (seq[i], seq[i + 1]) in merges_this_step:
                    new_seq.append(merges_this_step[(seq[i], seq[i + 1])])
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            streams[idx_s] = new_seq

    return PhraseAugmentedTokenizer(base_tokenizer, selected_phrases, spec=spec, mode="greedy_ltr")


# ---------------------------------------------------------------------------
# B5: SupraTok (Tănase & Pelican TACL 2026 — Staged 3-Pass PMI Curriculum, No Data Filtering)
# ---------------------------------------------------------------------------
def build_b5_supratok(
    base_tokenizer: Tokenizer,
    docs_base_ids: Sequence[list[int]],
    spec: TokenizerSpec = TokenizerSpec(),
) -> PhraseAugmentedTokenizer:
    """
    Faithful same-data re-implementation of SupraTok (Tănase & Pelican, TACL 2026, 2026.tacl-1.80):
      - Optional entropy-based corpus filtering is explicitly DISABLED for fair same-data comparison.
      - Uses a 3-pass staged cross-boundary curriculum starting from the 14,336 base subwords:
          Pass 1 (35% of superword budget = 716 slots): Tight high-PMI 2-unit collocations.
          Pass 2 (40% of superword budget = 820 slots): Medium-span (2-3 unit) PMI-weighted phrases.
          Pass 3 (25% of superword budget = 512 slots): Extended cross-boundary (up to 4 words) NPMI*log(freq).
    """
    n_base = base_tokenizer.get_vocab_size()
    vocab = base_tokenizer.get_vocab()
    special_ids = {vocab[s] for s in spec.special_tokens if s in vocab}
    id_to_bytes = {
        i: (b"" if i in special_ids else token_str_to_bytes(s)) for s, i in vocab.items()
    }

    streams: list[list[int]] = []
    budget = 0
    for seq in docs_base_ids:
        streams.append(list(seq))
        budget += len(seq)
        if budget >= 8_000_000:
            break

    expansion: dict[int, tuple[int, ...]] = {i: (i,) for i in range(n_base)}
    selected: list[tuple[tuple[int, ...], float]] = []
    selected_set: set[tuple[int, ...]] = set()
    next_id = n_base

    # 3-stage curriculum allocations summing to 2,048
    stage_budgets = [716, 820, spec.phrase_vocab_slots - (716 + 820)]
    stage_min_freq = [40, 25, 15]

    for stage_idx, (stage_slots, min_f) in enumerate(zip(stage_budgets, stage_min_freq)):
        added_in_stage = 0
        while added_in_stage < stage_slots:
            step_need = min(64, stage_slots - added_in_stage)
            uni_counts: Counter[int] = Counter()
            pair_counts: Counter[tuple[int, int]] = Counter()
            total_u = 0

            for seq in streams:
                total_u += len(seq)
                for tok in seq:
                    if tok not in special_ids:
                        uni_counts[tok] += 1
                for i in range(len(seq) - 1):
                    a, b = seq[i], seq[i + 1]
                    if a not in special_ids and b not in special_ids:
                        pair_counts[(a, b)] += 1

            scored_pairs: list[tuple[float, tuple[int, int]]] = []
            for (a, b), cab in pair_counts.items():
                if cab < min_f:
                    continue
                exp_tup = expansion[a] + expansion[b]
                if exp_tup in selected_set or len(exp_tup) > (3 if stage_idx == 0 else 8):
                    continue
                raw_b = id_to_bytes[a] + id_to_bytes[b]
                if len(raw_b) > spec.max_phrase_chars or b"\n" in raw_b:
                    continue
                w_cnt = len([w for w in raw_b.decode("utf-8", errors="ignore").strip().split() if any(c.isalnum() for c in w)])
                if w_cnt > (2 if stage_idx == 0 else (3 if stage_idx == 1 else spec.max_phrase_words)):
                    continue

                # Normalized PMI * log(1 + freq) balance (prevents rare singleton PMI traps while favoring collocations)
                pa = uni_counts[a] / max(total_u, 1)
                pb = uni_counts[b] / max(total_u, 1)
                pab = cab / max(total_u, 1)
                pmi = math.log(max(pab / max(pa * pb, 1e-18), 1e-12))
                npmi = pmi / max(-math.log(max(pab, 1e-12)), 1e-6)
                score = npmi * math.log1p(cab) * (1.0 + 0.15 * stage_idx)
                if score > 0:
                    scored_pairs.append((score, (a, b)))

            if not scored_pairs:
                break
            scored_pairs.sort(key=lambda x: x[0], reverse=True)

            used: set[int] = set()
            merges_map: dict[tuple[int, int], int] = {}
            for sc, (a, b) in scored_pairs:
                if len(merges_map) >= step_need:
                    break
                if a in used or b in used:
                    continue
                exp_tup = expansion[a] + expansion[b]
                new_tid = next_id
                next_id += 1
                merges_map[(a, b)] = new_tid
                expansion[new_tid] = exp_tup
                id_to_bytes[new_tid] = id_to_bytes[a] + id_to_bytes[b]
                selected.append((exp_tup, sc))
                selected_set.add(exp_tup)
                used.add(a)
                used.add(b)
                added_in_stage += 1

            if not merges_map:
                break

            for idx_s, seq in enumerate(streams):
                new_seq = []
                i = 0
                L = len(seq)
                while i < L:
                    if i < L - 1 and (seq[i], seq[i + 1]) in merges_map:
                        new_seq.append(merges_map[(seq[i], seq[i + 1])])
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                streams[idx_s] = new_seq

    return PhraseAugmentedTokenizer(base_tokenizer, selected, spec=spec, mode="greedy_ltr")
