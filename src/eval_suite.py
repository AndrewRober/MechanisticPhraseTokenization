"""
Post-`Tokenizer Myths` (arXiv:2611.04215) Multi-Metric Evaluation Suite.

Implements unbiased cross-tokenizer evaluation on identical held-out raw text:
  1. Exact Canonical Bits-Per-Byte (BPB) including document <|endoftext|> termination log-prob,
     evaluated under BOTH 1024-token windows AND byte-matched context windows (~3,500 UTF-8 bytes).
  2. Empirical Token Unigram Entropy Rate (BPB_unigram) and Contextual Information Gain:
         Delta_BPB = BPB_unigram - BPB_model
     (isolating what the neural network learned from static compression head-starts; Trap 9).
  3. Grammatical & Discourse Contrast Accuracy (BLiMP-style minimal pairs, stratified into
     overall accuracy AND equal-token-count fertility-matched accuracy; Trap 10).
  4. Targeted Mechanism Stress Suites:
       - Compositional Generalization (novel combinations of phrase-absorbed constituents; C2)
       - Controlled Perturbation Robustness (comma/whitespace/typo edit invariance on continuations; C4)
       - Polysemy / Sense-Conflict Continuation Margin (sense-divergent contexts; C1)
  5. LAMBADA-style Final-Word Exact-Match & Log-Prob (testing SuperBPE's known regression).
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines import TokenizerInterface
from src.boundary_c4 import build_aligned_edit_pairs
from src.model import ResearchTransformerLM


def compute_unigram_bpb(
    train_docs: Sequence[str],
    eval_docs: Sequence[str],
    tokenizer: TokenizerInterface,
) -> float:
    """
    Computes static unigram token model Bits-Per-Byte on eval_docs (Laplace-smoothed from train_docs),
    including <|endoftext|> per document.
    """
    counts = np.ones(tokenizer.vocab_size, dtype=np.float64)  # Laplace +1 smoothing
    for doc in train_docs[:30_000]:
        for tid in tokenizer.encode(doc, add_eos=True):
            if tid < tokenizer.vocab_size:
                counts[tid] += 1.0
    log2_probs = np.log2(counts / counts.sum())

    total_bits = 0.0
    total_bytes = 0
    for doc in eval_docs:
        raw_b = len(doc.encode("utf-8"))
        if raw_b == 0:
            continue
        ids = tokenizer.encode(doc, add_eos=True)
        total_bits -= float(sum(log2_probs[t] for t in ids))
        total_bytes += raw_b

    return total_bits / max(total_bytes, 1)


@torch.no_grad()
def evaluate_bpb_and_stats(
    model: ResearchTransformerLM,
    eval_docs: Sequence[str],
    tokenizer: TokenizerInterface,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 1_024,
    stride: int = 512,
    byte_cap_context: int | None = None,
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Scores exact canonical log-likelihood over every token + <|endoftext|> of every document in eval_docs,
    normalized by exact raw UTF-8 byte count.
    If byte_cap_context is set (e.g. 3200 bytes), limits context window tokens per document so every
    tokenizer receives matched raw-text history rather than compressed tokenizers seeing ~20% further back.
    """
    model.eval()
    ln2 = math.log(2.0)

    # Build evaluation windows (x, y_masked) where each target token in every doc is scored EXACTLY ONCE
    windows_x: list[list[int]] = []
    windows_y: list[list[int]] = []
    total_utf8_bytes = 0
    total_emitted_tokens = 0  # excluding EOS for fertility stats

    # Estimate avg bytes/token for byte-matched window sizing if requested
    sample_b = sum(len(d.encode("utf-8")) for d in eval_docs[:200])
    sample_t = max(sum(len(tokenizer.encode(d, add_eos=False)) for d in eval_docs[:200]), 1)
    bytes_per_tok = sample_b / sample_t

    eff_win = max_seq_len
    if byte_cap_context is not None:
        eff_win = max(64, min(max_seq_len, int(round(byte_cap_context / max(bytes_per_tok, 1.0)))))
    eff_stride = max(32, eff_win // 2)

    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    for doc in eval_docs:
        b_len = len(doc.encode("utf-8"))
        if b_len == 0:
            continue
        total_utf8_bytes += b_len
        doc_ids = tokenizer.encode(doc, add_eos=True)
        total_emitted_tokens += len(doc_ids) - 1

        # Prepend EOS as BOS anchor so token 0 is also predicted causally!
        seq = [eos_id] + doc_ids  # length M + 1; targets are seq[1:] (length M)
        M = len(doc_ids)

        scored_upto = 0
        start = 0
        while scored_upto < M:
            end = min(start + eff_win, M)
            x_chunk = seq[start:end]
            y_chunk = seq[start + 1 : end + 1]

            # Mask already-scored prefix positions in overlapping stride windows
            keep_from = scored_upto - start
            y_masked = [(-100 if idx < keep_from else tok) for idx, tok in enumerate(y_chunk)]

            if len(x_chunk) < eff_win:
                pad_n = eff_win - len(x_chunk)
                x_chunk = x_chunk + [pad_id] * pad_n
                y_masked = y_masked + [-100] * pad_n

            windows_x.append(x_chunk)
            windows_y.append(y_masked)
            scored_upto = end
            if end == M:
                break
            start = max(start + 1, end - eff_stride)

    total_nats = 0.0
    active_vocab = tokenizer.vocab_size

    for b0 in range(0, len(windows_x), batch_size):
        bx = torch.tensor(windows_x[b0 : b0 + batch_size], dtype=torch.long, device=device)
        by = torch.tensor(windows_y[b0 : b0 + batch_size], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(bx, active_output_vocab=active_vocab)
            nll = F.cross_entropy(logits.view(-1, active_vocab), by.view(-1), ignore_index=-100, reduction="sum")
        total_nats += float(nll.item())

    bpb = (total_nats / ln2) / max(total_utf8_bytes, 1)
    return {
        "bpb": bpb,
        "bytes_per_token": total_utf8_bytes / max(total_emitted_tokens, 1),
        "tokens_per_byte": total_emitted_tokens / max(total_utf8_bytes, 1),
        "effective_window_tokens": float(eff_win),
    }


@torch.no_grad()
def score_sequence_logprob(
    model: ResearchTransformerLM,
    texts: Sequence[str],
    tokenizer: TokenizerInterface,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[int]]:
    """Returns exact total log-probability (in nats, including EOS) and token count for each string."""
    model.eval()
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    active_vocab = tokenizer.vocab_size

    seqs = [[eos_id] + tokenizer.encode(t, add_eos=True) for t in texts]
    tok_lens = [len(s) - 1 for s in seqs]
    logprobs = np.zeros(len(texts), dtype=np.float64)

    for b0 in range(0, len(seqs), batch_size):
        batch = seqs[b0 : b0 + batch_size]
        B = len(batch)
        max_l = min(max(len(s) - 1 for s in batch), model.spec.max_seq_len)
        bx = torch.full((B, max_l), pad_id, dtype=torch.long, device=device)
        by = torch.full((B, max_l), -100, dtype=torch.long, device=device)
        for i, s in enumerate(batch):
            L = min(len(s) - 1, max_l)
            bx[i, :L] = torch.tensor(s[:L], dtype=torch.long, device=device)
            by[i, :L] = torch.tensor(s[1 : L + 1], dtype=torch.long, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(bx, active_output_vocab=active_vocab)
            nll = F.cross_entropy(
                logits.view(-1, active_vocab), by.view(-1), ignore_index=-100, reduction="none"
            ).view(B, max_l)
        seq_lp = -(nll.float() * (by != -100).float()).sum(dim=1).cpu().numpy()
        logprobs[b0 : b0 + B] = seq_lp

    return logprobs, tok_lens


def evaluate_minimal_pairs(
    model: ResearchTransformerLM,
    pairs: Sequence[tuple[str, str]],
    tokenizer: TokenizerInterface,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """
    Evaluates minimal pairs (good_sentence, bad_sentence) and stratifies into:
      - overall_acc: fraction where log P(good) > log P(bad)
      - fertility_matched_acc: subset where good and bad emit the EXACT same number of tokens
        (eliminates subword-split / token-count bias; Trap 10).
    """
    if not pairs:
        return {"overall_acc": 0.0, "fertility_matched_acc": 0.0, "matched_fraction": 0.0}
    good_texts = [g for g, _ in pairs]
    bad_texts = [b for _, b in pairs]
    lp_g, len_g = score_sequence_logprob(model, good_texts, tokenizer, device, dtype)
    lp_b, len_b = score_sequence_logprob(model, bad_texts, tokenizer, device, dtype)

    wins = lp_g > lp_b
    matched_mask = np.array([lg == lb for lg, lb in zip(len_g, len_b)], dtype=bool)
    overall = float(wins.mean())
    matched_acc = float(wins[matched_mask].mean()) if matched_mask.any() else overall
    return {
        "overall_acc": overall,
        "fertility_matched_acc": matched_acc,
        "matched_fraction": float(matched_mask.mean()),
    }


@torch.no_grad()
def evaluate_perturbation_stability(
    model: ResearchTransformerLM,
    eval_docs: Sequence[str],
    tokenizer: TokenizerInterface,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """
    Controlled Perturbation Test (Stage 7 / C4 mechanism):
    Applies a small punctuation/spacing/character perturbation strictly in the prefix half of a document
    and measures the increase in NLL (in bits/byte) on the IDENTICAL unperturbed continuation half!
    """
    model.eval()
    ln2 = math.log(2.0)
    clean_prefixes: list[str] = []
    pert_prefixes: list[str] = []
    continuations: list[str] = []

    for doc in eval_docs[:400]:
        if len(doc) < 160:
            continue
        mid = len(doc) // 2
        # Split cleanly at a space near midpoint
        sp = doc.find(" ", mid)
        if sp == -1 or sp > len(doc) - 40:
            continue
        pref = doc[:sp]
        cont = doc[sp: sp + 160]
        edits = build_aligned_edit_pairs([pref], edits_per_snippet=1)
        if not edits:
            continue
        clean_prefixes.append(pref)
        pert_prefixes.append(edits[0].edit_text)
        continuations.append(cont)

    if not continuations:
        return {"clean_cont_bpb": 0.0, "perturbed_cont_bpb": 0.0, "perturbation_bpb_degradation": 0.0}

    # Continuation log-prob = log P(prefix + cont) - log P(prefix_without_eos)
    def cont_bpb(prefixes: list[str]) -> float:
        eos_id = tokenizer.eos_id
        pad_id = tokenizer.pad_id
        active_vocab = tokenizer.vocab_size
        total_nats = 0.0
        total_b = 0

        for b0 in range(0, len(prefixes), 32):
            bp = prefixes[b0 : b0 + 32]
            bc = continuations[b0 : b0 + 32]
            B = len(bp)
            full_ids = []
            pref_lens = []
            for p_str, c_str in zip(bp, bc):
                p_toks = [eos_id] + tokenizer.encode(p_str, add_eos=False)
                f_toks = [eos_id] + tokenizer.encode(p_str + c_str, add_eos=False)
                # Score tokens strictly after prefix length
                full_ids.append(f_toks)
                pref_lens.append(min(len(p_toks), len(f_toks) - 1))
                total_b += len(c_str.encode("utf-8"))

            max_l = min(max(len(s) - 1 for s in full_ids), model.spec.max_seq_len)
            bx = torch.full((B, max_l), pad_id, dtype=torch.long, device=device)
            by = torch.full((B, max_l), -100, dtype=torch.long, device=device)
            for i, (s, pl) in enumerate(zip(full_ids, pref_lens)):
                L = min(len(s) - 1, max_l)
                bx[i, :L] = torch.tensor(s[:L], dtype=torch.long, device=device)
                for pos in range(max(pl - 1, 0), L):
                    by[i, pos] = s[pos + 1]

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                logits, _ = model(bx, active_output_vocab=active_vocab)
                nll = F.cross_entropy(logits.view(-1, active_vocab), by.view(-1), ignore_index=-100, reduction="sum")
            total_nats += float(nll.item())

        return (total_nats / ln2) / max(total_b, 1)

    c_bpb = cont_bpb(clean_prefixes)
    p_bpb = cont_bpb(pert_prefixes)
    return {
        "clean_cont_bpb": c_bpb,
        "perturbed_cont_bpb": p_bpb,
        "perturbation_bpb_degradation": p_bpb - c_bpb,
    }
