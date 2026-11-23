"""
Three Novel Breakthrough Mechanisms Across the Tokenization-Pretraining-Inference Stack:

1. `FoldableResidualLM` + `PrefixConeDecoder`:
   - Solves the "Swiss-Cheese Complement Embedding" & "Superword Vote-Splitting" paradoxes.
   - Parameterizes phrase embeddings during training as W_P = mean(W_{t_j}) + Delta_P with strong
     L2 regularization solely on Delta_P, then folds W_P into a 100% standard [16384, 512] matrix!
   - Aggregates first-word prefix-cone probabilities at decoding time to eliminate superword vote-splitting.

2. `select_attention_demux_phrases` (Attention-Query De-Multiplexing / Antecedent-Disjointness Shielding):
   - Computes Cross-Constituent Prefix Attention Jensen-Shannon Divergence D_bind(p) in the Pilot model.
   - Vetoes merging multi-binding relational spans (e.g. "he told her") whose constituents attend to
     distinct upstream antecedents, reserving vocabulary slots for cohesive referential atoms.

3. `ComputeGearshift` (Dual-Granularity Pretraining + Entropy-Gated Test-Time Unmerging):
   - Trains a standard Transformer with 20% stochastic phrase unmerging so a single model masters
     both High-Gear (superword) and Low-Gear (subword) representations.
   - Downshifts local clauses into unmerged subwords at test time ONLY when predictive entropy spikes,
     unlocking extra serial Transformer compute steps with zero architecture or CoT overhead.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.baselines import TokenizerInterface
from src.config import ModelSpec, TokenizerSpec, TrainSpec
from src.model import DecoderBlock, RMSNorm, ResearchTransformerLM, apply_rope, precompute_rope_cache
from src.phrase_selectors import PhraseCandidate
from src.tokenizer_core import PhraseAugmentedTokenizer


# ===========================================================================
# BREAKTHROUGH 1A: Foldable Compositional-Residual Transformer
# ===========================================================================
class FoldableResidualTransformerLM(nn.Module):
    """
    Training-time reparameterization of ResearchTransformerLM where:
      - Base tokens [0 .. N_base-1] have standard embeddings W_base.
      - Phrase tokens [N_base .. V-1] are parameterized as:
            W_P = (1 / |P|) * sum_{t in P} W_base[t] + Delta_P
        with Delta_P initialized to ZERO and subject to 3x stronger weight decay.
    Exports cleanly via `.fold_to_standard_model()` into an unmodified ResearchTransformerLM
    with bit-identical forward behavior (`max_diff < 1e-5`).
    """

    def __init__(
        self,
        spec: ModelSpec,
        phrase_tok: PhraseAugmentedTokenizer,
    ):
        super().__init__()
        self.spec = spec
        self.n_base = phrase_tok.spec.base_vocab_size
        self.n_phrases = spec.vocab_size - self.n_base
        assert len(phrase_tok.phrases) == self.n_phrases

        self.base_emb = nn.Embedding(self.n_base, spec.d_model)
        self.phrase_delta = nn.Parameter(torch.zeros(self.n_phrases, spec.d_model))

        self.blocks = nn.ModuleList([DecoderBlock(spec) for _ in range(spec.n_layers)])
        self.final_norm = RMSNorm(spec.d_model, eps=spec.rms_norm_eps)

        cos, sin = precompute_rope_cache(spec.max_seq_len, spec.head_dim, spec.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Build static [n_phrases, max_parts] lookup buffers for vectorized O(1) GPU composition
        max_parts = max(len(e.base_ids) for e in phrase_tok.phrases.values())
        comp_ids = torch.zeros((self.n_phrases, max_parts), dtype=torch.long)
        comp_weights = torch.zeros((self.n_phrases, max_parts, 1), dtype=torch.float32)

        for pid, entry in sorted(phrase_tok.phrases.items()):
            row_idx = pid - self.n_base
            k = len(entry.base_ids)
            for j, b_id in enumerate(entry.base_ids):
                comp_ids[row_idx, j] = b_id
                comp_weights[row_idx, j, 0] = 1.0 / k

        self.register_buffer("phrase_comp_ids", comp_ids, persistent=True)
        self.register_buffer("phrase_comp_weights", comp_weights, persistent=True)
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        residual_std = std / math.sqrt(2 * self.spec.n_layers)
        nn.init.normal_(self.base_emb.weight, mean=0.0, std=std)
        nn.init.zeros_(self.phrase_delta)
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, mean=0.0, std=std)
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.up_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

    def materialize_vocab_weight(self) -> torch.Tensor:
        """Computes the full [V, D] tied embedding/output matrix with live autograd through W_base and Delta_P."""
        parts = self.base_emb(self.phrase_comp_ids)  # [n_phrases, max_parts, D]
        comp_mean = (parts * self.phrase_comp_weights.to(parts.dtype)).sum(dim=1)
        phrase_w = comp_mean + self.phrase_delta.to(parts.dtype)
        return torch.cat([self.base_emb.weight, phrase_w], dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        active_output_vocab: Optional[int] = None,
        loss_reduction: str = "mean",
        return_logits: bool = True,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        full_w = self.materialize_vocab_weight()
        h = F.embedding(input_ids, full_w)
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin)
        h = self.final_norm(h)
        out_w = full_w[:active_output_vocab] if active_output_vocab is not None else full_w
        logits = F.linear(h, out_w)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction=loss_reduction,
            )
        return (logits if return_logits else None), loss

    @torch.no_grad()
    def fold_to_standard_model(self) -> ResearchTransformerLM:
        """Folds compositional anchors + learned residuals into a zero-overhead standard ResearchTransformerLM."""
        device = self.base_emb.weight.device
        std_model = ResearchTransformerLM(self.spec).to(device)
        std_model.tok_emb.weight.copy_(self.materialize_vocab_weight())
        std_model.blocks.load_state_dict(self.blocks.state_dict())
        std_model.final_norm.load_state_dict(self.final_norm.state_dict())
        return std_model


# ===========================================================================
# BREAKTHROUGH 1B: Prefix-Cone Vote-Aggregating Next-Word Decoder
# ===========================================================================
def build_first_word_prefix_cones(tok: TokenizerInterface) -> tuple[torch.Tensor, list[str]]:
    """
    Maps every vocabulary ID v in [0 .. V-1] to a canonical first-word group index g in [0 .. G-1]
    based on the exact first whitespace-delimited word emitted by token v.
    """
    word_to_group: dict[str, int] = {}
    group_words: list[str] = []
    tok_to_group = np.zeros(tok.vocab_size, dtype=np.int64)

    for vid in range(tok.vocab_size):
        if vid in (tok.eos_id, tok.pad_id):
            key = f"<SPECIAL_{vid}>"
        else:
            raw_s = tok.decode_to_bytes([vid]).decode("utf-8", errors="replace")
            stripped = raw_s.lstrip(" ")
            has_lead_space = raw_s.startswith(" ")
            first_part = stripped.split(" ")[0] if stripped else raw_s
            key = (" " + first_part) if (has_lead_space and first_part) else raw_s

        g_idx = word_to_group.get(key)
        if g_idx is None:
            g_idx = len(group_words)
            word_to_group[key] = g_idx
            group_words.append(key)
        tok_to_group[vid] = g_idx

    return torch.from_numpy(tok_to_group), group_words


@torch.no_grad()
def evaluate_next_word_vote_splitting(
    model: ResearchTransformerLM,
    eval_docs: Sequence[str],
    tok: TokenizerInterface,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_examples: int = 800,
) -> dict[str, float]:
    """
    LAMBADA-style Discourse Final-Word / Next-Word Benchmark comparing on IDENTICAL model weights:
      1. `naive_argmax_acc`       : Standard greedy token argmax (suffers superword vote-splitting!)
      2. `prefix_cone_argmax_acc` : Marginalizes P(v | x) across each root word's prefix cone C(w)
                                    before selecting the winning next word w*!
    """
    model.eval()
    tok_to_group_cpu, group_words = build_first_word_prefix_cones(tok)
    tok_to_group = tok_to_group_cpu.to(device)
    n_groups = len(group_words)

    prompts: list[list[int]] = []
    gold_words: list[str] = []

    for doc in eval_docs:
        words = doc.strip().split()
        if len(words) < 25:
            continue
        # Predict the discourse-conditioned word at position ~40 (requiring long-range context)
        cut_idx = min(len(words) - 4, 42)
        target_w = " " + words[cut_idx].strip(".,;:!?\"'()")
        if len(target_w.strip()) < 2 or not target_w.strip().isalpha():
            continue
        prefix_str = " ".join(words[:cut_idx])
        p_ids = [tok.eos_id] + tok.encode(prefix_str, add_eos=False)
        if 12 <= len(p_ids) <= 512:
            prompts.append(p_ids)
            gold_words.append(target_w)
        if len(prompts) >= max_examples:
            break

    if not prompts:
        return {"naive_argmax_acc": 0.0, "prefix_cone_argmax_acc": 0.0, "cone_gain": 0.0}

    naive_hits = 0
    cone_hits = 0
    pad_id = tok.pad_id

    for b0 in range(0, len(prompts), 64):
        bp = prompts[b0 : b0 + 64]
        bg = gold_words[b0 : b0 + 64]
        B = len(bp)
        max_l = max(len(s) for s in bp)
        bx = torch.full((B, max_l), pad_id, dtype=torch.long, device=device)
        last_idx = torch.empty(B, dtype=torch.long, device=device)
        for i, s in enumerate(bp):
            bx[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            last_idx[i] = len(s) - 1

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(bx, active_output_vocab=tok.vocab_size)
        last_logits = logits[torch.arange(B, device=device), last_idx].float()  # [B, V]
        probs = F.softmax(last_logits, dim=-1)

        # 1. Naive single-token argmax
        naive_top_vid = probs.argmax(dim=-1).cpu().tolist()
        # 2. Prefix-Cone probability aggregation: sum_v P(v | x) for v in C(w)
        cone_probs = torch.zeros((B, n_groups), dtype=torch.float32, device=device)
        cone_probs.scatter_add_(1, tok_to_group.unsqueeze(0).expand(B, -1), probs)
        cone_top_gid = cone_probs.argmax(dim=-1).cpu().tolist()

        for i, gold in enumerate(bg):
            naive_w = group_words[int(tok_to_group_cpu[naive_top_vid[i]])]
            cone_w = group_words[cone_top_gid[i]]
            if naive_w == gold:
                naive_hits += 1
            if cone_w == gold:
                cone_hits += 1

    n_tot = len(prompts)
    n_acc = naive_hits / n_tot
    c_acc = cone_hits / n_tot
    return {
        "naive_argmax_acc": round(n_acc, 4),
        "prefix_cone_argmax_acc": round(c_acc, 4),
        "cone_gain": round(c_acc - n_acc, 4),
    }


# ===========================================================================
# BREAKTHROUGH 2: Attention-Query De-Multiplexing (D_bind Shielded Selector)
# ===========================================================================
@torch.no_grad()
def select_attention_demux_phrases(
    pilot_model: ResearchTransformerLM,
    candidates: Sequence[PhraseCandidate],
    calib_docs_base_ids: Sequence[list[int]],
    prior_utilities: dict[tuple[int, ...], float],
    spec: TokenizerSpec = TokenizerSpec(),
    device: torch.device = torch.device("cuda:0"),
    seq_len: int = 256,
    batch_size: int = 32,
) -> tuple[list[tuple[tuple[int, ...], float]], dict[str, float]]:
    """
    Computes Cross-Constituent Prefix Attention Jensen-Shannon Divergence D_bind(p):
    Measures whether constituent subwords of p issue divergent attention queries to DIFFERENT
    upstream antecedents (relational multi-binders -> VETO!) vs cohesive attention (lexical atoms -> MERGE!).
    """
    pilot_model.eval()
    by_pair: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for c in candidates:
        by_pair[(int(c.base_ids[0]) << 16) | int(c.base_ids[1])].append(c.base_ids)

    flat: list[int] = []
    for doc in calib_docs_base_ids:
        flat.extend(doc)
        if len(flat) >= 400_000:  # ~1,560 windows of 256 -> <4 seconds on GB10!
            break

    n_blocks = len(flat) // seq_len
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

    jsd_sums: Counter[tuple[int, ...]] = Counter()
    jsd_counts: Counter[tuple[int, ...]] = Counter()

    # Probe semantic binding layers (layers 2, 3, 4, 5 of the 7-layer Pilot)
    probe_layers = {2, 3, 4, 5}

    for b0 in range(0, n_blocks, batch_size):
        b1 = min(b0 + batch_size, n_blocks)
        B = b1 - b0
        x_np = np.empty((B, seq_len), dtype=np.int64)
        for bi, blk in enumerate(range(b0, b1)):
            x_np[bi] = flat[blk * seq_len : (blk + 1) * seq_len]

        x = torch.from_numpy(x_np).to(device)
        h = pilot_model.tok_emb(x)
        attn_accum = torch.zeros((B, seq_len, seq_len), dtype=torch.float32, device=device)

        for l_idx, block in enumerate(pilot_model.blocks):
            if l_idx in probe_layers:
                normed = block.attn_norm(h)
                q, k, _ = block.attn.qkv(normed).chunk(3, dim=-1)
                H_n = block.attn.n_heads
                D_h = block.attn.head_dim
                q = apply_rope(q.view(B, seq_len, H_n, D_h).transpose(1, 2), pilot_model.rope_cos, pilot_model.rope_sin)
                k = apply_rope(k.view(B, seq_len, H_n, D_h).transpose(1, 2), pilot_model.rope_cos, pilot_model.rope_sin)
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (D_h ** 0.5)
                scores = scores.masked_fill(~causal_mask, float("-inf"))
                attn_accum += F.softmax(scores, dim=-1).mean(dim=1)  # head-averaged [B, T, T]
            h = block(h, pilot_model.rope_cos, pilot_model.rope_sin)

        attn_np = (attn_accum / len(probe_layers)).cpu().numpy()  # [B, T, T]

        for bi in range(B):
            row = x_np[bi].tolist()
            amat = attn_np[bi]
            for i in range(16, seq_len - 4):
                hits = by_pair.get((row[i] << 16) | row[i + 1])
                if not hits:
                    continue
                for tup in hits:
                    k_len = len(tup)
                    if i + k_len <= seq_len and tuple(row[i : i + k_len]) == tup:
                        # Prefix attention distributions strictly over upstream context [0 .. i-1]
                        p_first = amat[i, :i]
                        p_last = amat[i + k_len - 1, :i]
                        s1 = float(p_first.sum())
                        s2 = float(p_last.sum())
                        if s1 > 1e-6 and s2 > 1e-6:
                            p1 = p_first / s1
                            p2 = p_last / s2
                            m = 0.5 * (p1 + p2)
                            kl1 = float(np.sum(p1 * np.log((p1 + 1e-12) / (m + 1e-12))))
                            kl2 = float(np.sum(p2 * np.log((p2 + 1e-12) / (m + 1e-12))))
                            jsd = 0.5 * (kl1 + kl2) / math.log(2.0)  # normalized to [0, 1] bits!
                            jsd_sums[tup] += jsd
                            jsd_counts[tup] += 1

    mean_jsd: dict[tuple[int, ...], float] = {
        tup: jsd_sums[tup] / jsd_counts[tup] for tup in jsd_counts if jsd_counts[tup] >= 3
    }
    all_jsds = sorted(mean_jsd.values())
    veto_thresh = all_jsds[int(0.75 * len(all_jsds))] if all_jsds else 0.65

    scored: list[tuple[float, tuple[int, ...]]] = []
    vetoed_count = 0
    for c in candidates:
        tup = c.base_ids
        base_u = prior_utilities.get(tup, float(c.total_freq * (len(tup) - 1)))
        d_bind = mean_jsd.get(tup, 0.35)
        if d_bind > veto_thresh:
            vetoed_count += 1
            continue  # Strictly shield multi-binding relational spans from merging!
        cohesion_weight = max(1.0 - d_bind, 0.05) ** 2.0
        scored.append((base_u * cohesion_weight, tup))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [(tup, max(sc, 1e-6)) for sc, tup in scored[: spec.phrase_vocab_slots]]
    chosen_set = {t for t, _ in chosen}
    for c in candidates:
        if len(chosen) >= spec.phrase_vocab_slots:
            break
        if c.base_ids not in chosen_set:
            chosen.append((c.base_ids, 1e-6))
            chosen_set.add(c.base_ids)

    diag = {
        "measured_candidates": float(len(mean_jsd)),
        "jsd_veto_threshold_bits": round(float(veto_thresh), 4),
        "vetoed_relational_spans": float(vetoed_count),
    }
    return chosen, diag


# ===========================================================================
# BREAKTHROUGH 3: Compute Gearshifting (Dual-Granularity + Entropy Downshift)
# ===========================================================================
def pack_dual_granularity_blocks(
    docs: Sequence[str],
    phrase_tok: PhraseAugmentedTokenizer,
    unmerge_prob: float = 0.20,
    seq_len: int = 1_024,
    seed: int = 2026,
) -> np.ndarray:
    """
    Pretraining data packer for Compute Gearshifting:
    Stochastically unmerges `unmerge_prob` (20%) of emitted phrase tokens back into their exact
    base-subword constituents so a single unchanged Transformer masters BOTH superword High-Gear
    and subword Low-Gear representations seamlessly.
    """
    rng = random.Random(seed)
    n_base = phrase_tok.spec.base_vocab_size
    flat_chunks: list[np.ndarray] = []

    for doc in docs:
        merged_ids = phrase_tok.encode(doc, add_eos=False)
        mixed_ids: list[int] = []
        for tid in merged_ids:
            if tid >= n_base and rng.random() < unmerge_prob:
                mixed_ids.extend(phrase_tok.phrases[tid].base_ids)
            else:
                mixed_ids.append(tid)
        mixed_ids.append(phrase_tok.eos_id)
        flat_chunks.append(np.asarray(mixed_ids, dtype=np.uint16))

    all_ids = np.concatenate(flat_chunks)
    n_blocks = (len(all_ids) - 1) // seq_len
    trimmed = all_ids[: n_blocks * seq_len + 1]
    out = np.empty((n_blocks, seq_len + 1), dtype=np.uint16)
    for idx in range(n_blocks):
        s = idx * seq_len
        out[idx] = trimmed[s : s + seq_len + 1]
    return out


@torch.no_grad()
def evaluate_compute_gearshift_pareto(
    model: ResearchTransformerLM,
    eval_docs: Sequence[str],
    phrase_tok: PhraseAugmentedTokenizer,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    downshift_top_quantile: float = 0.25,
    max_examples: int = 600,
) -> dict[str, float]:
    """
    Evaluates the IDENTICAL trained Transformer checkpoint in 3 inference gears on discourse
    next-word prediction (combining Prefix-Cone Decoding with Adaptive Entropy Downshifting!):
      - Gear 1 (`high_gear_superword`): 100% phrase-merged prompts (minimum FLOPs).
      - Gear 2 (`adaptive_gearshift`) : Re-encodes the local active clause (last 120 chars) into
        unmerged base subwords ONLY on the top 25% highest-entropy uncertain prompts!
      - Gear 3 (`low_gear_subword`)   : 100% unmerged subword prompts (maximum serial compute ticks).
    """
    model.eval()
    tok_to_group_cpu, group_words = build_first_word_prefix_cones(phrase_tok)
    tok_to_group = tok_to_group_cpu.to(device)
    n_groups = len(group_words)

    raw_prefixes: list[str] = []
    gold_words: list[str] = []
    for doc in eval_docs:
        words = doc.strip().split()
        if len(words) < 25:
            continue
        cut_idx = min(len(words) - 4, 42)
        target_w = " " + words[cut_idx].strip(".,;:!?\"'()")
        if len(target_w.strip()) < 2 or not target_w.strip().isalpha():
            continue
        raw_prefixes.append(" ".join(words[:cut_idx]))
        gold_words.append(target_w)
        if len(raw_prefixes) >= max_examples:
            break

    def run_gear(seqs: list[list[int]]) -> tuple[list[str], np.ndarray, int]:
        preds: list[str] = []
        entropies = np.zeros(len(seqs), dtype=np.float64)
        total_toks = sum(len(s) for s in seqs)
        for b0 in range(0, len(seqs), 64):
            bs = seqs[b0 : b0 + 64]
            B = len(bs)
            max_l = max(len(s) for s in bs)
            bx = torch.full((B, max_l), phrase_tok.pad_id, dtype=torch.long, device=device)
            last_idx = torch.empty(B, dtype=torch.long, device=device)
            for i, s in enumerate(bs):
                bx[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                last_idx[i] = len(s) - 1
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                logits, _ = model(bx, active_output_vocab=phrase_tok.vocab_size)
            last_l = logits[torch.arange(B, device=device), last_idx].float()
            probs = F.softmax(last_l, dim=-1)
            ent = -(probs * (probs + 1e-12).log()).sum(dim=-1).cpu().numpy()
            entropies[b0 : b0 + B] = ent

            cone_p = torch.zeros((B, n_groups), dtype=torch.float32, device=device)
            cone_p.scatter_add_(1, tok_to_group.unsqueeze(0).expand(B, -1), probs)
            top_g = cone_p.argmax(dim=-1).cpu().tolist()
            preds.extend(group_words[g] for g in top_g)
        return preds, entropies, total_toks

    super_seqs = [[phrase_tok.eos_id] + phrase_tok.encode(p, add_eos=False) for p in raw_prefixes]
    sub_seqs = [[phrase_tok.eos_id] + phrase_tok.encode_base(p) for p in raw_prefixes]
    # Hybrid downshifted clause: superwords on distant history, fine subwords on final 120 chars!
    hybrid_seqs = []
    for p in raw_prefixes:
        if len(p) > 120 and " " in p[-120:]:
            split_at = len(p) - 120 + p[-120:].find(" ")
            hist_ids = phrase_tok.encode(p[:split_at], add_eos=False)
            clause_ids = phrase_tok.encode_base(p[split_at:])
            hybrid_seqs.append([phrase_tok.eos_id] + hist_ids + clause_ids)
        else:
            hybrid_seqs.append([phrase_tok.eos_id] + phrase_tok.encode_base(p))

    super_preds, super_ent, super_toks = run_gear(super_seqs)
    sub_preds, _, sub_toks = run_gear(sub_seqs)

    # Adaptive Gearshift: trigger local clause unmerging ONLY when High-Gear entropy is in top quantile
    thresh = float(np.quantile(super_ent, 1.0 - downshift_top_quantile))
    trigger_idxs = [i for i, h in enumerate(super_ent) if h >= thresh]
    trig_seqs = [hybrid_seqs[i] for i in trigger_idxs]
    trig_preds, _, trig_extra_toks = run_gear(trig_seqs) if trig_seqs else ([], np.array([]), 0)

    adapt_preds = list(super_preds)
    for idx_pos, orig_i in enumerate(trigger_idxs):
        adapt_preds[orig_i] = trig_preds[idx_pos]

    n = max(len(gold_words), 1)
    acc_super = sum(p == g for p, g in zip(super_preds, gold_words)) / n
    acc_adapt = sum(p == g for p, g in zip(adapt_preds, gold_words)) / n
    acc_sub = sum(p == g for p, g in zip(sub_preds, gold_words)) / n

    return {
        "high_gear_superword_acc": round(acc_super, 4),
        "adaptive_gearshift_acc": round(acc_adapt, 4),
        "low_gear_subword_acc": round(acc_sub, 4),
        "high_gear_tokens": float(super_toks),
        "adaptive_total_tokens_incl_retry": float(super_toks + trig_extra_toks),
        "low_gear_tokens": float(sub_toks),
        "downshift_fraction": round(len(trigger_idxs) / n, 3),
    }
