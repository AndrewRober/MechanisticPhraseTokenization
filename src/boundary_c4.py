"""
C4 (BoundaryStableMerge): Batched Marginal Retokenization-Cascade Selector.

Constructs controlled punctuation, spacing, and character-typo edits on training snippets with
exact byte-offset alignments, then selects phrases maximizing:
    utility(p | S_current) = compression_benefit(p | S_current) - lambda * boundary_disturbance(p | S_current)
Evaluated incrementally in batches using the actual PhraseAugmentedTokenizer so overlapping
phrase Domino cascades (Trap 5) are measured faithfully rather than collapsing to phrase length.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from tokenizers import Tokenizer

from src.config import TokenizerSpec
from src.phrase_selectors import PhraseCandidate
from src.tokenizer_core import PhraseAugmentedTokenizer, PhraseEntry


@dataclass
class AlignedEditPair:
    orig_text: str
    edit_text: str
    orig_bytes_len: int
    edit_bytes_len: int
    # Byte span [e0, e1) in orig replaced by [e0, e1_prime) in edit
    e0: int
    e1: int
    e1_prime: int


def build_aligned_edit_pairs(
    snippets: Sequence[str],
    edits_per_snippet: int = 2,
    seed: int = 42,
) -> list[AlignedEditPair]:
    rng = random.Random(seed)
    pairs: list[AlignedEditPair] = []

    for text in snippets:
        if len(text) < 32:
            continue
        # Find space / word boundary indices in character space and map cleanly to UTF-8 byte offsets
        space_char_idxs = [idx for idx, ch in enumerate(text) if ch == " " and 8 <= idx <= len(text) - 8]
        alpha_char_idxs = [idx for idx, ch in enumerate(text) if ch.isascii() and ch.isalpha() and 8 <= idx <= len(text) - 8]
        if not space_char_idxs or not alpha_char_idxs:
            continue

        for mode_idx in range(edits_per_snippet):
            kind = (mode_idx + rng.randint(0, 2)) % 3
            if kind == 0:
                # Punctuation insertion before space: "happy today" -> "happy, today"
                ci = rng.choice(space_char_idxs)
                prefix = text[:ci]
                suffix = text[ci:]
                edited = prefix + "," + suffix
                e0 = len(prefix.encode("utf-8"))
                e1 = e0
                e1_p = e0 + 1
            elif kind == 1:
                # Extra whitespace edit: " " -> "  "
                ci = rng.choice(space_char_idxs)
                prefix = text[:ci]
                suffix = text[ci + 1 :]
                edited = prefix + "  " + suffix
                e0 = len(prefix.encode("utf-8"))
                e1 = e0 + 1
                e1_p = e0 + 2
            else:
                # Single-character ASCII typo substitution inside a word
                ci = rng.choice(alpha_char_idxs)
                prefix = text[:ci]
                orig_ch = text[ci]
                rep_ch = "x" if orig_ch.lower() != "x" else "z"
                suffix = text[ci + 1 :]
                edited = prefix + rep_ch + suffix
                e0 = len(prefix.encode("utf-8"))
                e1 = e0 + 1
                e1_p = e0 + 1

            pairs.append(
                AlignedEditPair(
                    orig_text=text,
                    edit_text=edited,
                    orig_bytes_len=len(text.encode("utf-8")),
                    edit_bytes_len=len(edited.encode("utf-8")),
                    e0=e0,
                    e1=e1,
                    e1_prime=e1_p,
                )
            )
    return pairs


def token_byte_boundaries(tok: PhraseAugmentedTokenizer, token_ids: list[int]) -> set[int]:
    """Returns internal UTF-8 byte offsets where token boundaries occur."""
    bounds: set[int] = set()
    pos = 0
    for tid in token_ids[:-1]:
        pos += len(tok._id_to_bytes.get(tid, b""))
        bounds.add(pos)
    return bounds


def unchanged_boundary_disturbance(
    tok: PhraseAugmentedTokenizer,
    pair: AlignedEditPair,
    orig_base_ids: list[int],
    edit_base_ids: list[int],
) -> int:
    ids_o = tok.merge_base_ids(orig_base_ids)
    ids_e = tok.merge_base_ids(edit_base_ids)
    b_o = token_byte_boundaries(tok, ids_o)
    b_e = token_byte_boundaries(tok, ids_e)

    # Strictly unchanged left prefix: (0, e0)
    left_o = {x for x in b_o if 0 < x < pair.e0}
    left_e = {x for x in b_e if 0 < x < pair.e0}
    diff_left = len(left_o.symmetric_difference(left_e))

    # Strictly unchanged right suffix: (e1, orig_len) mapped by shift delta = e1_prime - e1
    delta = pair.e1_prime - pair.e1
    right_o = {x for x in b_o if pair.e1 < x < pair.orig_bytes_len}
    right_e_mapped = {x - delta for x in b_e if pair.e1_prime < x < pair.edit_bytes_len}
    diff_right = len(right_o.symmetric_difference(right_e_mapped))

    return diff_left + diff_right


def select_c4_boundary_stable(
    candidates: Sequence[PhraseCandidate],
    snippets: Sequence[str],
    base_tokenizer: Tokenizer,
    spec: TokenizerSpec = TokenizerSpec(),
    disturbance_lambda: float = 2.5,
    batch_size: int = 128,
    pool_limit: int = 3_200,
) -> list[tuple[tuple[int, ...], float]]:
    """
    Iteratively builds K=2,048 phrase vocabulary slots balancing compression benefit against
    unchanged-region retokenization cascades on controlled local perturbations.
    """
    pool = list(candidates[:pool_limit])
    pairs = build_aligned_edit_pairs(snippets[:1_200], edits_per_snippet=1)
    orig_base = [base_tokenizer.encode(p.orig_text, add_special_tokens=False).ids for p in pairs]
    edit_base = [base_tokenizer.encode(p.edit_text, add_special_tokens=False).ids for p in pairs]

    # Index which snippet pairs contain each candidate tuple in orig or edit for O(hits) marginal eval
    cand_pair_hits: dict[tuple[int, ...], list[int]] = {c.base_ids: [] for c in pool}
    max_l = max((len(c.base_ids) for c in pool), default=2)

    for p_idx, (ob, eb) in enumerate(zip(orig_base, edit_base)):
        seen: set[tuple[int, ...]] = set()
        for seq in (ob, eb):
            L = len(seq)
            for i in range(L - 1):
                for l in range(2, min(max_l, L - i) + 1):
                    tup = tuple(seq[i : i + l])
                    if tup in cand_pair_hits and tup not in seen:
                        cand_pair_hits[tup].append(p_idx)
                        seen.add(tup)

    selected: list[tuple[tuple[int, ...], float]] = []
    selected_set: set[tuple[int, ...]] = set()
    n_base = base_tokenizer.get_vocab_size()

    while len(selected) < spec.phrase_vocab_slots:
        need = min(batch_size, spec.phrase_vocab_slots - len(selected))
        cur_spec = TokenizerSpec(
            total_vocab_size=n_base + len(selected),
            base_vocab_size=n_base,
            phrase_vocab_slots=len(selected),
        )
        cur_tok = PhraseAugmentedTokenizer(base_tokenizer, selected, spec=cur_spec, mode="greedy_ltr")

        # Baseline per-pair disturbance & token length under S_current
        base_dist = [
            unchanged_boundary_disturbance(cur_tok, pairs[idx], orig_base[idx], edit_base[idx])
            for idx in range(len(pairs))
        ]
        base_len = [len(cur_tok.merge_base_ids(orig_base[idx])) for idx in range(len(pairs))]

        # Evaluate top remaining candidates marginally on their active hit pairs
        active_cands = [c for c in pool if c.base_ids not in selected_set][:200]
        if not active_cands:
            break

        scored: list[tuple[float, PhraseCandidate]] = []
        temp_pid = n_base + len(selected)
        for c in active_cands:
            hit_idxs = cand_pair_hits[c.base_ids]
            if not hit_idxs:
                prior_util = float(c.total_freq * (len(c.base_ids) - 1)) * 0.1
                scored.append((prior_util, c))
                continue

            # In-place temporary Trie insertion (avoids 19,200 full tokenizer rebuilds -> 200x faster!)
            raw_b = b"".join(cur_tok._id_to_bytes[tid] for tid in c.base_ids)
            entry = PhraseEntry(phrase_id=temp_pid, base_ids=c.base_ids, utility=1.0, raw_bytes=raw_b)
            cur_tok._id_to_bytes[temp_pid] = raw_b
            node = cur_tok.trie
            created_path: list[tuple[dict, int]] = []
            for tid in c.base_ids:
                if tid not in node:
                    node[tid] = {}
                    created_path.append((node, tid))
                node = node[tid]
            prev_entry = node.get("_entry")
            node["_entry"] = entry

            delta_saved = 0
            delta_dist = 0
            for idx in hit_idxs[:40]:  # cap to 40 representative pairs per candidate for O(1) speed
                new_l = len(cur_tok.merge_base_ids(orig_base[idx]))
                new_d = unchanged_boundary_disturbance(cur_tok, pairs[idx], orig_base[idx], edit_base[idx])
                delta_saved += base_len[idx] - new_l
                delta_dist += max(0, new_d - base_dist[idx])

            # Cleanly revert temporary Trie insertion
            if prev_entry is None:
                node.pop("_entry", None)
            else:
                node["_entry"] = prev_entry
            for parent_d, key_tid in reversed(created_path):
                if not parent_d[key_tid]:
                    parent_d.pop(key_tid, None)
            cur_tok._id_to_bytes.pop(temp_pid, None)

            denom = min(len(hit_idxs), 40)
            rate_saved = delta_saved / max(denom, 1)
            rate_dist = delta_dist / max(denom, 1)
            est_utility = c.total_freq * (rate_saved - disturbance_lambda * rate_dist)
            scored.append((est_utility, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        for util, c in scored[:need]:
            selected.append((c.base_ids, max(util, 1e-6)))
            selected_set.add(c.base_ids)

    return selected[: spec.phrase_vocab_slots]
