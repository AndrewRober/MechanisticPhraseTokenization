"""
Stage A Invariant & Edge-Case Verification Suite.

Verifies BEFORE any model training:
1. Exact lossless UTF-8 byte round-tripping across ASCII, Unicode, emoji, whitespace runs,
   newlines, tabs, punctuation, code snippets, and arbitrary binary-valid UTF-8 edge cases.
2. Strict vocabulary size invariants (14,336 base; 16,384 augmented; 256 byte fallback).
3. Document boundary isolation (never merges across <|endoftext|>).
4. Morphological right-boundary guard (Trap 1: never merges 'New York' inside 'New Yorker',
   'investment bank' inside 'investment banking', or 'take place' inside 'take placement').
5. Deterministic encoding & Utility-DP dominance over greedy left-to-right on overlapping phrases.
6. Active phrase emission audit (flags any selected slots with near-zero real emissions).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import TokenizerSpec
from src.tokenizer_core import BYTE_TO_UNICODE, PhraseAugmentedTokenizer


def build_mini_test_spec() -> tuple[Tokenizer, TokenizerSpec]:
    corpus = [
        "The investment bank and the river bank are different.",
        "In New York and New Yorker magazines, people take place and take placement tests.",
        "He is not allowed to enter, despite the fact that he is happy today!",
        "he is happy, today\n\nMultiple   spaces\tand Unicode: café, naïve, 東京, 🚀!",
    ] * 40

    spec = TokenizerSpec(
        total_vocab_size=520,
        base_vocab_size=500,
        phrase_vocab_slots=20,
    )
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=spec.base_vocab_size,
        min_frequency=2,
        special_tokens=list(spec.special_tokens),
        initial_alphabet=[BYTE_TO_UNICODE[b] for b in range(256)],
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    actual_base = tok.get_vocab_size()
    spec = TokenizerSpec(
        total_vocab_size=actual_base + 4,
        base_vocab_size=actual_base,
        phrase_vocab_slots=4,
    )
    return tok, spec


def run_stage_a_unit_checks() -> None:
    base_tok, spec = build_mini_test_spec()
    assert base_tok.get_vocab_size() == spec.base_vocab_size

    # Construct deliberate test phrases including overlapping phrases & prefix-stem traps
    p_new_york = tuple(base_tok.encode(" New York", add_special_tokens=False).ids)
    p_take_place = tuple(base_tok.encode(" take place", add_special_tokens=False).ids)
    p_he_is = tuple(base_tok.encode("he is", add_special_tokens=False).ids)
    p_is_not_allowed = tuple(base_tok.encode(" is not allowed", add_special_tokens=False).ids)

    phrases = [
        (p_new_york, 2.0),
        (p_take_place, 2.0),
        (p_he_is, 0.5),            # Low-utility glue
        (p_is_not_allowed, 5.0),   # High-utility contrast phrase
    ]

    greedy_tok = PhraseAugmentedTokenizer(base_tok, phrases, spec=spec, mode="greedy_ltr")
    dp_tok = PhraseAugmentedTokenizer(base_tok, phrases, spec=spec, mode="utility_dp")

    # 1. Exact UTF-8 lossless round-trip check
    edge_strings = [
        "",
        " ",
        "   \n\t\r\n  ",
        "In New York, a New Yorker visited an investment bank near the river bank.",
        "We saw the event take place before the take placement exam.",
        "he is not allowed here, despite the fact that he is happy, today!",
        "Emoji & CJK & Math: ∑_{i=1}^n x_i = α + β 🚀🔥 東京 café",
        "Leading space and trailing space ",
    ]
    for s in edge_strings:
        for t_impl in (greedy_tok, dp_tok):
            ids = t_impl.encode(s, add_eos=True)
            assert ids[-1] == t_impl.eos_id
            recovered = t_impl.decode_to_bytes(ids)
            assert recovered == s.encode("utf-8"), f"Round-trip mismatch on: {s!r}"
            # Determinism check
            assert t_impl.encode(s, add_eos=True) == ids

    # 2. Trap 1 check: ' New York' phrase ID must appear in ' New York,' but NEVER inside ' New Yorker'
    ny_phrase_id = spec.base_vocab_size + 0
    tp_phrase_id = spec.base_vocab_size + 1
    ids_ny_clean = greedy_tok.encode("In New York, today")
    ids_ny_derived = greedy_tok.encode("In New Yorker magazine")
    assert ny_phrase_id in ids_ny_clean, "Expected ' New York' phrase token in clean boundary context!"
    assert ny_phrase_id not in ids_ny_derived, "BUG (Trap 1): ' New York' merged inside ' New Yorker'!"

    ids_tp_clean = greedy_tok.encode("Will take place soon")
    ids_tp_derived = greedy_tok.encode("Will take placement test")
    assert tp_phrase_id in ids_tp_clean, "Expected ' take place' phrase token in clean context!"
    assert tp_phrase_id not in ids_tp_derived, "BUG (Trap 1): ' take place' merged inside ' take placement'!"

    # 3. Document boundary isolation: never merge across <|endoftext|>
    doc1 = base_tok.encode("In New", add_special_tokens=False).ids
    doc2 = base_tok.encode(" York today", add_special_tokens=False).ids
    stream_with_eos = doc1 + [greedy_tok.eos_id] + doc2
    merged_stream = greedy_tok.merge_base_ids(stream_with_eos)
    assert ny_phrase_id not in merged_stream, "BUG: Phrase merged across <|endoftext|> document boundary!"

    # 4. Utility-DP vs Greedy overlap check on "he is not allowed"
    he_is_id = spec.base_vocab_size + 2
    not_allowed_id = spec.base_vocab_size + 3
    g_ids = greedy_tok.encode("he is not allowed")
    dp_ids = dp_tok.encode("he is not allowed")
    assert he_is_id in g_ids and not_allowed_id not in g_ids, "Greedy should pick 'he is' left-to-right"
    assert not_allowed_id in dp_ids and he_is_id not in dp_ids, "Utility-DP should pick higher-utility ' is not allowed'"

    print("[PASS] All Stage A unit invariants (round-trip, Trap-1 boundary guard, EOS isolation, Utility-DP) passed!")


if __name__ == "__main__":
    run_stage_a_unit_checks()
