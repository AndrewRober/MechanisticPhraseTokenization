"""
Shared Tokenizer Core & Stage A Verification Suite.

Implements:
1. Exact lossless Byte-level BPE base vocabulary (14,336 entries: 2 special + 256 raw bytes + merges).
2. Multi-word phrase vocabulary extension (+2,048 phrase entries -> 16,384 total).
3. Word-boundary-guarded segmentation (fixes Trap 1: prevents matching '" New", " York"' inside '" New", " York", "er"').
4. Both Deterministic Left-to-Right Longest-Match Segmentation (controlled baseline for C0-C5)
   and O(N) Utility-Optimal Dynamic Programming Segmentation (Viterbi-Utility upgrade).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import TokenizerSpec

# GPT-2 byte-to-unicode reversible mapping (ensures every byte 0..255 is a printable char)
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
SPACE_MARKER = BYTE_TO_UNICODE[ord(" ")]  # 'Ġ'


def token_str_to_bytes(tok_str: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[ch] for ch in tok_str)


def is_word_start_token(tok_str: str, is_doc_start: bool = False) -> bool:
    """
    Returns True if a base Byte-BPE token starts a fresh pre-token boundary
    (begins with whitespace marker 'Ġ', newline/tab byte, or is at index 0 of document).
    """
    if is_doc_start:
        return True
    if not tok_str:
        return False
    first_byte = UNICODE_TO_BYTE.get(tok_str[0], -1)
    if first_byte in (ord(" "), ord("\n"), ord("\r"), ord("\t")):
        return True
    return False


def is_continuation_alpha_subword(tok_str: str) -> bool:
    """
    Returns True if tok_str is an intra-word continuation fragment (no leading whitespace,
    starts with an ASCII letter/digit or UTF-8 continuation byte), e.g. 'ing', 'er', 'ment', 's'.
    If the token immediately following a candidate phrase match is a continuation subword,
    merging the phrase would strand a suffix inside a single word (Trap 1).
    """
    if not tok_str:
        return False
    first_byte = UNICODE_TO_BYTE.get(tok_str[0], -1)
    if first_byte < 0:
        return False
    # ASCII alphanumeric without leading space, or UTF-8 multi-byte continuation (0x80..0xBF)
    if (ord("a") <= first_byte <= ord("z")) or (ord("A") <= first_byte <= ord("Z")) or (ord("0") <= first_byte <= ord("9")):
        return True
    if 0x80 <= first_byte <= 0xBF:
        return True
    return False


def train_byte_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    special_tokens: tuple[str, ...] = ("<|endoftext|>", "<|pad|>"),
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Trains a strictly lossless Byte-level BPE tokenizer (used for both B1 @ 16,384
    and the shared Base Tokenizer @ 14,336).
    """
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    initial_alphabet = [BYTE_TO_UNICODE[b] for b in range(256)]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=initial_alphabet,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    assert tok.get_vocab_size() == vocab_size, f"Expected vocab {vocab_size}, got {tok.get_vocab_size()}"
    return tok


@dataclass
class PhraseEntry:
    phrase_id: int
    base_ids: tuple[int, ...]
    utility: float
    raw_bytes: bytes


class PhraseAugmentedTokenizer:
    """
    Deterministic fixed-vocabulary tokenizer combining:
      - Shared 14,336-entry Byte-BPE base tokenizer
      - Exactly K (2,048) multi-token phrase entries (IDs 14,336 .. 16,383)
    Enforces:
      - Exact lossless UTF-8 byte round-tripping
      - Document/special-token boundary isolation (never merges across <|endoftext|>)
      - Right-hand morphological boundary guard (never strands suffixes like 'New York' + 'er')
      - Choice of 'greedy_ltr' (canonical Stage B baseline) or 'utility_dp' (O(N) optimal interval DP)
    """

    def __init__(
        self,
        base_tokenizer: Tokenizer,
        phrases: list[tuple[tuple[int, ...], float]],
        spec: TokenizerSpec = TokenizerSpec(),
        mode: str = "greedy_ltr",
    ):
        assert mode in ("greedy_ltr", "utility_dp")
        self.base = base_tokenizer
        self.spec = spec
        self.mode = mode

        base_vocab = self.base.get_vocab()
        self.id_to_tok: dict[int, str] = {idx: s for s, idx in base_vocab.items()}
        self.special_ids: set[int] = {base_vocab[s] for s in spec.special_tokens if s in base_vocab}
        self.eos_id: int = base_vocab[spec.special_tokens[0]]
        self.pad_id: int = base_vocab[spec.special_tokens[1]]

        # Precompute token properties for O(1) boundary checks
        base_n = self.base.get_vocab_size()
        assert base_n == spec.base_vocab_size, f"Base vocab {base_n} != {spec.base_vocab_size}"
        self._is_continuation = [False] * base_n
        self._is_word_start = [False] * base_n
        self._id_to_bytes: dict[int, bytes] = {}

        for tid in range(base_n):
            if tid in self.special_ids:
                self._id_to_bytes[tid] = b""
                continue
            tstr = self.id_to_tok[tid]
            self._id_to_bytes[tid] = token_str_to_bytes(tstr)
            self._is_continuation[tid] = is_continuation_alpha_subword(tstr)
            self._is_word_start[tid] = is_word_start_token(tstr, is_doc_start=False)

        # Build Trie over base_ids for fast O(N * L_max) matching
        self.phrases: dict[int, PhraseEntry] = {}
        self.trie: dict[int, dict] = {}
        next_id = spec.base_vocab_size

        for base_tuple, utility in phrases:
            assert len(base_tuple) >= 2, "Phrase must span at least 2 base tokens"
            assert all(tid not in self.special_ids for tid in base_tuple), "Phrase cannot contain special tokens"
            raw_b = b"".join(self._id_to_bytes[tid] for tid in base_tuple)
            entry = PhraseEntry(phrase_id=next_id, base_ids=tuple(base_tuple), utility=float(utility), raw_bytes=raw_b)
            self.phrases[next_id] = entry
            self._id_to_bytes[next_id] = raw_b

            node = self.trie
            for tid in base_tuple:
                node = node.setdefault(tid, {})
            node["_entry"] = entry
            next_id += 1

        self.vocab_size = next_id

    def _valid_span_boundary(self, base_ids: list[int], start: int, end: int) -> bool:
        """Checks left and right morphological word boundaries so suffixes are never stranded."""
        first_tid = base_ids[start]
        if start > 0 and base_ids[start - 1] not in self.special_ids:
            if not self._is_word_start[first_tid]:
                return False
        if end < len(base_ids):
            next_tid = base_ids[end]
            if next_tid not in self.special_ids and self._is_continuation[next_tid]:
                return False
        return True

    def encode_base(self, text: str) -> list[int]:
        return self.base.encode(text, add_special_tokens=False).ids

    def merge_base_ids(self, base_ids: list[int]) -> list[int]:
        n = len(base_ids)
        if n < 2 or not self.phrases:
            return base_ids

        if self.mode == "greedy_ltr":
            out: list[int] = []
            i = 0
            while i < n:
                tid = base_ids[i]
                if tid in self.special_ids or tid not in self.trie:
                    out.append(tid)
                    i += 1
                    continue

                # Walk trie to find longest valid boundary-safe match starting at i
                node = self.trie
                best_entry: Optional[PhraseEntry] = None
                best_end = i + 1
                j = i
                while j < n and base_ids[j] in node:
                    node = node[base_ids[j]]
                    j += 1
                    entry = node.get("_entry")
                    if entry is not None and self._valid_span_boundary(base_ids, i, j):
                        best_entry = entry
                        best_end = j

                if best_entry is not None:
                    out.append(best_entry.phrase_id)
                    i = best_end
                else:
                    out.append(tid)
                    i += 1
            return out

        # mode == "utility_dp": O(N) Weighted Interval Scheduling DP
        dp_val = [0.0] * (n + 1)
        dp_next = [0] * n
        dp_emit = [0] * n

        for i in range(n - 1, -1, -1):
            # Default: emit single base token base_ids[i]
            best_v = dp_val[i + 1]
            best_nxt = i + 1
            best_tok = base_ids[i]

            tid = base_ids[i]
            if tid not in self.special_ids and tid in self.trie:
                node = self.trie
                j = i
                while j < n and base_ids[j] in node:
                    node = node[base_ids[j]]
                    j += 1
                    entry = node.get("_entry")
                    if entry is not None and self._valid_span_boundary(base_ids, i, j):
                        cand_v = max(entry.utility, 1e-9) + dp_val[j]
                        if cand_v > best_v:
                            best_v = cand_v
                            best_nxt = j
                            best_tok = entry.phrase_id

            dp_val[i] = best_v
            dp_next[i] = best_nxt
            dp_emit[i] = best_tok

        out = []
        idx = 0
        while idx < n:
            out.append(dp_emit[idx])
            idx = dp_next[idx]
        return out

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self.merge_base_ids(self.encode_base(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode_to_bytes(self, token_ids: Iterable[int]) -> bytes:
        chunks = [self._id_to_bytes[tid] for tid in token_ids if tid not in self.special_ids]
        return b"".join(chunks)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.decode_to_bytes(token_ids).decode("utf-8", errors="replace")

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.base.save(str(d / "base_tokenizer.json"))
        meta = {
            "mode": self.mode,
            "vocab_size": self.vocab_size,
            "phrases": [
                {"phrase_id": e.phrase_id, "base_ids": list(e.base_ids), "utility": e.utility}
                for e in sorted(self.phrases.values(), key=lambda x: x.phrase_id)
            ],
        }
        (d / "phrases.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, spec: TokenizerSpec = TokenizerSpec()) -> "PhraseAugmentedTokenizer":
        d = Path(directory)
        base = Tokenizer.from_file(str(d / "base_tokenizer.json"))
        meta = json.loads((d / "phrases.json").read_text(encoding="utf-8"))
        phrases = [(tuple(item["base_ids"]), float(item["utility"])) for item in meta["phrases"]]
        return cls(base_tokenizer=base, phrases=phrases, spec=spec, mode=meta["mode"])
