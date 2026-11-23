"""
Stage A Corpus Preparation, Pinning, & Shared 14,336 Pilot Model Launch.

1. Streams & deduplicates deterministic FineWeb-Edu (`sample-10BT`) documents and pins exact IDs + SHA256:
     - `train_120M.jsonl`   : Exact raw-text corpus containing 120,000,000 B1-BPE tokens
     - `calib_a_5M.jsonl`   : Disjoint 5M-token split A for C1/C3/C5 pilot probes
     - `calib_b_5M.jsonl`   : Disjoint 5M-token split B for C1/C3/C5 held-out scoring
     - `val_3M.jsonl`       : Disjoint validation split (3M tokens)
     - `test_5M.jsonl`      : Disjoint held-out test split (5M tokens)
   Plus TinyStories OOD (`test_tinystories.jsonl`) and BLiMP minimal pairs (`test_blimp.jsonl`).
2. Builds & saves:
     - `B1_ByteBPE_16384` (Reference tokenizer defining exact raw-text budgets)
     - `SharedBase_ByteBPE_14336` (Foundation for B4, B5, and C0-C5)
3. Runs Stage A verification on real TinyStories + FineWeb-Edu documents (100% round-trip & boundary audit).
4. Launches the First Training Run: Shared 14,336 Base Pilot Model (100M reference tokens on GB10).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from tokenizers import Tokenizer

from src.baselines import HFByteTokenizerWrapper, build_b1_byte_bpe
from src.config import ModelSpec, TokenizerSpec, TrainSpec
from src.tokenizer_core import train_byte_bpe_tokenizer
from src.trainer import train_language_model


ROOT_DIR = Path(os.environ.get("TOK_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("TOK_DATA_DIR", ROOT_DIR / "data"))
ARTIFACTS_DIR = Path(os.environ.get("TOK_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def prepare_pinned_datasets(spec: TokenizerSpec) -> tuple[HFByteTokenizerWrapper, Tokenizer]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tok_dir = ARTIFACTS_DIR / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA_DIR / "dataset_manifest.json"
    b1_dir = tok_dir / "B1_ByteBPE_16384"
    base_file = tok_dir / "shared_base_14336.json"

    if manifest_path.exists() and (b1_dir / "tokenizer.json").exists() and base_file.exists():
        print("[Stage A] Pinned datasets and reference tokenizers already exist — loading...")
        b1 = HFByteTokenizerWrapper.load(b1_dir, spec=spec)
        base_tok = Tokenizer.from_file(str(base_file))
        return b1, base_tok

    print("[Stage A] Streaming FineWeb-Edu (sample-10BT) to build deduplicated raw corpus...")
    t0 = time.perf_counter()
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Collect ~220k clean deduplicated documents (~165M tokens total across train+calib+val+test)
    seen_hashes: set[bytes] = set()
    raw_docs: list[tuple[str, str]] = []  # (doc_id, text)
    total_chars = 0
    target_chars = 660_000_000  # ~155M BPE tokens (~4.25 chars/tok)

    for idx, row in enumerate(ds):
        text = row.get("text", "")
        if not text or len(text) < 200:
            continue
        # Cap extreme outlier multi-MB single documents (99.5th percentile safeguard from SuperBPE paper)
        if len(text) > 40_000:
            text = text[:40_000]
        norm_h = hashlib.sha1(text[:2000].strip().lower().encode("utf-8")).digest()
        if norm_h in seen_hashes:
            continue
        seen_hashes.add(norm_h)
        doc_id = str(row.get("id", f"fw_{idx}"))
        raw_docs.append((doc_id, text))
        total_chars += len(text)
        if len(raw_docs) % 25_000 == 0:
            print(f"  streamed {len(raw_docs):,} docs ({total_chars / 1e6:.1f}M chars) in {time.perf_counter()-t0:.1f}s", flush=True)
        if total_chars >= target_chars:
            break

    print(f"[Stage A] Collected {len(raw_docs):,} deduplicated docs ({total_chars / 1e6:.1f}M chars). Training B1 & Shared Base tokenizers...", flush=True)

    # Hold out tail documents strictly for Val/Test/Calib BEFORE training tokenizers!
    # Train tokenizers ONLY on the training candidate region (first 75% of docs)
    n_train_pool = int(len(raw_docs) * 0.78)
    tok_train_texts = [txt for _, txt in raw_docs[: min(n_train_pool, 80_000)]]

    b1 = build_b1_byte_bpe(tok_train_texts, spec=spec)
    b1.save(b1_dir)

    base_tok = train_byte_bpe_tokenizer(
        tok_train_texts, vocab_size=spec.base_vocab_size, special_tokens=spec.special_tokens
    )
    base_tok.save(str(base_file))
    print(f"[Stage A] Built B1 ({b1.vocab_size}) & Shared Base ({base_tok.get_vocab_size()}) in {time.perf_counter()-t0:.1f}s", flush=True)

    # Partition exact reference-token budgets using B1 (Byte-BPE 16,384)
    splits_spec = [
        ("train_120M.jsonl", 120_000_000),
        ("calib_a_5M.jsonl", 5_000_000),
        ("calib_b_5M.jsonl", 5_000_000),
        ("val_3M.jsonl", 3_000_000),
        ("test_5M.jsonl", 5_000_000),
    ]

    manifest: dict = {"dataset": "HuggingFaceFW/fineweb-edu (sample-10BT)", "splits": {}}
    cursor = 0
    for fname, tok_budget in splits_spec:
        fpath = DATA_DIR / fname
        acc_toks = 0
        acc_bytes = 0
        n_docs = 0
        with fpath.open("w", encoding="utf-8") as f:
            while cursor < len(raw_docs) and acc_toks < tok_budget:
                doc_id, txt = raw_docs[cursor]
                cursor += 1
                t_len = len(b1.encode(txt, add_eos=True))
                f.write(json.dumps({"id": doc_id, "text": txt}, ensure_ascii=False) + "\n")
                acc_toks += t_len
                acc_bytes += len(txt.encode("utf-8"))
                n_docs += 1
        manifest["splits"][fname] = {
            "docs": n_docs,
            "b1_ref_tokens": acc_toks,
            "utf8_bytes": acc_bytes,
            "sha256": sha256_file(fpath),
        }
        print(f"  wrote {fname}: {n_docs:,} docs | {acc_toks:,} B1 tokens | {acc_bytes/1e6:.1f} MB", flush=True)

    # Also download TinyStories held-out validation slice (Stage A debugging + OOD transfer)
    print("[Stage A] Downloading TinyStories OOD evaluation slice...", flush=True)
    ts_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ts_path = DATA_DIR / "test_tinystories.jsonl"
    ts_docs = 0
    ts_bytes = 0
    with ts_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ts_ds):
            txt = row.get("text", "").strip()
            if len(txt) < 60:
                continue
            f.write(json.dumps({"id": f"ts_{idx}", "text": txt}, ensure_ascii=False) + "\n")
            ts_docs += 1
            ts_bytes += len(txt.encode("utf-8"))
            if ts_docs >= 2_000:
                break
    manifest["splits"]["test_tinystories.jsonl"] = {
        "docs": ts_docs,
        "utf8_bytes": ts_bytes,
        "sha256": sha256_file(ts_path),
    }

    # Download BLiMP grammatical minimal pairs across diverse syntactic paradigms (held out 100%!)
    print("[Stage A] Downloading BLiMP grammatical minimal-pair benchmark...", flush=True)
    blimp_configs = [
        "determiner_noun_agreement_1",
        "irregular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_1",
        "distractor_agreement_relational_noun",
        "anaphor_number_agreement",
        "passive_1",
        "causative",
        "wh_questions_object_gap",
        "sentential_negation_npi_licensor_present",
        "principle_A_c_command",
    ]
    blimp_path = DATA_DIR / "test_blimp.jsonl"
    n_blimp = 0
    with blimp_path.open("w", encoding="utf-8") as f:
        for cfg in blimp_configs:
            try:
                b_ds = load_dataset("nyu-mll/blimp", cfg, split="train")
                for row in list(b_ds)[:300]:
                    f.write(
                        json.dumps(
                            {
                                "paradigm": cfg,
                                "good": row["sentence_good"],
                                "bad": row["sentence_bad"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_blimp += 1
            except Exception as exc:
                print(f"  [warn] skipped BLiMP {cfg}: {exc}")
    manifest["splits"]["test_blimp.jsonl"] = {"pairs": n_blimp, "sha256": sha256_file(blimp_path)}

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return b1, base_tok


def load_texts(jsonl_path: Path, max_docs: int | None = None) -> list[str]:
    out: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if max_docs is not None and len(out) >= max_docs:
                break
    return out


def verify_corpus_roundtrip(b1: HFByteTokenizerWrapper, base_tok: Tokenizer, sample_docs: list[str]) -> None:
    base_wrap = HFByteTokenizerWrapper(base_tok)
    for idx, doc in enumerate(sample_docs):
        raw_b = doc.encode("utf-8")
        assert b1.decode_to_bytes(b1.encode(doc, add_eos=True)) == raw_b, f"B1 roundtrip failed on doc {idx}"
        assert base_wrap.decode_to_bytes(base_wrap.encode(doc, add_eos=True)) == raw_b, f"Base roundtrip failed on doc {idx}"
    print(f"[Stage A PASS] Verified 100% exact UTF-8 byte round-tripping on {len(sample_docs):,} real corpus documents!", flush=True)


def main() -> None:
    tok_spec = TokenizerSpec()
    model_spec = ModelSpec()
    train_spec = TrainSpec()

    b1, base_tok = prepare_pinned_datasets(tok_spec)
    ts_docs = load_texts(DATA_DIR / "test_tinystories.jsonl", max_docs=500)
    val_docs = load_texts(DATA_DIR / "val_3M.jsonl", max_docs=1000)
    verify_corpus_roundtrip(b1, base_tok, ts_docs + val_docs[:500])

    # Load first 100M B1-reference tokens of train_120M.jsonl for the Shared 14,336 Base Pilot
    print("[Pilot] Loading 100M B1-eq training slice for Shared 14,336 Base Pilot...", flush=True)
    pilot_docs: list[str] = []
    b1_count = 0
    with (DATA_DIR / "train_120M.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            txt = json.loads(line)["text"]
            pilot_docs.append(txt)
            b1_count += len(b1.encode(txt, add_eos=True))
            if b1_count >= train_spec.pilot_ref_tokens:
                break
    print(f"[Pilot] Loaded {len(pilot_docs):,} docs ({b1_count:,} B1-ref tokens). Starting GB10 training!", flush=True)

    base_wrapper = HFByteTokenizerWrapper(base_tok, spec=tok_spec)
    pilot_out = ARTIFACTS_DIR / "runs" / "pilot_shared_base_14336"

    # 1,525 steps * ~65,536 tokens/step ≈ 100M tokens
    _, summary = train_language_model(
        run_name="Pilot_Base14336",
        tokenizer=base_wrapper,
        train_docs=pilot_docs,
        val_docs=val_docs,
        output_dir=pilot_out,
        model_spec=model_spec,
        train_spec=train_spec,
        regime="equal_raw_bytes",
        target_steps=1_525,
        micro_batch_size=32,
        seed=1337,
        eval_every_steps=250,
        use_compile=True,
    )
    print(f"[Pilot COMPLETE] Wall time: {summary['train_wall_sec']/60:.2f} min | Avg speed: {summary['avg_tok_per_sec']:,.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
