"""
Parallelized Stage B 11-Method Screening Orchestrator (B1-B5 + C0-C5).

Maximizes GX10 (20-core Grace CPU + Blackwell GB10 GPU) hardware utilization:
  1. Overlaps CPU tokenizer construction (B2, B3, B4, B5, C0, C2, C4 across parallel CPU workers)
     WITH GPU pilot-probe vocabulary construction (C1 Ward Conflict + C3 Spillover + C5 Contrast)!
  2. Pre-packs all 11 tokenized 120M training block arrays in parallel across CPU cores (`0s` GPU wait).
  3. Runs a live 1-worker vs 2-parallel-worker GPU throughput duel on GB10:
     automatically runs 2 models in parallel if multi-process GPU scaling > 1.08x, or 1 fused
     152k+ tok/s worker if GB10 SMs/bandwidth (95%+ utilized) are faster without L2 contention.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tokenizers import Tokenizer

from src.baselines import (
    HFByteTokenizerWrapper,
    TokenizerInterface,
    build_b2_unigram,
    build_b3_sage,
    build_b4_superbpe,
    build_b5_supratok,
)
from src.boundary_c4 import select_c4_boundary_stable
from src.config import ModelSpec, TokenizerSpec, TrainSpec
from src.eval_suite import (
    compute_unigram_bpb,
    evaluate_bpb_and_stats,
    evaluate_minimal_pairs,
    evaluate_perturbation_stability,
)
from src.model import ResearchTransformerLM
from src.phrase_selectors import (
    PhraseCandidate,
    mine_phrase_candidates,
    select_c0_freq_or_c2_reserve,
    select_c1_conflict_and_vegad_control,
)
from src.probes_c3_c5 import fit_and_score_c3_c5_probes
from src.tokenizer_core import PhraseAugmentedTokenizer
from src.trainer import pack_documents_to_blocks, train_language_model


ROOT_DIR = Path(os.environ.get("TOK_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("TOK_DATA_DIR", ROOT_DIR / "data"))
ARTIFACTS_DIR = Path(os.environ.get("TOK_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))


def load_texts(path: Path, max_docs: int | None = None) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if max_docs is not None and len(docs) >= max_docs:
                break
    return docs


def load_blimp(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not path.exists():
        return pairs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pairs.append((r["good"], r["bad"]))
    return pairs


def load_saved_tokenizer(name: str, spec: TokenizerSpec = TokenizerSpec()) -> TokenizerInterface:
    tok_root = ARTIFACTS_DIR / "tokenizers"
    hf_dirs = {
        "B1_ByteBPE": "B1_ByteBPE_16384",
        "B2_Unigram": "B2_Unigram_16384",
        "B3_SaGe": "B3_SaGe_16384",
    }
    phrase_dirs = {
        "B4_SuperBPE": "B4_SuperBPE_16384",
        "B5_SupraTok": "B5_SupraTok_16384",
        "C0_FreqMerge": "C0_FreqMerge_16384",
        "C1_ConflictMerge": "C1_ConflictMerge_16384",
        "C2_ReserveMerge": "C2_ReserveMerge_16384",
        "C3_SpilloverMerge": "C3_SpilloverMerge_16384",
        "C4_BoundaryStable": "C4_BoundaryStable_16384",
        "C5_ContrastMerge": "C5_ContrastMerge_16384",
    }
    if name in hf_dirs:
        return HFByteTokenizerWrapper.load(tok_root / hf_dirs[name], spec=spec)
    return PhraseAugmentedTokenizer.load(tok_root / phrase_dirs[name], spec=spec)


# --- Top-level CPU worker functions (spawn-safe) ---
def _cpu_build_worker(task_name: str, candidates: list[PhraseCandidate] | None) -> tuple[str, float]:
    spec = TokenizerSpec()
    tok_root = ARTIFACTS_DIR / "tokenizers"
    base_tok = Tokenizer.from_file(str(tok_root / "shared_base_14336.json"))
    train_slice = load_texts(DATA_DIR / "train_120M.jsonl", max_docs=60_000)

    t0 = time.perf_counter()
    if task_name == "B2_Unigram":
        d = tok_root / "B2_Unigram_16384"
        if not (d / "tokenizer.json").exists():
            build_b2_unigram(train_slice, spec=spec).save(d)
    elif task_name == "B3_SaGe":
        d = tok_root / "B3_SaGe_16384"
        if not (d / "tokenizer.json").exists():
            build_b3_sage(train_slice, spec=spec).save(d)
    elif task_name == "B4_SuperBPE":
        d = tok_root / "B4_SuperBPE_16384"
        if not (d / "phrases.json").exists():
            docs_ids = [base_tok.encode(x, add_special_tokens=False).ids for x in train_slice[:35_000]]
            build_b4_superbpe(base_tok, docs_ids, spec=spec).save(d)
    elif task_name == "B5_SupraTok":
        d = tok_root / "B5_SupraTok_16384"
        if not (d / "phrases.json").exists():
            docs_ids = [base_tok.encode(x, add_special_tokens=False).ids for x in train_slice[:35_000]]
            build_b5_supratok(base_tok, docs_ids, spec=spec).save(d)
    elif task_name == "C0_FreqMerge":
        d = tok_root / "C0_FreqMerge_16384"
        if not (d / "phrases.json").exists():
            assert candidates is not None
            sample_ids = [base_tok.encode(x, add_special_tokens=False).ids for x in train_slice[:20_000]]
            ph = select_c0_freq_or_c2_reserve(candidates, sample_ids, base_tok, spec=spec, penalty_lambda=0.0)
            PhraseAugmentedTokenizer(base_tok, ph, spec=spec, mode="greedy_ltr").save(d)
    elif task_name == "C2_ReserveMerge":
        d = tok_root / "C2_ReserveMerge_16384"
        if not (d / "phrases.json").exists():
            assert candidates is not None
            sample_ids = [base_tok.encode(x, add_special_tokens=False).ids for x in train_slice[:20_000]]
            ph = select_c0_freq_or_c2_reserve(candidates, sample_ids, base_tok, spec=spec, penalty_lambda=4.0)
            PhraseAugmentedTokenizer(base_tok, ph, spec=spec, mode="greedy_ltr").save(d)
    elif task_name == "C4_BoundaryStable":
        d = tok_root / "C4_BoundaryStable_16384"
        if not (d / "phrases.json").exists():
            assert candidates is not None
            ph = select_c4_boundary_stable(candidates, train_slice[:3_000], base_tok, spec=spec)
            PhraseAugmentedTokenizer(base_tok, ph, spec=spec, mode="greedy_ltr").save(d)

    return task_name, time.perf_counter() - t0


def _cpu_prepack_worker(method_name: str) -> tuple[str, str, float]:
    cache_dir = ARTIFACTS_DIR / "packed_blocks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{method_name}_120M.npy"
    t0 = time.perf_counter()
    if not npy_path.exists():
        tok = load_saved_tokenizer(method_name)
        train_docs = load_texts(DATA_DIR / "train_120M.jsonl")
        blocks = pack_documents_to_blocks(train_docs, tok, seq_len=1_024)
        np.save(npy_path, blocks)
    return method_name, str(npy_path), time.perf_counter() - t0


def _gpu_speed_probe_worker(steps: int = 20) -> float:
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    spec = ModelSpec()
    mdl = torch.compile(ResearchTransformerLM(spec).to(device))
    opt = torch.optim.AdamW(mdl.parameters(), lr=6e-4, fused=True)
    x = torch.randint(0, spec.vocab_size, (32, 1024), device=device)
    y = torch.randint(0, spec.vocab_size, (32, 1024), device=device)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = mdl(x, y, loss_reduction="mean", return_logits=False)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = mdl(x, y, loss_reduction="mean", return_logits=False)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    return (32 * 1024 * steps) / (time.perf_counter() - t0)


def _gpu_train_and_eval_method(method_name: str, build_sec: float) -> tuple[str, dict]:
    spec = TokenizerSpec()
    model_spec = ModelSpec()
    train_spec = TrainSpec()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    tok = load_saved_tokenizer(method_name, spec=spec)
    blocks_path = ARTIFACTS_DIR / "packed_blocks" / f"{method_name}_120M.npy"
    blocks = np.load(blocks_path, mmap_mode="r")

    val_docs = load_texts(DATA_DIR / "val_3M.jsonl")
    test_docs = load_texts(DATA_DIR / "test_5M.jsonl", max_docs=1_500)
    train_sample = load_texts(DATA_DIR / "train_120M.jsonl", max_docs=25_000)
    ood_docs = load_texts(DATA_DIR / "test_tinystories.jsonl", max_docs=800)
    blimp_pairs = load_blimp(DATA_DIR / "test_blimp.jsonl")

    run_dir = ARTIFACTS_DIR / "runs" / f"stageB_{method_name}_seed2026"
    print(f"\n[Stage B START] {method_name} (120M B1-eq raw text, 1,831 matched steps)", flush=True)
    model, train_summary = train_language_model(
        run_name=method_name,
        tokenizer=tok,
        train_docs=None,
        prepacked_blocks=blocks,
        val_docs=val_docs,
        output_dir=run_dir,
        model_spec=model_spec,
        train_spec=train_spec,
        regime="equal_raw_bytes",
        target_steps=1_831,
        micro_batch_size=32,
        seed=2026,
        eval_every_steps=450,
        use_compile=True,
    )

    test_1024 = evaluate_bpb_and_stats(model, test_docs, tok, device, dtype, byte_cap_context=None)
    test_matched = evaluate_bpb_and_stats(model, test_docs, tok, device, dtype, byte_cap_context=3200)
    unigram_bpb = compute_unigram_bpb(train_sample, test_docs, tok)
    ood_eval = evaluate_bpb_and_stats(model, ood_docs, tok, device, dtype, byte_cap_context=3200)
    blimp_eval = evaluate_minimal_pairs(model, blimp_pairs, tok, device, dtype)
    pert_eval = evaluate_perturbation_stability(model, test_docs[:400], tok, device, dtype)

    res = {
        "vocab_build_sec": round(build_sec, 2),
        "train_wall_sec": round(train_summary["train_wall_sec"], 2),
        "avg_tok_per_sec": round(train_summary["avg_tok_per_sec"], 1),
        "emitted_train_tokens": train_summary["emitted_tokens_processed"],
        "bytes_per_token": round(test_matched["bytes_per_token"], 4),
        "test_bpb_1024tok": round(test_1024["bpb"], 5),
        "test_bpb_matched_bytes": round(test_matched["bpb"], 5),
        "unigram_bpb": round(unigram_bpb, 5),
        "contextual_gain_bpb": round(unigram_bpb - test_matched["bpb"], 5),
        "ood_tinystories_bpb": round(ood_eval["bpb"], 5),
        "blimp_overall_acc": round(blimp_eval["overall_acc"], 4),
        "blimp_fertility_matched_acc": round(blimp_eval["fertility_matched_acc"], 4),
        "perturbation_bpb_degradation": round(pert_eval["perturbation_bpb_degradation"], 5),
    }
    del model
    torch.cuda.empty_cache()
    return method_name, res


def audit_tokenizer_stage_a(name: str, tok: TokenizerInterface, sample_docs: list[str], spec: TokenizerSpec) -> dict:
    assert tok.vocab_size == spec.total_vocab_size, f"[{name}] Vocab {tok.vocab_size} != {spec.total_vocab_size}"
    total_bytes = 0
    total_toks = 0
    phrase_slot_hits: dict[int, int] = {}
    for idx, doc in enumerate(sample_docs):
        raw_b = doc.encode("utf-8")
        ids = tok.encode(doc, add_eos=True)
        if idx < 300:
            assert tok.decode_to_bytes(ids) == raw_b, f"[{name}] Byte round-trip mismatch at doc {idx}"
        total_bytes += len(raw_b)
        total_toks += len(ids) - 1
        for tid in ids[:-1]:
            if tid >= spec.base_vocab_size:
                phrase_slot_hits[tid] = phrase_slot_hits.get(tid, 0) + 1

    active_phrases = sum(1 for c in phrase_slot_hits.values() if c >= 1)
    bpt = total_bytes / max(total_toks, 1)
    stats = {
        "name": name,
        "vocab_size": tok.vocab_size,
        "bytes_per_token": round(bpt, 4),
        "active_phrase_slots_in_sample": active_phrases,
        "phrase_token_share": round(sum(phrase_slot_hits.values()) / max(total_toks, 1), 4),
    }
    print(
        f"[Stage A Audit] {name:20s} | {bpt:.3f} B/tok | "
        f"active_phrases={active_phrases}/{spec.phrase_vocab_slots} | share={stats['phrase_token_share']*100:.1f}%",
        flush=True,
    )
    return stats


def main() -> None:
    spec = TokenizerSpec()
    model_spec = ModelSpec()
    tok_root = ARTIFACTS_DIR / "tokenizers"
    spawn_ctx = mp.get_context("spawn")

    print("[Stage B] Mining shared phrase candidate pool on CPU...", flush=True)
    base_tok = Tokenizer.from_file(str(tok_root / "shared_base_14336.json"))
    train_slice = load_texts(DATA_DIR / "train_120M.jsonl", max_docs=60_000)
    docs_base_ids = [base_tok.encode(d, add_special_tokens=False).ids for d in train_slice]
    candidates = mine_phrase_candidates(docs_base_ids, base_tok, spec=spec, max_candidates=10_000)
    print(f"[Stage B] Mined {len(candidates):,} phrase candidates. Launching parallel CPU + GPU builders!", flush=True)

    build_times: dict[str, float] = {"B1_ByteBPE": 0.0}
    cpu_tasks = [
        ("B2_Unigram", None),
        ("B3_SaGe", None),
        ("B4_SuperBPE", None),
        ("B5_SupraTok", None),
        ("C0_FreqMerge", candidates),
        ("C2_ReserveMerge", candidates),
        ("C4_BoundaryStable", candidates),
    ]

    # Launch all 7 CPU-only tokenizer builds in parallel WHILE the GPU runs C1, C3, and C5!
    with ProcessPoolExecutor(max_workers=7, mp_context=spawn_ctx) as cpu_pool:
        futures = [cpu_pool.submit(_cpu_build_worker, tname, cands) for tname, cands in cpu_tasks]

        # Simultaneously run GPU Pilot-guided selectors (C1, C3, C5) on GB10!
        device = torch.device("cuda:0")
        calib_a_ids = [base_tok.encode(d, add_special_tokens=False).ids for d in load_texts(DATA_DIR / "calib_a_5M.jsonl")]
        calib_b_ids = [base_tok.encode(d, add_special_tokens=False).ids for d in load_texts(DATA_DIR / "calib_b_5M.jsonl")]

        pilot_ckpt = ARTIFACTS_DIR / "runs" / "pilot_shared_base_14336" / "ckpt_final.pt"
        pilot_model = ResearchTransformerLM(model_spec).to(device)
        pilot_model.load_state_dict(torch.load(pilot_ckpt, map_location=device, weights_only=False)["model"])

        c1_dir = tok_root / "C1_ConflictMerge_16384"
        vegad_dir = tok_root / "Control_VEGAD_16384"
        t0 = time.perf_counter()
        if not (c1_dir / "phrases.json").exists():
            print("[GPU Build] Running C1 ConflictMerge + VEGAD Control...", flush=True)
            c1_ph, veg_ph = select_c1_conflict_and_vegad_control(
                pilot_model, candidates[:6_000], calib_a_ids, calib_b_ids, spec=spec, device=device
            )
            PhraseAugmentedTokenizer(base_tok, c1_ph, spec=spec, mode="greedy_ltr").save(c1_dir)
            PhraseAugmentedTokenizer(base_tok, veg_ph, spec=spec, mode="greedy_ltr").save(vegad_dir)
        build_times["C1_ConflictMerge"] = time.perf_counter() - t0

        c3_dir = tok_root / "C3_SpilloverMerge_16384"
        c5_dir = tok_root / "C5_ContrastMerge_16384"
        t0 = time.perf_counter()
        if not ((c3_dir / "phrases.json").exists() and (c5_dir / "phrases.json").exists()):
            print("[GPU Build] Running batched C3 SpilloverMerge + C5 ContrastMerge probes...", flush=True)
            c3_ph, c5_ph = fit_and_score_c3_c5_probes(
                pilot_model, candidates[:6_000], calib_a_ids, calib_b_ids, base_tok, spec=spec, device=device
            )
            PhraseAugmentedTokenizer(base_tok, c3_ph, spec=spec, mode="greedy_ltr").save(c3_dir)
            PhraseAugmentedTokenizer(base_tok, c5_ph, spec=spec, mode="greedy_ltr").save(c5_dir)
        p_sec = time.perf_counter() - t0
        build_times["C3_SpilloverMerge"] = p_sec
        build_times["C5_ContrastMerge"] = p_sec

        del pilot_model
        torch.cuda.empty_cache()

        for fut in as_completed(futures):
            tname, sec = fut.result()
            build_times[tname] = sec
            print(f"[CPU Build DONE] {tname} in {sec:.1f}s", flush=True)

    run_order = [
        "B1_ByteBPE",
        "C0_FreqMerge",
        "C2_ReserveMerge",
        "C1_ConflictMerge",
        "C3_SpilloverMerge",
        "B4_SuperBPE",
        "B5_SupraTok",
        "C4_BoundaryStable",
        "C5_ContrastMerge",
        "B2_Unigram",
        "B3_SaGe",
    ]

    # Stage A Full Audit across all 11 methods
    val_sample = load_texts(DATA_DIR / "val_3M.jsonl", max_docs=500) + load_texts(DATA_DIR / "test_tinystories.jsonl", max_docs=200)
    audit_results = {name: audit_tokenizer_stage_a(name, load_saved_tokenizer(name, spec), val_sample, spec) for name in run_order}
    (ARTIFACTS_DIR / "stage_a_audit.json").write_text(json.dumps(audit_results, indent=2), encoding="utf-8")

    # Parallel CPU pre-packing of all 11 training corpora
    print("[Prepack] Packing all 11 120M corpora in parallel across Grace CPU cores...", flush=True)
    with ProcessPoolExecutor(max_workers=11, mp_context=spawn_ctx) as pack_pool:
        for mname, _, psec in pack_pool.map(_cpu_prepack_worker, run_order):
            print(f"  packed {mname} in {psec:.1f}s", flush=True)

    # Empirical 1-worker vs 2-parallel-worker GPU throughput duel on GB10!
    print("[GPU Duel] Measuring 1-process vs 2-parallel-process aggregate throughput on GB10...", flush=True)
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn_ctx) as p1:
        r1 = list(p1.map(_gpu_speed_probe_worker, [20]))[0]
    with ProcessPoolExecutor(max_workers=2, mp_context=spawn_ctx) as p2:
        r2_list = list(p2.map(_gpu_speed_probe_worker, [20, 20]))
        r2_total = sum(r2_list)

    gpu_concurrency = 2 if r2_total > 1.08 * r1 else 1
    print(
        f"[GPU Duel Result] 1-worker={r1:,.0f} tok/s | 2-workers={r2_total:,.0f} tok/s "
        f"({r2_list[0]:,.0f} + {r2_list[1]:,.0f}) -> Selecting gpu_concurrency={gpu_concurrency}!",
        flush=True,
    )

    scoreboard_path = ARTIFACTS_DIR / "stage_b_scoreboard.json"
    scoreboard: dict = json.loads(scoreboard_path.read_text(encoding="utf-8")) if scoreboard_path.exists() else {}
    todo = [
        m for m in run_order
        if not (m in scoreboard and (ARTIFACTS_DIR / "runs" / f"stageB_{m}_seed2026" / "ckpt_final.pt").exists())
    ]

    if gpu_concurrency == 1:
        for mname in todo:
            _, res = _gpu_train_and_eval_method(mname, build_times.get(mname, 0.0))
            scoreboard[mname] = res
            scoreboard_path.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
            print(f"[Scoreboard Updated] {mname}: {json.dumps(res)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=gpu_concurrency, mp_context=spawn_ctx) as gpu_pool:
            fut_map = {
                gpu_pool.submit(_gpu_train_and_eval_method, mname, build_times.get(mname, 0.0)): mname
                for mname in todo
            }
            for fut in as_completed(fut_map):
                mname, res = fut.result()
                scoreboard[mname] = res
                scoreboard_path.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
                print(f"[Scoreboard Updated] {mname}: {json.dumps(res)}", flush=True)


if __name__ == "__main__":
    main()
