"""
Stages C & E Publication Scale-Up Orchestrator:
  - Stage C (Statistical Confirmation @ 30.42M Params, 300M Unique Tokens):
      Runs 4 finalists (`B1_ByteBPE`, `B4_SuperBPE`, `B5_SupraTok`, `BT3_FullSynthesis`) across
      3 paired random seeds `{1001, 2002, 3003}` at exact Equal Training FLOPs (`300,000,768`
      emitted tokens = `292,969` blocks, `4,577` steps, ZERO document repetition!) plus `Iso-Bytes`
      (`300M` B1-reference raw-text exposure) to report paired mean ± std and p-values.
  - Stage E (5x Zero-Shot Proxy-to-Target Scale-Up @ 149.63M Params, 300M Tokens):
      Tests whether the `16,384` vocabulary discovered by the `30.4M` Pilot transfers ZERO-SHOT
      to a `5x` larger `149.63M`-parameter Transformer (`14` layers, `d=896`, `14` heads, `head_dim=64`,
      `d_ff=3584`) against `B1_ByteBPE`, `B4_SuperBPE`, and `B5_SupraTok` under identical FLOPs!
  - Expanded Evaluation Suite:
      Adds real OpenAI/CIMEC `LAMBADA` test set, `ARC-Easy`, and `SciQ` zero-shot multiple-choice
      accuracy alongside Byte-Matched `BPB`, `TinyStories` OOD, Fertility-Matched `BLiMP`,
      `Prefix-Cone` vs `Naive` decoding, `Gearshift Pareto`, and `14-Layer Attention X-Ray`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_breakthroughs_stage_cd import train_foldable_residual_model
from scripts.run_stage_b_all import load_blimp, load_saved_tokenizer, load_texts
from scripts.run_stage_cde_mechanistic import STAGE_E_150M_SPEC, run_layer_offloading_xray
from src.baselines import HFByteTokenizerWrapper, TokenizerInterface
from src.breakthroughs import (
    build_first_word_prefix_cones,
    evaluate_compute_gearshift_pareto,
    evaluate_next_word_vote_splitting,
    pack_dual_granularity_blocks,
)
from src.config import ModelSpec, TokenizerSpec, TrainSpec
from src.eval_suite import (
    compute_unigram_bpb,
    evaluate_bpb_and_stats,
    evaluate_minimal_pairs,
    evaluate_perturbation_stability,
)
from src.model import ResearchTransformerLM
from src.tokenizer_core import PhraseAugmentedTokenizer
from src.trainer import pack_documents_to_blocks, train_language_model


ROOT_DIR = Path(os.environ.get("TOK_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("TOK_DATA_DIR", ROOT_DIR / "data"))
ARTIFACTS_DIR = Path(os.environ.get("TOK_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))


def sha1_prefix(text: str) -> bytes:
    return hashlib.sha1(text[:2000].strip().lower().encode("utf-8")).digest()


def ensure_350m_corpus_and_downstream_evals(b1: HFByteTokenizerWrapper) -> None:
    """
    Streams a strictly disjoint 350M-B1-token FineWeb-Edu corpus (guaranteeing 0% overlap with
    calib_a, calib_b, val_3M, and test_5M, and 0% repetition even for compressed Iso-FLOPs runs!),
    plus caches real LAMBADA, ARC-Easy, and SciQ held-out benchmarks.
    """
    train_350_path = DATA_DIR / "train_350M.jsonl"
    if not train_350_path.exists():
        print("[Stage C/E Data] Building strictly disjoint 350M-B1-token FineWeb-Edu corpus...", flush=True)
        forbidden_hashes: set[bytes] = set()
        for holdout_name in ("calib_a_5M.jsonl", "calib_b_5M.jsonl", "val_3M.jsonl", "test_5M.jsonl"):
            hp = DATA_DIR / holdout_name
            if hp.exists():
                with hp.open("r", encoding="utf-8") as f:
                    for line in f:
                        forbidden_hashes.add(sha1_prefix(json.loads(line)["text"]))

        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        seen = set(forbidden_hashes)
        b1_toks = 0
        n_docs = 0
        t0 = time.perf_counter()
        with train_350_path.open("w", encoding="utf-8") as out_f:
            for idx, row in enumerate(ds):
                txt = row.get("text", "")
                if not txt or len(txt) < 200:
                    continue
                if len(txt) > 40_000:
                    txt = txt[:40_000]
                h = sha1_prefix(txt)
                if h in seen:
                    continue
                seen.add(h)
                n_t = len(b1.encode(txt, add_eos=True))
                out_f.write(json.dumps({"id": str(row.get("id", f"fw350_{idx}")), "text": txt}, ensure_ascii=False) + "\n")
                b1_toks += n_t
                n_docs += 1
                if n_docs % 50_000 == 0:
                    print(f"  streamed {n_docs:,} docs ({b1_toks/1e6:.1f}M B1 tokens) in {time.perf_counter()-t0:.1f}s", flush=True)
                if b1_toks >= 350_000_000:
                    break
        print(f"[Stage C/E Data] Saved train_350M.jsonl: {n_docs:,} docs, {b1_toks:,} B1 tokens.", flush=True)

    # Cache LAMBADA held-out test set
    lambada_path = DATA_DIR / "test_lambada.jsonl"
    if not lambada_path.exists():
        print("[Stage C/E Data] Downloading held-out LAMBADA test benchmark...", flush=True)
        try:
            lds = load_dataset("cimec/lambada", split="test")
        except Exception:
            lds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        with lambada_path.open("w", encoding="utf-8") as f:
            for row in list(lds)[:1_000]:
                txt = row["text"].strip()
                if " " in txt:
                    f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

    # Cache ARC-Easy & SciQ multiple-choice reasoning tasks
    mc_path = DATA_DIR / "test_mc_reasoning.jsonl"
    if not mc_path.exists():
        print("[Stage C/E Data] Downloading ARC-Easy & SciQ zero-shot multiple-choice benchmarks...", flush=True)
        items: list[dict] = []
        try:
            arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            for r in list(arc)[:500]:
                labels = r["choices"]["label"]
                texts = r["choices"]["text"]
                if r["answerKey"] in labels and len(texts) >= 3:
                    ans_idx = labels.index(r["answerKey"])
                    items.append({"task": "arc_easy", "stem": f"Question: {r['question']}\nAnswer:", "choices": texts, "gold": ans_idx})
        except Exception as exc:
            print(f"  [warn] ARC-Easy download skipped: {exc}", flush=True)

        try:
            sciq = load_dataset("allenai/sciq", split="test")
            for r in list(sciq)[:500]:
                choices = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
                items.append({"task": "sciq", "stem": f"Question: {r['question']}\nAnswer:", "choices": choices, "gold": 0})
        except Exception as exc:
            print(f"  [warn] SciQ download skipped: {exc}", flush=True)

        with mc_path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[Stage C/E Data] Saved {len(items)} ARC-Easy + SciQ items.", flush=True)


@torch.no_grad()
def evaluate_real_lambada(
    model: ResearchTransformerLM,
    tok: TokenizerInterface,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    lpath = DATA_DIR / "test_lambada.jsonl"
    if not lpath.exists():
        return {"lambada_naive_acc": 0.0, "lambada_prefix_cone_acc": 0.0}

    tok_to_group_cpu, group_words = build_first_word_prefix_cones(tok)
    tok_to_group = tok_to_group_cpu.to(device)
    n_groups = len(group_words)

    prompts: list[list[int]] = []
    gold_words: list[str] = []
    with lpath.open("r", encoding="utf-8") as f:
        for line in f:
            txt = json.loads(line)["text"].strip()
            idx = txt.rfind(" ")
            if idx <= 10:
                continue
            pref = txt[:idx]
            target_w = " " + txt[idx + 1 :].strip(".,;:!?\"'()")
            if not target_w.strip():
                continue
            p_ids = [tok.eos_id] + tok.encode(pref, add_eos=False)
            if len(p_ids) > model.spec.max_seq_len:
                p_ids = p_ids[-model.spec.max_seq_len :]
            prompts.append(p_ids)
            gold_words.append(target_w)

    naive_ok = 0
    cone_ok = 0
    for b0 in range(0, len(prompts), 64):
        bp = prompts[b0 : b0 + 64]
        bg = gold_words[b0 : b0 + 64]
        B = len(bp)
        max_l = max(len(s) for s in bp)
        bx = torch.full((B, max_l), tok.pad_id, dtype=torch.long, device=device)
        last_idx = torch.empty(B, dtype=torch.long, device=device)
        for i, s in enumerate(bp):
            bx[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            last_idx[i] = len(s) - 1
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(bx, active_output_vocab=tok.vocab_size)
        probs = F.softmax(logits[torch.arange(B, device=device), last_idx].float(), dim=-1)

        n_vid = probs.argmax(dim=-1).cpu().tolist()
        cone_p = torch.zeros((B, n_groups), dtype=torch.float32, device=device)
        cone_p.scatter_add_(1, tok_to_group.unsqueeze(0).expand(B, -1), probs)
        c_gid = cone_p.argmax(dim=-1).cpu().tolist()

        for i, g in enumerate(bg):
            if group_words[int(tok_to_group_cpu[n_vid[i]])].lower() == g.lower():
                naive_ok += 1
            if group_words[c_gid[i]].lower() == g.lower():
                cone_ok += 1

    tot = max(len(prompts), 1)
    return {
        "lambada_naive_acc": round(naive_ok / tot, 4),
        "lambada_prefix_cone_acc": round(cone_ok / tot, 4),
        "lambada_cone_gain": round((cone_ok - naive_ok) / tot, 4),
    }


@torch.no_grad()
def evaluate_mc_reasoning(
    model: ResearchTransformerLM,
    tok: TokenizerInterface,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Byte-length-normalized multiple-choice completion accuracy on ARC-Easy & SciQ."""
    mc_path = DATA_DIR / "test_mc_reasoning.jsonl"
    if not mc_path.exists():
        return {"arc_easy_acc": 0.0, "sciq_acc": 0.0}

    items = [json.loads(line) for line in mc_path.open("r", encoding="utf-8")]
    hits: dict[str, list[int]] = {"arc_easy": [], "sciq": []}
    model.eval()

    for it in items:
        stem = it["stem"]
        stem_ids = [tok.eos_id] + tok.encode(stem, add_eos=False)
        choice_scores: list[float] = []
        seqs: list[list[int]] = []
        c_byte_lens: list[int] = []
        s_lens: list[int] = []

        for ch in it["choices"]:
            ch_str = " " + ch.strip()
            full_ids = [tok.eos_id] + tok.encode(stem + ch_str, add_eos=False)
            seqs.append(full_ids)
            s_lens.append(min(len(stem_ids), len(full_ids) - 1))
            c_byte_lens.append(max(len(ch_str.encode("utf-8")), 1))

        B = len(seqs)
        max_l = min(max(len(s) - 1 for s in seqs), model.spec.max_seq_len)
        bx = torch.full((B, max_l), tok.pad_id, dtype=torch.long, device=device)
        by = torch.full((B, max_l), -100, dtype=torch.long, device=device)
        for i, (s, sl) in enumerate(zip(seqs, s_lens)):
            L = min(len(s) - 1, max_l)
            bx[i, :L] = torch.tensor(s[:L], dtype=torch.long, device=device)
            for pos in range(max(sl - 1, 0), L):
                by[i, pos] = s[pos + 1]

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(bx, active_output_vocab=tok.vocab_size)
            nll = F.cross_entropy(logits.view(-1, tok.vocab_size), by.view(-1), ignore_index=-100, reduction="none").view(B, max_l)
        seq_nll = (nll.float() * (by != -100).float()).sum(dim=1).cpu().tolist()
        # Normalize log-prob by raw UTF-8 byte length of choice (standard EleutherAI lm-eval byte_norm!)
        choice_scores = [-nll_v / b_len for nll_v, b_len in zip(seq_nll, c_byte_lens)]
        pred = int(np.argmax(choice_scores))
        hits[it["task"]].append(1 if pred == it["gold"] else 0)

    return {
        "arc_easy_acc": round(float(np.mean(hits["arc_easy"])) if hits["arc_easy"] else 0.0, 4),
        "sciq_acc": round(float(np.mean(hits["sciq"])) if hits["sciq"] else 0.0, 4),
    }


def comprehensive_eval(
    model: ResearchTransformerLM,
    tok: TokenizerInterface,
    train_sample: list[str],
    test_docs: list[str],
    ood_docs: list[str],
    blimp_pairs: list[tuple[str, str]],
    device: torch.device,
) -> dict:
    dtype = torch.bfloat16
    t_1024 = evaluate_bpb_and_stats(model, test_docs, tok, device, dtype, byte_cap_context=None)
    t_match = evaluate_bpb_and_stats(model, test_docs, tok, device, dtype, byte_cap_context=3200)
    uni_bpb = compute_unigram_bpb(train_sample, test_docs, tok)
    ood = evaluate_bpb_and_stats(model, ood_docs, tok, device, dtype, byte_cap_context=3200)
    blimp = evaluate_minimal_pairs(model, blimp_pairs, tok, device, dtype)
    pert = evaluate_perturbation_stability(model, test_docs[:400], tok, device, dtype)
    vote = evaluate_next_word_vote_splitting(model, test_docs[:800], tok, device, dtype)
    lamb = evaluate_real_lambada(model, tok, device, dtype)
    mc = evaluate_mc_reasoning(model, tok, device, dtype)
    return {
        "bytes_per_token": round(t_match["bytes_per_token"], 4),
        "test_bpb_1024tok": round(t_1024["bpb"], 5),
        "test_bpb_matched_bytes": round(t_match["bpb"], 5),
        "unigram_bpb": round(uni_bpb, 5),
        "contextual_gain_bpb": round(uni_bpb - t_match["bpb"], 5),
        "ood_tinystories_bpb": round(ood["bpb"], 5),
        "blimp_overall_acc": round(blimp["overall_acc"], 4),
        "blimp_fertility_matched_acc": round(blimp["fertility_matched_acc"], 4),
        "lambada_naive_acc": lamb["lambada_naive_acc"],
        "lambada_prefix_cone_acc": lamb["lambada_prefix_cone_acc"],
        "arc_easy_acc": mc["arc_easy_acc"],
        "sciq_acc": mc["sciq_acc"],
        "next_word_naive_acc": vote["naive_argmax_acc"],
        "next_word_prefix_cone_acc": vote["prefix_cone_argmax_acc"],
        "perturbation_bpb_degradation": round(pert["perturbation_bpb_degradation"], 5),
    }


def _pack_300m_worker(args: tuple[str, bool]) -> tuple[str, float]:
    mname, dual_gear = args
    spec = TokenizerSpec()
    if mname == "BT3_FullSynthesis":
        tok_obj = PhraseAugmentedTokenizer.load(ARTIFACTS_DIR / "tokenizers" / "BT2_AttnDemux_16384", spec=spec)
    else:
        tok_obj = load_saved_tokenizer(mname, spec=spec)
    p = ARTIFACTS_DIR / "packed_blocks_300M" / f"{mname}_300M_isoflops.npy"
    t0 = time.perf_counter()
    if not p.exists():
        docs = load_texts(DATA_DIR / "train_350M.jsonl")
        if dual_gear:
            assert isinstance(tok_obj, PhraseAugmentedTokenizer)
            arr = pack_dual_granularity_blocks(docs, tok_obj, unmerge_prob=0.20, seq_len=1_024, seed=2026)
        else:
            arr = pack_documents_to_blocks(docs, tok_obj, seq_len=1_024)
        assert len(arr) >= 292_969, f"{mname} produced {len(arr)} < 292969 blocks!"
        np.save(p, arr[:292_969])
    return mname, time.perf_counter() - t0


def reclaim_clean_page_cache(gb: int = 24) -> None:
    """Nudges Linux kswapd to evict stale clean file page-cache so GB10 unified cudaMalloc has >30 GiB MemFree."""
    import gc
    try:
        buf = np.ones((gb * 512, 2 * 1024 * 1024), dtype=np.uint8)
        del buf
        gc.collect()
    except MemoryError:
        gc.collect()


def main() -> None:
    reclaim_clean_page_cache(24)
    spec = TokenizerSpec()
    spec_30m = ModelSpec()
    spec_150m = STAGE_E_150M_SPEC
    train_spec_30m = TrainSpec()
    train_spec_150m = TrainSpec(peak_lr=4e-4, min_lr=4e-5)  # standard slightly lower LR for 150M scale
    device = torch.device("cuda:0")

    b1 = load_saved_tokenizer("B1_ByteBPE", spec=spec)
    assert isinstance(b1, HFByteTokenizerWrapper)
    ensure_350m_corpus_and_downstream_evals(b1)

    b4 = load_saved_tokenizer("B4_SuperBPE", spec=spec)
    b5 = load_saved_tokenizer("B5_SupraTok", spec=spec)
    bt2_tok = PhraseAugmentedTokenizer.load(ARTIFACTS_DIR / "tokenizers" / "BT2_AttnDemux_16384", spec=spec)

    val_docs = load_texts(DATA_DIR / "val_3M.jsonl")
    test_docs = load_texts(DATA_DIR / "test_5M.jsonl", max_docs=1_500)
    ood_docs = load_texts(DATA_DIR / "test_tinystories.jsonl", max_docs=800)
    blimp_pairs = load_blimp(DATA_DIR / "test_blimp.jsonl")

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    cache_dir = ARTIFACTS_DIR / "packed_blocks_300M"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pack_jobs = [
        ("B1_ByteBPE", False),
        ("B4_SuperBPE", False),
        ("B5_SupraTok", False),
        ("BT3_FullSynthesis", True),
    ]
    print("[Stage C/E] Packing all 4 finalist 300M corpora in parallel across 4 CPU workers...", flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
        for mname, psec in pool.map(_pack_300m_worker, pack_jobs):
            print(f"  packed {mname} (300.0M tokens) in {psec:.1f}s", flush=True)

    train_sample = load_texts(DATA_DIR / "train_350M.jsonl", max_docs=25_000)
    TARGET_300M_BLOCKS = 292_969
    STEPS_300M = 4_578

    blocks_map = {
        mname: np.load(cache_dir / f"{mname}_300M_isoflops.npy", mmap_mode="r")
        for mname, _ in pack_jobs
    }
    tok_map = {
        "B1_ByteBPE": b1,
        "B4_SuperBPE": b4,
        "B5_SupraTok": b5,
        "BT3_FullSynthesis": bt2_tok,
    }

    out_path = ARTIFACTS_DIR / "stage_ce_scaleup_results.json"
    results: dict = (
        json.loads(out_path.read_text(encoding="utf-8"))
        if out_path.exists()
        else {"stage_c_300M_3seed": {}, "stage_e_150M_scaleup": {}, "stage_e_14layer_xray": {}}
    )

    def flush() -> None:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # =======================================================================
    # EXECUTION ORDER:
    # 1. FULL STAGE C FIRST (All 3 paired seeds {1001, 2002, 3003} x 4 finalists
    #    @ 30.42M params, 300.0M unique tokens -> ~6.5 hours total, writes STAGE_C_COMPLETE.json!)
    # 2. STAGE E SECOND (149.58M 5x Zero-Shot Scale-Up x 4 finalists @ 300.0M unique tokens -> +7.8 hours)
    # =======================================================================
    finalists = ["B1_ByteBPE", "B4_SuperBPE", "B5_SupraTok", "BT3_FullSynthesis"]

    # --- Phase 1: FULL STAGE C — All 3 Paired Seeds {1001, 2002, 3003} @ 300.0M Unique Tokens ---
    for seed in (1001, 2002, 3003):
        for mname in finalists:
            key = f"{mname}_seed{seed}"
            rdir = ARTIFACTS_DIR / "runs" / f"stageC_300M_{key}"
            if key in results["stage_c_300M_3seed"] and (rdir / "ckpt_final.pt").exists():
                print(f"[Skip] {key} already complete.", flush=True)
                continue
            print(f"\n=== [Stage C 300M (Seed {seed})] {key} (30.42M params, 300.0M unique tokens) ===", flush=True)
            if mname == "BT3_FullSynthesis":
                mdl, raw_f, tsum = train_foldable_residual_model(
                    run_name=key, phrase_tok=bt2_tok, blocks=blocks_map[mname], val_docs=val_docs,
                    output_dir=rdir, model_spec=spec_30m, train_spec=train_spec_30m,
                    target_steps=STEPS_300M, seed=seed,
                )
                del raw_f
            else:
                mdl, tsum = train_language_model(
                    run_name=key, tokenizer=tok_map[mname], train_docs=None, prepacked_blocks=blocks_map[mname],
                    val_docs=val_docs, output_dir=rdir, model_spec=spec_30m, train_spec=train_spec_30m,
                    regime="equal_raw_bytes", target_steps=STEPS_300M, seed=seed, eval_every_steps=900, use_compile=True,
                )
            ev = comprehensive_eval(mdl, tok_map[mname], train_sample, test_docs, ood_docs, blimp_pairs, device)
            if mname == "BT3_FullSynthesis" and seed == 1001:
                ev["gearshift_pareto"] = evaluate_compute_gearshift_pareto(mdl, test_docs[:600], bt2_tok, device)
            ev["train_wall_sec"] = round(tsum["train_wall_sec"], 2)
            results["stage_c_300M_3seed"][key] = ev
            flush()
            print(f"[Done {key}] {json.dumps(ev)}", flush=True)
            del mdl
            torch.cuda.empty_cache()

    (ARTIFACTS_DIR / "STAGE_C_COMPLETE.json").write_text(
        json.dumps({"stage_c_done": True, "runs": len(results["stage_c_300M_3seed"])}, indent=2),
        encoding="utf-8",
    )
    print("\n=== [STAGE C 3-SEED 300M CONFIRMATION 100% COMPLETE -> STARTING STAGE E 150M SCALE-UP] ===", flush=True)

    # --- Phase 2: STAGE E — 149.58M Parameter 5x Zero-Shot Scale-Up @ 300M Unique Tokens! ---
    for mname in finalists:
        key = f"150M_{mname}_seed1001"
        rdir = ARTIFACTS_DIR / "runs" / f"stageE_{key}"
        if key in results["stage_e_150M_scaleup"] and (rdir / "ckpt_final.pt").exists():
            print(f"[Skip] {key} already complete.", flush=True)
            continue
        reclaim_clean_page_cache(20)
        print(f"\n=== [STAGE E 150M SCALE-UP] {key} (149.58M params, 300.0M unique tokens, Zero-Shot Vocab Transfer!) ===", flush=True)
        if mname == "BT3_FullSynthesis":
            mdl, raw_f, tsum = train_foldable_residual_model(
                run_name=key, phrase_tok=bt2_tok, blocks=blocks_map[mname], val_docs=val_docs,
                output_dir=rdir, model_spec=spec_150m, train_spec=train_spec_150m,
                target_steps=STEPS_300M, micro_batch_size=16, seed=1001,
            )
            del raw_f
        else:
            mdl, tsum = train_language_model(
                run_name=key, tokenizer=tok_map[mname], train_docs=None, prepacked_blocks=blocks_map[mname],
                val_docs=val_docs, output_dir=rdir, model_spec=spec_150m, train_spec=train_spec_150m,
                regime="equal_raw_bytes", target_steps=STEPS_300M, micro_batch_size=16, seed=1001,
                eval_every_steps=900, use_compile=True,
            )
        ev = comprehensive_eval(mdl, tok_map[mname], train_sample, test_docs, ood_docs, blimp_pairs, device)
        if mname == "BT3_FullSynthesis":
            ev["gearshift_pareto"] = evaluate_compute_gearshift_pareto(mdl, test_docs[:600], bt2_tok, device)
        ev["train_wall_sec"] = round(tsum["train_wall_sec"], 2)
        results["stage_e_150M_scaleup"][key] = ev
        results["stage_e_14layer_xray"][mname] = run_layer_offloading_xray(mdl, tok_map[mname], test_docs[:120], device)
        flush()
        print(f"[Done {key}] {json.dumps(ev)}", flush=True)
        del mdl
        torch.cuda.empty_cache()

    print("\n[ALL STAGE C 3-SEED & STAGE E 150M RUNS 100% COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
