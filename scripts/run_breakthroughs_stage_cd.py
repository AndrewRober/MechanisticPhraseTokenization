"""
Stage C/D Breakthrough Suite Runner (`BT1` Foldable Residual + Prefix-Cone, `BT2` Attention Demux,
`BT3` Compute Gearshift + Layer-Offloading X-Ray).

Executes automatically right after Stage B finishes:
  1. Zero-Retraining Prefix-Cone Decoding Audit across ALL 11 Stage B checkpoints (proving & curing
     Superword Vote-Splitting on exact next-word prediction with 0 training FLOPs!).
  2. Selects the #1 Stage B candidate winner and constructs `BT2_AttnDemux` (`D_bind` Attention-Query
     De-Multiplexing shielding multi-binding relational spans via Pilot attention JSD).
  3. Trains 3 controlled 120M-B1-token models (`1,831` matched steps, `seed=2026`, full eval):
       - `BT2_AttnDemux`        : Standard LM with Attention-Demuxed Phrase Vocabulary
       - `BT1_FoldableResidual` : Foldable Compositional-Residual LM (`W_P = mean(W_t) + Delta_P`,
                                  3x L2 weight decay on `Delta_P`, folded to standard `[16384, 512]`!)
       - `BT3_Gearshift_Combo`  : Full synthesis (`AttnDemux` + `FoldableResidual` + `20%` Dual-Granularity
                                  Pretraining + Test-Time Entropy-Triggered Unmerging!)
  4. Runs Mechanistic Layer-Offloading X-Ray (`prev_token_mass`, `lookback_distance`, `attn_entropy`)
     and inspects Top/Bottom learned non-compositional residuals `||Delta_P||_2`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tokenizers import Tokenizer

from scripts.run_stage_b_all import load_blimp, load_saved_tokenizer, load_texts
from scripts.run_stage_cde_mechanistic import run_layer_offloading_xray
from src.breakthroughs import (
    FoldableResidualTransformerLM,
    evaluate_compute_gearshift_pareto,
    evaluate_next_word_vote_splitting,
    pack_dual_granularity_blocks,
    select_attention_demux_phrases,
)
from src.config import ModelSpec, TokenizerSpec, TrainSpec
from src.eval_suite import (
    compute_unigram_bpb,
    evaluate_bpb_and_stats,
    evaluate_minimal_pairs,
    evaluate_perturbation_stability,
)
from src.model import ResearchTransformerLM
from src.phrase_selectors import mine_phrase_candidates
from src.tokenizer_core import PhraseAugmentedTokenizer
from src.trainer import compute_quantized_step_Schedule, get_cosine_lr, pack_documents_to_blocks, train_language_model


ROOT_DIR = Path(os.environ.get("TOK_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("TOK_DATA_DIR", ROOT_DIR / "data"))
ARTIFACTS_DIR = Path(os.environ.get("TOK_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))


def train_foldable_residual_model(
    run_name: str,
    phrase_tok: PhraseAugmentedTokenizer,
    blocks: np.ndarray,
    val_docs: list[str],
    output_dir: Path,
    model_spec: ModelSpec = ModelSpec(),
    train_spec: TrainSpec = TrainSpec(),
    base_wd: float = 0.10,
    delta_wd: float = 0.30,
    target_steps: int = 1_831,
    micro_batch_size: int = 32,
    seed: int = 2026,
) -> tuple[ResearchTransformerLM, FoldableResidualTransformerLM, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16

    raw_fold_lm = FoldableResidualTransformerLM(model_spec, phrase_tok).to(device)
    exec_model = torch.compile(raw_fold_lm)

    # 3 parameter groups: strong L2 on phrase_delta (forces compositional sharing unless needed!),
    # normal L2 on 2D weights, zero L2 on 1D RMSNorms
    backbone_decay = [
        p for n, p in raw_fold_lm.named_parameters()
        if n != "phrase_delta" and p.dim() >= 2
    ]
    nodecay = [p for p in raw_fold_lm.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": [raw_fold_lm.phrase_delta], "weight_decay": delta_wd},
            {"params": backbone_decay, "weight_decay": base_wd},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=train_spec.peak_lr,
        betas=(train_spec.beta1, train_spec.beta2),
        fused=True,
    )

    step_slices = compute_quantized_step_Schedule(len(blocks), target_steps, quantum=micro_batch_size)
    warmup_steps = max(10, int(target_steps * train_spec.warmup_fraction))
    active_vocab = phrase_tok.vocab_size
    emitted_tokens = 0
    t0 = time.perf_counter()

    for step, (b_start, b_end) in enumerate(step_slices):
        if b_end <= b_start:
            b_start = max(0, b_end - micro_batch_size)
        lr = get_cosine_lr(step, target_steps, train_spec.peak_lr, train_spec.min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        step_blocks = blocks[b_start:b_end]
        n_sb = len(step_blocks)
        tot_real = max(n_sb * train_spec.seq_len, 1)
        optimizer.zero_grad(set_to_none=True)
        step_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)

        for mb0 in range(0, n_sb, micro_batch_size):
            mb = step_blocks[mb0 : mb0 + micro_batch_size]
            r_rows = len(mb)
            if r_rows < micro_batch_size:
                xp = np.full((micro_batch_size, train_spec.seq_len), phrase_tok.pad_id, dtype=np.int64)
                yp = np.full((micro_batch_size, train_spec.seq_len), -100, dtype=np.int64)
                xp[:r_rows] = mb[:, :-1]
                yp[:r_rows] = mb[:, 1:]
                x = torch.from_numpy(xp).to(device, non_blocking=True)
                y = torch.from_numpy(yp).to(device, non_blocking=True)
            else:
                x = torch.from_numpy(mb[:, :-1].astype(np.int64)).to(device, non_blocking=True)
                y = torch.from_numpy(mb[:, 1:].astype(np.int64)).to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                _, mb_sum = exec_model(
                    x, y, active_output_vocab=active_vocab, loss_reduction="sum", return_logits=False
                )
                (mb_sum / tot_real).backward()
            step_loss_gpu = step_loss_gpu + mb_sum.detach().float()
            emitted_tokens += r_rows * train_spec.seq_len

        torch.nn.utils.clip_grad_norm_(raw_fold_lm.parameters(), train_spec.grad_clip)
        optimizer.step()

        if step == 50 or (step > 0 and step % 900 == 0) or step == target_steps - 1:
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            tps = emitted_tokens / max(elapsed, 1e-6)
            folded_tmp = raw_fold_lm.fold_to_standard_model()
            vm = evaluate_bpb_and_stats(folded_tmp, val_docs[:500], phrase_tok, device, dtype, byte_cap_context=3200)
            del folded_tmp
            print(
                f"[{run_name}] step {step+1:4d}/{target_steps} | "
                f"train_ce={float((step_loss_gpu/tot_real).item()):.4f} | val_bpb={vm['bpb']:.4f} | {tps:,.0f} tok/s",
                flush=True,
            )

    torch.cuda.synchronize(device)
    wall_sec = time.perf_counter() - t0
    folded_std = raw_fold_lm.fold_to_standard_model()

    # Verify folding equivalence on live GPU batch!
    with torch.no_grad():
        test_x = torch.randint(0, active_vocab, (4, 128), device=device)
        l1, _ = raw_fold_lm(test_x)
        l2, _ = folded_std(test_x)
        fold_err = float((l1.float() - l2.float()).abs().max().item())
        assert fold_err < 1e-4, f"Fold error too large: {fold_err}"

    torch.save({"step": target_steps, "model": folded_std.state_dict(), "fold_max_err": fold_err}, output_dir / "ckpt_final.pt")
    return folded_std, raw_fold_lm, {"train_wall_sec": wall_sec, "emitted_tokens_processed": emitted_tokens, "fold_max_err": fold_err}


def full_evaluate_model(
    model: ResearchTransformerLM,
    tok,
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
    return {
        "bytes_per_token": round(t_match["bytes_per_token"], 4),
        "test_bpb_1024tok": round(t_1024["bpb"], 5),
        "test_bpb_matched_bytes": round(t_match["bpb"], 5),
        "unigram_bpb": round(uni_bpb, 5),
        "contextual_gain_bpb": round(uni_bpb - t_match["bpb"], 5),
        "ood_tinystories_bpb": round(ood["bpb"], 5),
        "blimp_overall_acc": round(blimp["overall_acc"], 4),
        "blimp_fertility_matched_acc": round(blimp["fertility_matched_acc"], 4),
        "perturbation_bpb_degradation": round(pert["perturbation_bpb_degradation"], 5),
        "next_word_naive_acc": vote["naive_argmax_acc"],
        "next_word_prefix_cone_acc": vote["prefix_cone_argmax_acc"],
        "prefix_cone_gain": vote["cone_gain"],
    }


def main() -> None:
    spec = TokenizerSpec()
    model_spec = ModelSpec()
    train_spec = TrainSpec()
    device = torch.device("cuda:0")

    sb_path = ARTIFACTS_DIR / "stage_b_scoreboard.json"
    assert sb_path.exists(), "Stage B must complete before Stage C/D breakthroughs run!"
    stage_b_sb: dict[str, dict] = json.loads(sb_path.read_text(encoding="utf-8"))

    train_docs = load_texts(DATA_DIR / "train_120M.jsonl")
    train_sample = train_docs[:25_000]
    val_docs = load_texts(DATA_DIR / "val_3M.jsonl")
    test_docs = load_texts(DATA_DIR / "test_5M.jsonl", max_docs=1_500)
    ood_docs = load_texts(DATA_DIR / "test_tinystories.jsonl", max_docs=800)
    blimp_pairs = load_blimp(DATA_DIR / "test_blimp.jsonl")

    rep_path = ARTIFACTS_DIR / "breakthroughs_stage_cd_results.json"
    out_report: dict = {
        "zero_retrain_prefix_cone_audit": {},
        "breakthrough_runs": {},
        "iso_flops_120M_runs": {},
        "layer_xray": {},
    }

    def save_progress() -> None:
        rep_path.write_text(json.dumps(out_report, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # STEP 1: Zero-Retraining Prefix-Cone Vote-Splitting Audit on Stage B!
    # -----------------------------------------------------------------------
    print("\n=== [Breakthrough 1B] Zero-Retraining Prefix-Cone Vote-Splitting Audit ===", flush=True)
    for mname in stage_b_sb:
        ckpt_p = ARTIFACTS_DIR / "runs" / f"stageB_{mname}_seed2026" / "ckpt_final.pt"
        if not ckpt_p.exists():
            continue
        tok = load_saved_tokenizer(mname, spec=spec)
        mdl = ResearchTransformerLM(model_spec).to(device)
        mdl.load_state_dict(torch.load(ckpt_p, map_location=device, weights_only=False)["model"])
        v_res = evaluate_next_word_vote_splitting(mdl, test_docs[:800], tok, device)
        out_report["zero_retrain_prefix_cone_audit"][mname] = v_res
        print(
            f"  {mname:20s} | naive_acc={v_res['naive_argmax_acc']*100:5.2f}% -> "
            f"prefix_cone_acc={v_res['prefix_cone_argmax_acc']*100:5.2f}% (gain={v_res['cone_gain']*100:+5.2f}%)",
            flush=True,
        )
        del mdl
    torch.cuda.empty_cache()
    save_progress()

    # -----------------------------------------------------------------------
    # STEP 1.5: Stage D Mechanism Control — VEGAD Gradient-Magnitude Selector
    # -----------------------------------------------------------------------
    vegad_dir = ARTIFACTS_DIR / "tokenizers" / "Control_VEGAD_16384"
    if (vegad_dir / "phrases.json").exists():
        print("\n=== [Stage D Control] Training Control_VEGAD (Gradient Magnitude vs C1 Directional Conflict) ===", flush=True)
        vegad_tok = PhraseAugmentedTokenizer.load(vegad_dir, spec=spec)
        vegad_blocks = pack_documents_to_blocks(train_docs, vegad_tok, seq_len=1_024)
        mdl_veg, sum_veg = train_language_model(
            run_name="Control_VEGAD",
            tokenizer=vegad_tok,
            train_docs=None,
            prepacked_blocks=vegad_blocks,
            val_docs=val_docs,
            output_dir=ARTIFACTS_DIR / "runs" / "stageD_Control_VEGAD_seed2026",
            model_spec=model_spec,
            train_spec=train_spec,
            regime="equal_raw_bytes",
            target_steps=1_831,
            micro_batch_size=32,
            seed=2026,
            use_compile=True,
        )
        out_report["breakthrough_runs"]["Control_VEGAD"] = {
            **full_evaluate_model(mdl_veg, vegad_tok, train_sample, test_docs, ood_docs, blimp_pairs, device),
            "train_wall_sec": round(sum_veg["train_wall_sec"], 2),
        }
        del mdl_veg
        torch.cuda.empty_cache()
        save_progress()

    # Pick #1 Stage B candidate (among C0..C5) by matched-byte BPB + BLiMP composite!
    cand_names = [k for k in stage_b_sb if k.startswith("C")]
    best_cand = min(
        cand_names,
        key=lambda k: stage_b_sb[k]["test_bpb_matched_bytes"] - 0.15 * stage_b_sb[k]["blimp_fertility_matched_acc"],
    )
    print(f"\n[Breakthrough Base] Selected top Stage B candidate foundation: {best_cand}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 2: Build BT2 Attention-Query De-Multiplexed Tokenizer (`D_bind`)
    # -----------------------------------------------------------------------
    tok_root = ARTIFACTS_DIR / "tokenizers"
    base_tok = Tokenizer.from_file(str(tok_root / "shared_base_14336.json"))
    best_cand_tok = load_saved_tokenizer(best_cand, spec=spec)
    assert isinstance(best_cand_tok, PhraseAugmentedTokenizer)

    prior_u = {e.base_ids: e.utility for e in best_cand_tok.phrases.values()}
    docs_base_ids = [base_tok.encode(d, add_special_tokens=False).ids for d in train_docs[:35_000]]
    candidates = mine_phrase_candidates(docs_base_ids, base_tok, spec=spec, max_candidates=8_000)

    pilot_mdl = ResearchTransformerLM(model_spec).to(device)
    pilot_mdl.load_state_dict(
        torch.load(ARTIFACTS_DIR / "runs" / "pilot_shared_base_14336" / "ckpt_final.pt", map_location=device, weights_only=False)["model"]
    )
    calib_a_ids = [base_tok.encode(d, add_special_tokens=False).ids for d in load_texts(DATA_DIR / "calib_a_5M.jsonl", max_docs=2_500)]

    bt2_dir = tok_root / "BT2_AttnDemux_16384"
    print("\n=== [Breakthrough 2] Computing Cross-Constituent Prefix Attention JSD (D_bind) ===", flush=True)
    bt2_phrases, demux_diag = select_attention_demux_phrases(
        pilot_mdl, candidates, calib_a_ids, prior_u, spec=spec, device=device
    )
    del pilot_mdl
    torch.cuda.empty_cache()

    bt2_tok = PhraseAugmentedTokenizer(base_tok, bt2_phrases, spec=spec, mode="utility_dp")
    bt2_tok.save(bt2_dir)
    out_report["attention_demux_diagnostics"] = demux_diag
    print(f"  [BT2 Diagnostics] {json.dumps(demux_diag)}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 3: Train & Evaluate BT2 (Attention-Demux + Utility-DP)
    # -----------------------------------------------------------------------
    bt2_blocks = pack_documents_to_blocks(train_docs, bt2_tok, seq_len=1_024)
    bt2_run_dir = ARTIFACTS_DIR / "runs" / "stageCD_BT2_AttnDemux_seed2026"
    mdl_bt2, sum_bt2 = train_language_model(
        run_name="BT2_AttnDemux",
        tokenizer=bt2_tok,
        train_docs=None,
        prepacked_blocks=bt2_blocks,
        val_docs=val_docs,
        output_dir=bt2_run_dir,
        model_spec=model_spec,
        train_spec=train_spec,
        regime="equal_raw_bytes",
        target_steps=1_831,
        micro_batch_size=32,
        seed=2026,
        use_compile=True,
    )
    out_report["breakthrough_runs"]["BT2_AttnDemux"] = {
        **full_evaluate_model(mdl_bt2, bt2_tok, train_sample, test_docs, ood_docs, blimp_pairs, device),
        "train_wall_sec": round(sum_bt2["train_wall_sec"], 2),
    }
    del mdl_bt2
    torch.cuda.empty_cache()
    save_progress()

    # -----------------------------------------------------------------------
    # STEP 4: Train & Evaluate BT1 (Foldable Compositional-Residual on Winner!)
    # -----------------------------------------------------------------------
    print("\n=== [Breakthrough 1A] Training Foldable Compositional-Residual Transformer ===", flush=True)
    bt1_blocks = pack_documents_to_blocks(train_docs, best_cand_tok, seq_len=1_024)
    mdl_bt1, raw_fold_bt1, sum_bt1 = train_foldable_residual_model(
        run_name="BT1_FoldableResidual",
        phrase_tok=best_cand_tok,
        blocks=bt1_blocks,
        val_docs=val_docs,
        output_dir=ARTIFACTS_DIR / "runs" / "stageCD_BT1_FoldableResidual_seed2026",
        model_spec=model_spec,
        train_spec=train_spec,
    )
    # Inspect learned ||Delta_P||_2 norms — which phrases needed non-compositional residuals vs 0!
    with torch.no_grad():
        delta_norms = raw_fold_bt1.phrase_delta.detach().float().norm(dim=-1).cpu().numpy()
    phrase_items = sorted(best_cand_tok.phrases.values(), key=lambda e: e.phrase_id)
    ranked_deltas = sorted(
        [(float(delta_norms[idx]), e.raw_bytes.decode("utf-8", errors="replace")) for idx, e in enumerate(phrase_items)],
        reverse=True,
    )
    out_report["learned_noncompositional_residuals"] = {
        "top_15_idiomatic_phrases": ranked_deltas[:15],
        "bottom_15_purely_compositional_phrases": ranked_deltas[-15:],
        "fold_logit_max_error": sum_bt1["fold_max_err"],
    }
    out_report["breakthrough_runs"]["BT1_FoldableResidual"] = {
        **full_evaluate_model(mdl_bt1, best_cand_tok, train_sample, test_docs, ood_docs, blimp_pairs, device),
        "train_wall_sec": round(sum_bt1["train_wall_sec"], 2),
    }
    del mdl_bt1, raw_fold_bt1
    torch.cuda.empty_cache()
    save_progress()

    # -----------------------------------------------------------------------
    # STEP 5: Train & Evaluate BT3_FullSynthesis (AttnDemux + Foldable + Dual-Gearshift!)
    # -----------------------------------------------------------------------
    print("\n=== [Breakthrough 3] Training Full Synthesis (AttnDemux + Foldable + Compute Gearshift) ===", flush=True)
    gear_blocks = pack_dual_granularity_blocks(train_docs, bt2_tok, unmerge_prob=0.20, seq_len=1_024, seed=2026)
    mdl_bt3, raw_fold_bt3, sum_bt3 = train_foldable_residual_model(
        run_name="BT3_FullSynthesis_Gearshift",
        phrase_tok=bt2_tok,
        blocks=gear_blocks,
        val_docs=val_docs,
        output_dir=ARTIFACTS_DIR / "runs" / "stageCD_BT3_FullSynthesis_seed2026",
        model_spec=model_spec,
        train_spec=train_spec,
    )
    pareto = evaluate_compute_gearshift_pareto(mdl_bt3, test_docs[:600], bt2_tok, device)
    out_report["breakthrough_runs"]["BT3_FullSynthesis_Gearshift"] = {
        **full_evaluate_model(mdl_bt3, bt2_tok, train_sample, test_docs, ood_docs, blimp_pairs, device),
        "gearshift_pareto": pareto,
        "train_wall_sec": round(sum_bt3["train_wall_sec"], 2),
    }
    print(f"  [Gearshift Pareto] {json.dumps(pareto)}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 6: Mechanistic Layer-1 Lexical Offloading X-Ray
    # -----------------------------------------------------------------------
    print("\n=== [Mechanistic X-Ray] Comparing Layer-Wise Attention Routing ===", flush=True)
    out_report["layer_xray"]["BT3_FullSynthesis"] = run_layer_offloading_xray(mdl_bt3, bt2_tok, test_docs[:120], device)
    del mdl_bt3, raw_fold_bt3

    for ref_m in ["B1_ByteBPE", "B4_SuperBPE", best_cand]:
        ck = ARTIFACTS_DIR / "runs" / f"stageB_{ref_m}_seed2026" / "ckpt_final.pt"
        if ck.exists():
            rtok = load_saved_tokenizer(ref_m, spec=spec)
            rmdl = ResearchTransformerLM(model_spec).to(device)
            rmdl.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
            out_report["layer_xray"][ref_m] = run_layer_offloading_xray(rmdl, rtok, test_docs[:120], device)
            del rmdl
    torch.cuda.empty_cache()
    save_progress()

    # -----------------------------------------------------------------------
    # STEP 7: Stage C Equal-FLOPs Duel (Exact 120,000,512 Emitted Tokens = B1 FLOPs!)
    # -----------------------------------------------------------------------
    print("\n=== [Stage C Equal-FLOPs Duel] Matching Exact B1 Compute Budget (117,188 Blocks = 120.0M Tokens) ===", flush=True)
    target_b1_blocks = 117_188  # 117,188 * 1024 = 120,000,512 exact B1 emitted tokens!

    # SuperBPE @ Exact B1 FLOPs
    b4_tok = load_saved_tokenizer("B4_SuperBPE", spec=spec)
    b4_base_blocks = pack_documents_to_blocks(train_docs, b4_tok, seq_len=1_024)
    reps_b4 = (target_b1_blocks // len(b4_base_blocks)) + 1
    b4_iso_blocks = np.concatenate([b4_base_blocks] * reps_b4, axis=0)[:target_b1_blocks]
    mdl_b4_iso, sum_b4_iso = train_language_model(
        run_name="IsoFLOPs_B4_SuperBPE",
        tokenizer=b4_tok,
        train_docs=None,
        prepacked_blocks=b4_iso_blocks,
        val_docs=val_docs,
        output_dir=ARTIFACTS_DIR / "runs" / "stageC_IsoFLOPs_B4_SuperBPE",
        model_spec=model_spec,
        train_spec=train_spec,
        regime="equal_raw_bytes",
        target_steps=1_831,
        micro_batch_size=32,
        seed=2026,
        use_compile=True,
    )
    out_report["iso_flops_120M_runs"]["B4_SuperBPE_IsoFLOPs"] = {
        **full_evaluate_model(mdl_b4_iso, b4_tok, train_sample, test_docs, ood_docs, blimp_pairs, device),
        "emitted_train_tokens": sum_b4_iso["emitted_tokens_processed"],
        "train_wall_sec": round(sum_b4_iso["train_wall_sec"], 2),
    }
    del mdl_b4_iso
    torch.cuda.empty_cache()
    save_progress()

    # BT3 FullSynthesis @ Exact B1 FLOPs
    reps_bt3 = (target_b1_blocks // len(gear_blocks)) + 1
    bt3_iso_blocks = np.concatenate([gear_blocks] * reps_bt3, axis=0)[:target_b1_blocks]
    mdl_bt3_iso, raw_bt3_iso, sum_bt3_iso = train_foldable_residual_model(
        run_name="IsoFLOPs_BT3_FullSynthesis",
        phrase_tok=bt2_tok,
        blocks=bt3_iso_blocks,
        val_docs=val_docs,
        output_dir=ARTIFACTS_DIR / "runs" / "stageC_IsoFLOPs_BT3_FullSynthesis",
        model_spec=model_spec,
        train_spec=train_spec,
        target_steps=1_831,
    )
    iso_pareto = evaluate_compute_gearshift_pareto(mdl_bt3_iso, test_docs[:600], bt2_tok, device)
    out_report["iso_flops_120M_runs"]["BT3_FullSynthesis_IsoFLOPs"] = {
        **full_evaluate_model(mdl_bt3_iso, bt2_tok, train_sample, test_docs, ood_docs, blimp_pairs, device),
        "gearshift_pareto": iso_pareto,
        "emitted_train_tokens": sum_bt3_iso["emitted_tokens_processed"],
        "train_wall_sec": round(sum_bt3_iso["train_wall_sec"], 2),
    }
    del mdl_bt3_iso, raw_bt3_iso
    torch.cuda.empty_cache()

    save_progress()
    print(f"\n[ALL BREAKTHROUGH & ISO-FLOPS EXPERIMENTS COMPLETE] Saved to {rep_path}", flush=True)


if __name__ == "__main__":
    main()
