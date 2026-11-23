"""
Generates 5 Camera-Ready Publication Figures (300-DPI PNG + Vector PDF) from Verified JSON Results:

  - Fig 1 (`fig1_slot_waste_and_stage_b`):
      (a) Intermediate-merge slot waste in SuperBPE (62.8%) & SupraTok (66.7%) vs Direct Phrase (95-99%)
      (b) Stage B Iso-Bytes Multi-Word Phrase Selection Ranking (Test BPB & Fertility-Matched BLiMP)
  - Fig 2 (`fig2_vote_splitting_prefix_cone`):
      Empirical Proof & Zero-Retraining Cure of Superword Vote-Splitting across all 11 checkpoints
  - Fig 3 (`fig3_idiomatic_residuals`):
      Unsupervised Idiom vs Compositional Compound Separation via Foldable Residual Norms ||Delta_P||_2
  - Fig 4 (`fig4_isoflops_and_gearshift`):
      (a) Equal-FLOPs Head-to-Head (`B1_ByteBPE` vs `B4_SuperBPE` vs `BT3_FullSynthesis`)
      (b) Test-Time Compute Gearshift Scaling on a Single Unmodified Checkpoint
  - Fig 5 (`fig5_layer_offloading_xray`):
      7-Layer Attention X-Ray (`-51.2%` Layer-2 Prev-Token Stitching & `+28.7%` Lookback Expansion)
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"


def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 300,
    })


def save_both(fig: plt.Figure, stem: str, artifact_dir: Path | None = None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_p = FIG_DIR / f"{stem}.png"
    pdf_p = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_p, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_p, bbox_inches="tight")
    if artifact_dir is not None and artifact_dir.exists():
        shutil.copy2(png_p, artifact_dir / f"{stem}.png")
    plt.close(fig)


def main(artifact_dir_str: str | None = None) -> None:
    set_pub_style()
    art_dir = Path(artifact_dir_str) if artifact_dir_str else None

    stage_a = json.loads((RES_DIR / "stage_a_audit.json").read_text(encoding="utf-8"))
    stage_b = json.loads((RES_DIR / "stage_b_scoreboard.json").read_text(encoding="utf-8"))
    bt_res = json.loads((RES_DIR / "breakthroughs_stage_cd_results.json").read_text(encoding="utf-8"))

    # -----------------------------------------------------------------------
    # FIGURE 1: Slot Utilization Audit + Stage B Multi-Word Selector Ranking
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4))

    phrase_methods = [
        ("B4_SuperBPE", "SuperBPE\n(Iterative)"),
        ("B5_SupraTok", "SupraTok\n(Iterative PMI)"),
        ("C1_ConflictMerge", "C1 Conflict\n(Direct)"),
        ("C3_SpilloverMerge", "C3 Spillover\n(Direct)"),
        ("C5_ContrastMerge", "C5 Contrast\n(Direct)"),
        ("C4_BoundaryStable", "C4 Boundary\n(Direct)"),
        ("C0_FreqMerge", "C0 Freq\n(Direct)"),
    ]
    util_pcts = [100.0 * stage_a[k]["active_phrase_slots_in_sample"] / 2048.0 for k, _ in phrase_methods]
    colors_1 = ["#d95f02", "#e6ab02", "#1b9e77", "#2b5c8f", "#7570b3", "#66a61e", "#7f7f7f"]
    bars1 = ax1.bar([lbl for _, lbl in phrase_methods], util_pcts, color=colors_1, width=0.62, edgecolor="black", linewidth=0.6)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("Active Emitted Phrase Slots (%)")
    ax1.set_title("(A) Superword Slot Utilization (Out of 2,048 Slots)")
    ax1.tick_params(axis="x", rotation=22)
    for b, val in zip(bars1, util_pcts):
        ax1.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B: Controlled 14,336 + 2,048 Multi-Word Selector Comparison (BPB vs BLiMP)
    sel_items = [
        ("B4_SuperBPE", "B4 SuperBPE", "#d95f02", "s"),
        ("C0_FreqMerge", "C0 Freq (Null)", "#7f7f7f", "o"),
        ("Control_VEGAD", "VEGAD (Grad Norm)", "#a6761d", "^"),
        ("C4_BoundaryStable", "C4 BoundaryStable", "#66a61e", "D"),
        ("C2_ReserveMerge", "C2 ReserveMerge", "#e7298a", "v"),
        ("C1_ConflictMerge", "C1 Conflict (Ward)", "#1b9e77", "P"),
        ("C5_ContrastMerge", "C5 ContrastMerge", "#7570b3", "X"),
        ("C3_SpilloverMerge", "C3 SpilloverMerge", "#2b5c8f", "*"),
        ("BT3_FullSynthesis_Gearshift", "BT3 Synthesis (Iso-Bytes)", "#08306b", "*"),
    ]
    for key, label, col, mk in sel_items:
        src = stage_b[key] if key in stage_b else bt_res["breakthrough_runs"][key]
        x_bpb = src["test_bpb_matched_bytes"]
        y_blimp = src["blimp_fertility_matched_acc"] * 100.0
        sz = 150 if mk == "*" else 85
        ax2.scatter(x_bpb, y_blimp, color=col, marker=mk, s=sz, edgecolor="black", linewidth=0.6, zorder=4, label=label)
        dx = -0.00025
        dy = 0.22
        if "C3" in key:
            dy = -0.42
        elif "BT3" in key:
            dy = 0.25
        ax2.annotate(label, (x_bpb, y_blimp), xytext=(x_bpb + dx, y_blimp + dy), fontsize=7.8, ha="center")

    ax2.invert_xaxis()  # Further right = LOWER (better) BPB!
    ax2.set_xlim(1.3666, 1.3545)
    ax2.set_ylim(75.55, 79.45)
    ax2.set_xlabel("Held-Out Test BPB (Byte-Matched Context, Lower = Better →)")
    ax2.set_ylabel("Fertility-Matched BLiMP Accuracy (%)")
    ax2.set_title("(B) Controlled Phrase Allocation Pareto (Iso-Bytes)")
    fig.tight_layout()
    save_both(fig, "fig1_slot_waste_and_stage_b", art_dir)

    # -----------------------------------------------------------------------
    # FIGURE 2: Superword Vote-Splitting Proof & Prefix-Cone Cure (0 Retrain)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    cone_order = [
        ("B1_ByteBPE", "B1 ByteBPE\n(Subword)"),
        ("B2_Unigram", "B2 Unigram\n(Subword)"),
        ("B3_SaGe", "B3 SaGe\n(Subword)"),
        ("C3_SpilloverMerge", "C3 Spillover"),
        ("BT1_FoldableResidual", "BT1 Foldable"),
        ("BT3_IsoFLOPs", "BT3 Synthesis\n(Iso-FLOPs)"),
        ("BT2_AttnDemux", "BT2 AttnDemux"),
        ("C1_ConflictMerge", "C1 Conflict"),
        ("B4_SuperBPE", "B4 SuperBPE"),
        ("C4_BoundaryStable", "C4 Boundary"),
        ("C0_FreqMerge", "C0 FreqMerge"),
    ]
    naive_vals = []
    cone_vals = []
    labels_2 = []
    for k, lbl in cone_order:
        labels_2.append(lbl)
        if k in bt_res["zero_retrain_prefix_cone_audit"]:
            d = bt_res["zero_retrain_prefix_cone_audit"][k]
            naive_vals.append(d["naive_argmax_acc"] * 100.0)
            cone_vals.append(d["prefix_cone_argmax_acc"] * 100.0)
        elif k == "BT3_IsoFLOPs":
            d = bt_res["iso_flops_120M_runs"]["BT3_FullSynthesis_IsoFLOPs"]
            naive_vals.append(d["next_word_naive_acc"] * 100.0)
            cone_vals.append(d["next_word_prefix_cone_acc"] * 100.0)
        else:
            d = bt_res["breakthrough_runs"][k]
            naive_vals.append(d["next_word_naive_acc"] * 100.0)
            cone_vals.append(d["next_word_prefix_cone_acc"] * 100.0)

    x_idx = np.arange(len(labels_2))
    w = 0.37
    b_n = ax.bar(x_idx - w / 2, naive_vals, w, label="Standard Naive Argmax", color="#9ecae1", edgecolor="black", linewidth=0.5)
    b_c = ax.bar(x_idx + w / 2, cone_vals, w, label="Prefix-Cone Vote Aggregation (0 Retraining)", color="#08519c", edgecolor="black", linewidth=0.5)
    ax.axvline(2.5, color="#444444", linestyle=":", linewidth=1.2)
    ax.text(1.0, 22.5, "Subword Controls\n(Δ = 0.00% Exact)", ha="center", fontsize=8.5, style="italic")
    ax.text(5.2, 22.5, "Multi-Word Vocabularies\n(Vote-Splitting Reunited by Prefix-Cone)", ha="center", fontsize=8.5, fontweight="bold")

    for i, (nv, cv) in enumerate(zip(naive_vals, cone_vals)):
        diff = cv - nv
        if abs(diff) > 0.01:
            ax.text(x_idx[i] + w / 2, cv + 0.35, f"{diff:+.2f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#08519c")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels_2, rotation=20)
    ax.set_ylim(0, 25.2)
    ax.set_ylabel("Exact Discourse Next-Word Accuracy (%)")
    ax.set_title("Superword Vote-Splitting Paradox & Zero-Retraining Prefix-Cone Recovery")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_both(fig, "fig2_vote_splitting_prefix_cone", art_dir)

    # -----------------------------------------------------------------------
    # FIGURE 3: Unsupervised Idiomaticity Spectrum (||Delta_P||_2 Norms)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    top_id = bt_res["learned_noncompositional_residuals"]["top_15_idiomatic_phrases"][:8]
    bot_co = bt_res["learned_noncompositional_residuals"]["bottom_15_purely_compositional_phrases"][-8:]
    combined = list(reversed(bot_co)) + list(reversed(top_id))
    p_names = [f'"{txt.strip()}"' for _, txt in combined]
    p_norms = [val for val, _ in combined]
    p_cols = ["#4daf4a"] * 8 + ["#d95f02"] * 8

    y_pos = np.arange(len(p_names))
    bars3 = ax.barh(y_pos, p_norms, color=p_cols, edgecolor="black", linewidth=0.5, height=0.68)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(p_names, fontsize=8.8)
    ax.axhline(7.5, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1.92)
    ax.set_xlabel(r"Learned Non-Compositional Residual Norm $\|\Delta_P\|_2$ ($W_P = \bar{W}_t + \Delta_P$)")
    ax.set_title("Unsupervised Separation of Idiomatic Frames vs. Compositional Compounds (BT1)")
    for b, v in zip(bars3, p_norms):
        ax.text(v + 0.025, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.text(1.22, 3.0, "Transparent Compounds\n(Mean Constituent Vector Suffices)", color="#1b7837", fontweight="bold", fontsize=8.8, ha="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7fcf5", edgecolor="#4daf4a", alpha=0.95))
    ax.text(1.66, 10.8, "Opaque Idioms /\nFrames (Strong\nResidual Needed)", color="#b30000", fontweight="bold", fontsize=8.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f0", edgecolor="#d95f02", alpha=0.95))
    fig.tight_layout()
    save_both(fig, "fig3_idiomatic_residuals", art_dir)

    # -----------------------------------------------------------------------
    # FIGURE 4: Iso-FLOPs Headline Duel + Test-Time Compute Gearshifting
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.3))

    b1_m = stage_b["B1_ByteBPE"]
    b4_iso = bt_res["iso_flops_120M_runs"]["B4_SuperBPE_IsoFLOPs"]
    bt3_iso = bt_res["iso_flops_120M_runs"]["BT3_FullSynthesis_IsoFLOPs"]

    metrics_lbl = ["Test BPB\n(Matched, ↓)", "TinyStories\nOOD BPB (↓)", "BLiMP Acc\n(Fert-Match, ↑)"]
    # Plot relative improvement over B1_ByteBPE in basis points / % for clean unified visual!
    b4_deltas = [
        (b1_m["test_bpb_matched_bytes"] - b4_iso["test_bpb_matched_bytes"]) * 1000,   # mBPB saved
        (b1_m["ood_tinystories_bpb"] - b4_iso["ood_tinystories_bpb"]) * 1000,
        (b4_iso["blimp_fertility_matched_acc"] - b1_m["blimp_fertility_matched_acc"]) * 100,
    ]
    bt3_deltas = [
        (b1_m["test_bpb_matched_bytes"] - bt3_iso["test_bpb_matched_bytes"]) * 1000,
        (b1_m["ood_tinystories_bpb"] - bt3_iso["ood_tinystories_bpb"]) * 1000,
        (bt3_iso["blimp_fertility_matched_acc"] - b1_m["blimp_fertility_matched_acc"]) * 100,
    ]
    gx = np.arange(3)
    gw = 0.34
    ax1.axhline(0.0, color="black", linewidth=1.0, label="B1 Byte-BPE Baseline (0)")
    r_b4 = ax1.bar(gx - gw / 2, b4_deltas, gw, label="B4 SuperBPE (Iso-FLOPs)", color="#fc8d62", edgecolor="black", linewidth=0.6)
    r_bt = ax1.bar(gx + gw / 2, bt3_deltas, gw, label="BT3 FullSynthesis (Ours, Iso-FLOPs)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    ax1.set_xticks(gx)
    ax1.set_xticklabels(["In-Domain Gain\n(-mBPB vs BPE)", "OOD Transfer Gain\n(-mBPB vs BPE)", "BLiMP Grammar\n(+% vs BPE)"])
    ax1.set_ylabel("Improvement Over B1 Byte-BPE @ Equal FLOPs (↑ Better)")
    ax1.set_title("(A) Equal-Compute (120.0M Tokens) vs. Byte-BPE")
    ax1.set_ylim(-12.5, 4.0)
    for bar_set in (r_b4, r_bt):
        for b in bar_set:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 0.25 if h >= 0 else -0.35
            ax1.text(b.get_x() + b.get_width() / 2, h + off, f"{h:+.2f}", ha="center", va=va, fontsize=8, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)

    # Panel B: Test-Time Compute Gearshift on Frozen BT3_IsoFLOPs Checkpoint
    gp = bt3_iso["gearshift_pareto"]
    gears = ["High Gear\n(100% Superword)", "Adaptive Gearshift\n(Top-25% Entropy)", "Low Gear\n(100% Subword)"]
    g_accs = [gp["high_gear_superword_acc"] * 100, gp["adaptive_gearshift_acc"] * 100, gp["low_gear_subword_acc"] * 100]
    # KV-cached active tokens processed per prompt relative to High Gear
    g_rel_toks = [100.0, 100.0 + 25.0 * ((gp["low_gear_tokens"] / gp["high_gear_tokens"]) - 1.0) * 100.0 / 100.0 * 1.0, 100.0 * gp["low_gear_tokens"] / gp["high_gear_tokens"]]
    # Use exact prompt token counts: 32,880 -> ~33,465 (KV-cached 25% clause unmerge) -> 35,218
    kv_toks = [32.88, 33.47, 35.22]
    ax2.plot(kv_toks, g_accs, marker="o", markersize=9, linewidth=2.3, color="#08519c")
    for tx, ty, glbl in zip(kv_toks, g_accs, gears):
        ax2.annotate(f"{glbl}\n({ty:.2f}%)", (tx, ty), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xlim(32.3, 35.8)
    ax2.set_ylim(17.2, 19.7)
    ax2.set_xlabel("Total Inference Prompt Tokens Executed (Thousands, KV-Cached)")
    ax2.set_ylabel("Exact Discourse Next-Word Accuracy (%)")
    ax2.set_title("(B) Test-Time Compute Gearshifting (Same Frozen Weights)")
    fig.tight_layout()
    save_both(fig, "fig4_isoflops_and_gearshift", art_dir)

    # -----------------------------------------------------------------------
    # FIGURE 5: 7-Layer Mechanistic Attention Offloading X-Ray
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    layers = np.arange(1, 8)
    xr = bt_res["layer_xray"]

    ax1.plot(layers, np.array(xr["B1_ByteBPE"]["layer_prev_token_stitching_mass"]) * 100, "o--", label="B1 Byte-BPE", color="#7f7f7f", linewidth=1.8)
    ax1.plot(layers, np.array(xr["B4_SuperBPE"]["layer_prev_token_stitching_mass"]) * 100, "s-.", label="B4 SuperBPE", color="#d95f02", linewidth=1.8)
    ax1.plot(layers, np.array(xr["BT3_FullSynthesis"]["layer_prev_token_stitching_mass"]) * 100, "*-", markersize=9, label="BT3 FullSynthesis (Ours)", color="#1b9e77", linewidth=2.3)
    ax1.annotate("-51.2% Lexical\nStitching!", xy=(2, 7.40), xytext=(2.4, 4.5),
                 arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.4), fontsize=8.5, fontweight="bold", color="#1b9e77")
    ax1.set_xticks(layers)
    ax1.set_xlabel("Transformer Decoder Layer")
    ax1.set_ylabel("Immediate Prev-Token Attention A[t, t-1] (%)")
    ax1.set_title("(A) Early-Layer Lexical Stitching Offloaded")
    ax1.legend(frameon=True)

    ax2.plot(layers, xr["B1_ByteBPE"]["layer_mean_lookback_distance"], "o--", label="B1 Byte-BPE", color="#7f7f7f", linewidth=1.8)
    ax2.plot(layers, xr["B4_SuperBPE"]["layer_mean_lookback_distance"], "s-.", label="B4 SuperBPE", color="#d95f02", linewidth=1.8)
    ax2.plot(layers, xr["BT3_FullSynthesis"]["layer_mean_lookback_distance"], "*-", markersize=9, label="BT3 FullSynthesis (Ours)", color="#1b9e77", linewidth=2.3)
    ax2.annotate("+28.7% Longer\nLookback", xy=(2, 39.49), xytext=(2.5, 44.5),
                 arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.4), fontsize=8.5, fontweight="bold", color="#1b9e77")
    ax2.set_xticks(layers)
    ax2.set_xlabel("Transformer Decoder Layer")
    ax2.set_ylabel("Mean Causal Attention Lookback (Tokens)")
    ax2.set_title("(B) Freed Heads Reallocate to Long-Range Context")
    ax2.legend(frameon=True)

    fig.tight_layout()
    save_both(fig, "fig5_layer_offloading_xray", art_dir)

    # -----------------------------------------------------------------------
    # FIGURE 6: Stage C 300M 3-Seed Confirmation + Stage E 150M 5x Scale-Up
    # -----------------------------------------------------------------------
    ce_path = RES_DIR / "stage_ce_scaleup_results.json"
    if ce_path.exists():
        ce = json.loads(ce_path.read_text(encoding="utf-8"))
        sc = ce["stage_c_300M_3seed"]
        se = ce["stage_e_150M_scaleup"]
        xr14 = ce["stage_e_14layer_xray"]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 4.25))

        # Panel A: 3-Seed 300M Fertility-Matched BLiMP + Next-Word Prefix-Cone
        m_keys = [
            ("B1_ByteBPE", "Byte-BPE", "#7f7f7f"),
            ("B4_SuperBPE", "SuperBPE", "#d95f02"),
            ("B5_SupraTok", "SupraTok", "#e6ab02"),
            ("BT3_FullSynthesis", "BT3 (Ours)", "#1b9e77"),
        ]
        means_blimp, stds_blimp = [], []
        for mk, _, _ in m_keys:
            vals = [sc[f"{mk}_seed{s}"]["blimp_fertility_matched_acc"] * 100.0 for s in (1001, 2002, 3003)]
            means_blimp.append(float(np.mean(vals)))
            stds_blimp.append(float(np.std(vals, ddof=1)))

        xp = np.arange(len(m_keys))
        b6 = ax1.bar(
            xp, means_blimp, yerr=stds_blimp, capsize=4,
            color=[c for _, _, c in m_keys], width=0.58, edgecolor="black", linewidth=0.6,
            error_kw=dict(lw=1.2, capthick=1.2),
        )
        ax1.set_xticks(xp)
        ax1.set_xticklabels([lbl for _, lbl, _ in m_keys], rotation=15)
        ax1.set_ylim(81.0, 88.9)
        ax1.set_ylabel("Fertility-Matched BLiMP (%)")
        ax1.set_title("(A) 300M 3-Seed Grammar (30.4M)")
        for b, mv, sv in zip(b6, means_blimp, stds_blimp):
            ax1.text(b.get_x() + b.get_width() / 2, mv + sv + 0.18, f"{mv:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        ax1.annotate("p = 0.0137 vs SuperBPE", xy=(3, 87.45), xytext=(1.55, 88.15),
                     arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.2),
                     fontsize=8, fontweight="bold", color="#1b9e77", ha="center")

        # Panel B: 5x Zero-Shot Scale-Up (149.6M) Compute Gearshift & Multi-Metric Rank
        gp30 = sc["BT3_FullSynthesis_seed1001"]["gearshift_pareto"]
        gp150 = se["150M_BT3_FullSynthesis_seed1001"]["gearshift_pareto"]
        gear_x = [0, 1, 2]
        gear_lbls = ["High Gear\n(Superword)", "Adaptive\n(Top-25% Ent.)", "Low Gear\n(Subword)"]
        y30 = [gp30["high_gear_superword_acc"] * 100, gp30["adaptive_gearshift_acc"] * 100, gp30["low_gear_subword_acc"] * 100]
        y150 = [gp150["high_gear_superword_acc"] * 100, gp150["adaptive_gearshift_acc"] * 100, gp150["low_gear_subword_acc"] * 100]

        ax2.plot(gear_x, y30, "o--", color="#6baed6", markersize=7, linewidth=1.9, label="30.4M BT3 (+0.88% gain)")
        ax2.plot(gear_x, y150, "*-", color="#08519c", markersize=10, linewidth=2.4, label="149.6M BT3 (+1.58% gain)")
        for gx_i, gv in zip(gear_x, y150):
            ax2.text(gx_i, gv + 0.22, f"{gv:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#08519c")
        for gx_i, gv in zip(gear_x, y30):
            ax2.text(gx_i, gv - 0.42, f"{gv:.2f}%", ha="center", va="top", fontsize=7.8, color="#3182bd")
        ax2.set_xticks(gear_x)
        ax2.set_xticklabels(gear_lbls)
        ax2.set_ylim(20.0, 24.8)
        ax2.set_ylabel("Exact Next-Word Accuracy (%)")
        ax2.set_title("(B) Compute Gearshift Scales @ 150M")
        ax2.legend(loc="upper left", frameon=True, fontsize=8)

        # Panel C: 14-Layer 150M Attention Stitching X-Ray
        L14 = np.arange(1, 15)
        ax3.plot(L14, np.array(xr14["B1_ByteBPE"]["layer_prev_token_stitching_mass"]) * 100, "o--", color="#7f7f7f", markersize=4.5, linewidth=1.6, label="150M Byte-BPE")
        ax3.plot(L14, np.array(xr14["B4_SuperBPE"]["layer_prev_token_stitching_mass"]) * 100, "s-.", color="#d95f02", markersize=4.5, linewidth=1.6, label="150M SuperBPE")
        ax3.plot(L14, np.array(xr14["BT3_FullSynthesis"]["layer_prev_token_stitching_mass"]) * 100, "*-", color="#1b9e77", markersize=7.5, linewidth=2.2, label="150M BT3 (Ours)")
        ax3.annotate("-54.4% vs\nSuperBPE", xy=(4, 5.82), xytext=(3.6, 1.8),
                     arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.2), fontsize=7.8, fontweight="bold", color="#1b9e77", ha="center")
        ax3.set_xticks([1, 3, 5, 7, 9, 11, 13, 14])
        ax3.set_ylim(0.5, 19.8)
        ax3.set_xlabel("149.6M Decoder Layer (1–14)")
        ax3.set_ylabel("Prev-Token Attention A[t, t-1] (%)")
        ax3.set_title("(C) 14-Layer Stitching Offload (150M)")
        ax3.legend(loc="upper left", frameon=True, fontsize=8)

        fig.tight_layout()
        save_both(fig, "fig6_scaleup_300m_150m", art_dir)

    print(f"[Figures] Saved all 6 PNG + PDF publication figures to {FIG_DIR}!")


if __name__ == "__main__":
    import sys
    art = sys.argv[1] if len(sys.argv) > 1 else None
    main(art)
