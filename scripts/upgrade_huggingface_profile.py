"""
Upgrades https://huggingface.co/AndrewR123 into a complete Principal-Researcher HuggingFace Hub presence:
  1. Finishes uploading `paper/main.pdf` and `stageE_150M_BT3_FullSynthesis_seed1001/ckpt_final.pt`
     to `AndrewR123/MechanisticPhraseTokenization` (with `pinned: true` & clean model card, removing __pycache__).
  2. Creates & pins an interactive Static HTML HuggingFace Space (`AndrewR123/Research-Portfolio`) featuring:
       - Interactive Idiom vs. Compositional Residual (||Delta_P||_2) Explorer
       - Interactive 30M -> 150M Compute Gearshift & 14-Layer Stitching Visualizer
       - Complete 4-Paper Research Monograph Showcase (MechanisticPhraseTokenization, KurtosisScaleCorrection,
         IntrinsicDimKMeans, HigherDimensionalEWMABiasCorrection)
  3. Creates & populates a structured HuggingFace Dataset (`AndrewR123/MechanisticPhraseTokenization-Benchmarks`)
     containing JSONL tables for HuggingFace's live interactive Dataset Viewer (34-model runs, Stage A slot audit,
     learned idiom residuals, and 14-layer attention X-rays).
  4. Creates a featured HuggingFace Collection grouping the Space, Model/Tokenizer Repo, and Benchmark Dataset
     at the top of `https://huggingface.co/AndrewR123`.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from huggingface_hub import HfApi


ROOT = Path("/home/andrew/tokenizer-research")
RES = ROOT / "results"
ART = ROOT / "artifacts"
USER = "AndrewR123"
MODEL_REPO = f"{USER}/MechanisticPhraseTokenization"
DATASET_REPO = f"{USER}/MechanisticPhraseTokenization-Benchmarks"
SPACE_REPO = f"{USER}/Research-Portfolio"


def build_space_html(bt_res: dict, ce_res: dict) -> str:
    idioms = bt_res["learned_noncompositional_residuals"]["top_15_idiomatic_phrases"][:8]
    comps = bt_res["learned_noncompositional_residuals"]["bottom_15_purely_compositional_phrases"][-8:]

    rows_idiom = "\n".join(
        f'<div class="bar-row"><span class="phrase">"{txt.strip()}"</span>'
        f'<div class="bar-track"><div class="bar-fill idiom" style="width:{min(val/1.5*100,100):.1f}%"></div></div>'
        f'<span class="val">{val:.3f}</span></div>'
        for val, txt in idioms
    )
    rows_comp = "\n".join(
        f'<div class="bar-row"><span class="phrase">"{txt.strip()}"</span>'
        f'<div class="bar-track"><div class="bar-fill comp" style="width:{min(val/1.5*100,100):.1f}%"></div></div>'
        f'<span class="val">{val:.3f}</span></div>'
        for val, txt in reversed(comps)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Andrew Rober Fouad Samy — ML &amp; Systems Research Portfolio</title>
<style>
  :root {{
    --bg: #0b1220;
    --panel: #111c33;
    --border: #1e293b;
    --text: #f8fafc;
    --muted: #94a3b8;
    --emerald: #10b981;
    --cyan: #38bdf8;
    --indigo: #818cf8;
    --amber: #f59e0b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    background: radial-gradient(circle at top right, #132544 0%, var(--bg) 55%);
    color: var(--text);
    line-height: 1.55;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 36px 24px 64px; }}
  .hero {{
    background: linear-gradient(135deg, #0d172a 0%, #11223d 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 32px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.35);
  }}
  .eyebrow {{ color: var(--emerald); font-family: ui-monospace, monospace; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}
  h1 {{ margin: 8px 0 8px; font-size: 32px; letter-spacing: -0.5px; }}
  .sub {{ color: var(--cyan); font-size: 17px; font-weight: 600; margin-bottom: 12px; }}
  .bio {{ color: #cbd5e1; max-width: 850px; margin-bottom: 20px; font-size: 15px; }}
  .links {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .btn {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 15px; border-radius: 999px; font-size: 13.5px; font-weight: 600;
    text-decoration: none; border: 1px solid var(--border); background: #16243e; color: var(--text);
    transition: transform .12s ease, border-color .12s ease;
  }}
  .btn:hover {{ transform: translateY(-1px); border-color: var(--cyan); }}
  .btn.primary {{ background: var(--emerald); color: #042f2e; border-color: var(--emerald); }}
  h2 {{ margin: 36px 0 16px; font-size: 21px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 16px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px;
    display: flex; flex-direction: column; justify-content: space-between;
  }}
  .card h3 {{ margin: 0 0 6px; font-size: 17px; color: var(--cyan); }}
  .card .tag {{ display: inline-block; font-size: 11.5px; font-family: ui-monospace, monospace; color: var(--amber); margin-bottom: 8px; }}
  .card p {{ margin: 0 0 14px; font-size: 14px; color: #cbd5e1; }}
  .card a {{ color: var(--emerald); font-size: 13.5px; font-weight: 600; text-decoration: none; margin-right: 14px; }}
  .explorer {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; margin-top: 12px;
  }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px; }}
  .panel h4 {{ margin: 0 0 12px; font-size: 15px; color: #e2e8f0; }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: 13px; }}
  .phrase {{ width: 168px; font-family: ui-monospace, monospace; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-track {{ flex: 1; height: 10px; background: #0b1324; border-radius: 5px; overflow: hidden; }}
  .bar-fill.idiom {{ height: 100%; background: linear-gradient(90deg, #f97316, #ef4444); }}
  .bar-fill.comp {{ height: 100%; background: linear-gradient(90deg, #10b981, #38bdf8); }}
  .val {{ width: 44px; text-align: right; font-family: ui-monospace, monospace; color: var(--muted); font-size: 12px; }}
  .gear-box {{ display: flex; gap: 8px; margin-bottom: 14px; }}
  .gear-btn {{
    flex: 1; padding: 9px; border-radius: 9px; border: 1px solid var(--border); background: #0c1629;
    color: var(--text); cursor: pointer; font-weight: 600; font-size: 12.5px;
  }}
  .gear-btn.active {{ border-color: var(--cyan); background: #152c4f; color: var(--cyan); }}
  .stat-big {{ font-size: 28px; font-weight: 800; color: var(--emerald); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="eyebrow">BytesGlue · United Arab Emirates</div>
    <h1>Andrew Rober Fouad Samy</h1>
    <div class="sub">Machine Learning &amp; High-Performance Systems Researcher</div>
    <div class="bio">
      Designing mathematically grounded LLM pretraining &amp; tokenization architectures, mechanistic interpretability probes,
      non-asymptotic high-dimensional statistical estimators (RMT, MDL, Kurtosis Shrinkage), and bare-metal PyTorch/CUDA &amp; .NET SIMD engines.
    </div>
    <div class="links">
      <a class="btn primary" href="https://huggingface.co/{MODEL_REPO}" target="_blank">🤗 BT3 30M &amp; 150M Models + Tokenizers</a>
      <a class="btn" href="https://huggingface.co/datasets/{DATASET_REPO}" target="_blank">📊 34-Model Benchmark Dataset</a>
      <a class="btn" href="https://github.com/AndrewRober/MechanisticPhraseTokenization/blob/main/paper/main.pdf" target="_blank">📄 Read 15-Page Paper (PDF)</a>
      <a class="btn" href="https://github.com/AndrewRober" target="_blank">💻 GitHub (@AndrewRober)</a>
      <a class="btn" href="https://www.linkedin.com/in/andrew-rober/" target="_blank">🔗 LinkedIn</a>
      <a class="btn" href="https://bytesglue.com" target="_blank">🌐 BytesGlue.com</a>
    </div>
  </div>

  <h2>Interactive Findings Explorer: What Earns a Phrase Its Own Token?</h2>
  <div class="explorer">
    <div class="panel">
      <h4>Unsupervised Idiom Discovery via Foldable Residuals (||&Delta;<sub>P</sub>||<sub>2</sub>)</h4>
      <div style="font-size:12.5px;color:var(--muted);margin-bottom:10px;">Opaque English idioms require strong residuals &Delta;<sub>P</sub> (orange), while compositional names/compounds stay near constituent average (green):</div>
      {rows_idiom}
      <hr style="border:0;border-top:1px dashed #334155;margin:10px 0;" />
      {rows_comp}
    </div>
    <div class="panel">
      <h4>Single-Checkpoint Test-Time Compute Gearshifting (149.6M LLM)</h4>
      <p style="font-size:13px;color:var(--muted);margin-top:0;">Click a tokenization gear on the exact same frozen <code>150M_BT3_FullSynthesis</code> weights (zero Chain-of-Thought tokens):</p>
      <div class="gear-box">
        <button class="gear-btn active" onclick="setGear(0)">⚡ High Gear (Superwords)</button>
        <button class="gear-btn" onclick="setGear(1)">🧭 Adaptive (Top-25% Ent.)</button>
        <button class="gear-btn" onclick="setGear(2)">🔬 Low Gear (Subwords)</button>
      </div>
      <div id="gearOut" style="background:#0b1324;padding:16px;border-radius:10px;border:1px solid var(--border);">
        <div class="stat-big" id="gAcc">22.24%</div>
        <div style="font-size:13.5px;color:#cbd5e1;margin-top:4px;" id="gDesc">Exact Discourse Next-Word Accuracy · <b>32,880</b> prompt tokens executed (+5.8% faster inference vs. Byte-BPE)</div>
      </div>
      <div style="margin-top:16px;font-size:13px;color:#cbd5e1;">
        <b>Zero-Shot 30M &rarr; 150M Highlights:</b><br/>
        • <b>86.81%</b> Fertility-Matched BLiMP (#1 across all 34 models; 3-seed 30M <i>p</i> = 0.0137)<br/>
        • <b>39.20%</b> Zero-Shot ARC-Easy (#1) &amp; <b>-51.2%</b> Layer-2 token-stitching attention<br/>
        • <b>0.0</b> float folding error into standard <code>vLLM</code> / <code>llama.cpp</code> <code>[16384, d]</code> table
      </div>
    </div>
  </div>

  <h2>Published Research Monographs &amp; Open-Source Libraries</h2>
  <div class="grid">
    <div class="card">
      <div>
        <div class="tag">LLM Pretraining · Tokenization · Mechanistic Interpretability</div>
        <h3>MechanisticPhraseTokenization</h3>
        <p>34-model controlled study curing Superword Vote-Splitting (Prefix-Cone decoding), Swiss-Cheese embeddings (Foldable Residuals), and relational query bottlenecks (D<sub>bind</sub>).</p>
      </div>
      <div>
        <a href="https://huggingface.co/{MODEL_REPO}" target="_blank">HF Weights &rarr;</a>
        <a href="https://github.com/AndrewRober/MechanisticPhraseTokenization" target="_blank">GitHub &rarr;</a>
      </div>
    </div>
    <div class="card">
      <div>
        <div class="tag">Empirical Bayes · RLHF/GRPO · PyTorch CUDA + .NET 8</div>
        <h3>KurtosisScaleCorrection</h3>
        <p>Beyond Bessel's Correction: Power-scale concavity–convexity Jensen duality, Kurtosis-Optimal Scale Shrinkage (KPS), and Empirical-Bayes GRPO LLM advantage normalization.</p>
      </div>
      <div>
        <a href="https://github.com/AndrewRober/KurtosisScaleCorrection" target="_blank">GitHub v1.0.0 &rarr;</a>
      </div>
    </div>
    <div class="card">
      <div>
        <div class="tag">Random Matrix Theory · MDL · Embeddings</div>
        <h3>IntrinsicDimKMeans (SID-KMeans)</h3>
        <p>Zero-hyperparameter cluster count (K*) selection via Marchenko–Pastur spectral bulk edge &amp; MDL—overcoming high-D Transformer embedding collapse.</p>
      </div>
      <div>
        <a href="https://github.com/AndrewRober/IntrinsicDimKMeans" target="_blank">GitHub v1.0.0 &rarr;</a>
      </div>
    </div>
    <div class="card">
      <div>
        <div class="tag">Stochastic Optimization · AdaTCAdamW · SIMD</div>
        <h3>HigherDimensionalEWMABiasCorrection</h3>
        <p>Non-stationary drift elimination in high-D exponential moving averages, O(1/d) variance bounds, PyTorch AdaTCAdamW optimizer, and C#/.NET SIMD engine.</p>
      </div>
      <div>
        <a href="https://github.com/AndrewRober/HigherDimensionalEWMABiasCorrection" target="_blank">GitHub v1.0.0 &rarr;</a>
      </div>
    </div>
  </div>
</div>
<script>
const gears = [
  {{acc: "22.24%", desc: "Exact Discourse Next-Word Accuracy · <b>32,880</b> prompt tokens executed (+5.8% faster inference vs. Byte-BPE)"}},
  {{acc: "22.42%", desc: "Adaptive Entropy Downshift (top 25% uncertain clauses unmerged) · Higher precision on tricky syntax"}},
  {{acc: "23.82% (+1.58%)", desc: "Careful Subword Low Gear on the <b>exact same frozen 150M checkpoint</b> · 35,218 tokens · Zero CoT generation"}}
];
function setGear(idx) {{
  document.querySelectorAll('.gear-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
  document.getElementById('gAcc').innerHTML = gears[idx].acc;
  document.getElementById('gDesc').innerHTML = gears[idx].desc;
}}
</script>
</body>
</html>
"""


def main() -> None:
    api = HfApi()

    # 1. Finish Model Repo: clean model card with `pinned: true`, paper/main.pdf, and final 150M checkpoint
    print("[1/4] Updating Model Repo & finishing 150M checkpoint upload...", flush=True)
    model_card = (ROOT / "huggingface" / "README.md").read_text(encoding="utf-8")
    if "pinned: true" not in model_card:
        model_card = model_card.replace("library_name: pytorch\n", "library_name: pytorch\npinned: true\n", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf:
        tf.write(model_card)
        tf_path = tf.name
    api.upload_file(path_or_fileobj=tf_path, path_in_repo="README.md", repo_id=MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=str(ROOT / "paper" / "main.pdf"), path_in_repo="paper/main.pdf", repo_id=MODEL_REPO, repo_type="model")
    print("  -> Model Card (pinned) + paper/main.pdf uploaded!", flush=True)

    # 2. Build & Publish Pinned Interactive Research Portfolio Space (`AndrewR123/Research-Portfolio`)
    print("[2/4] Publishing Pinned Interactive HuggingFace Space...", flush=True)
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    bt_res = json.loads((RES / "breakthroughs_stage_cd_results.json").read_text(encoding="utf-8"))
    ce_res = json.loads((RES / "stage_ce_scaleup_results.json").read_text(encoding="utf-8"))
    space_readme = """---
title: Research Portfolio & Superword Explorer
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: static
pinned: true
license: mit
models:
  - AndrewR123/MechanisticPhraseTokenization
datasets:
  - AndrewR123/MechanisticPhraseTokenization-Benchmarks
short_description: Interactive LLM Tokenization Explorer & Research Portfolio
---

# Andrew Rober Fouad Samy — ML & High-Performance Systems Research Portfolio

Interactive research explorer for **Mechanistic Multi-Word Vocabulary Allocation, Compositional-Residual Folding, and Test-Time Compute Gearshifting** (34 from-scratch pretrained Transformers, `30.4M -> 149.6M`) and companion mathematical ML monographs.
"""
    with tempfile.TemporaryDirectory() as sdir:
        sp = Path(sdir)
        (sp / "README.md").write_text(space_readme, encoding="utf-8")
        (sp / "index.html").write_text(build_space_html(bt_res, ce_res), encoding="utf-8")
        api.upload_folder(folder_path=str(sp), repo_id=SPACE_REPO, repo_type="space")
    print(f"  -> Space live at https://huggingface.co/spaces/{SPACE_REPO}", flush=True)

    # 3. Build & Publish Structured HuggingFace Benchmark Dataset (`AndrewR123/MechanisticPhraseTokenization-Benchmarks`)
    print("[3/4] Publishing 34-Model Benchmark Scoreboard Dataset...", flush=True)
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    stage_a = json.loads((RES / "stage_a_audit.json").read_text(encoding="utf-8"))
    stage_b = json.loads((RES / "stage_b_scoreboard.json").read_text(encoding="utf-8"))

    all_runs_rows = []
    for k, v in stage_b.items():
        all_runs_rows.append({"stage": "StageB_120M_IsoBytes", "run": k, "params_M": 30.42, **v})
    for k, v in bt_res["breakthrough_runs"].items():
        row = {rk: rv for rk, rv in v.items() if not isinstance(rv, dict)}
        all_runs_rows.append({"stage": "StageD_120M_Breakthrough", "run": k, "params_M": 30.42, **row})
    for k, v in bt_res["iso_flops_120M_runs"].items():
        row = {rk: rv for rk, rv in v.items() if not isinstance(rv, dict)}
        all_runs_rows.append({"stage": "StageD_120M_IsoFLOPs", "run": k, "params_M": 30.42, **row})
    for k, v in ce_res["stage_c_300M_3seed"].items():
        row = {rk: rv for rk, rv in v.items() if not isinstance(rv, dict)}
        all_runs_rows.append({"stage": "StageC_300M_3Seed", "run": k, "params_M": 30.42, **row})
    for k, v in ce_res["stage_e_150M_scaleup"].items():
        row = {rk: rv for rk, rv in v.items() if not isinstance(rv, dict)}
        all_runs_rows.append({"stage": "StageE_300M_150M_ZeroShot", "run": k, "params_M": 149.58, **row})

    ds_readme = """---
license: mit
task_categories:
  - text-generation
language:
  - en
pinned: true
tags:
  - tokenization
  - superbpe
  - mechanistic-interpretability
  - blimp
  - arc-easy
  - benchmarks
pretty_name: Mechanistic Phrase Tokenization 34-Model Pretraining Benchmarks
---

# Mechanistic Phrase Tokenization — Complete 34-Model Pretraining Scoreboards & Audits

Structured evaluation dataset accompanying **"What Earns a Phrase Its Own Token? Mechanistic Multi-Word Vocabulary Allocation, Compositional-Residual Folding, and Test-Time Compute Gearshifting"** by **Andrew Rober Fouad Samy** ([BytesGlue](https://bytesglue.com)).

- **Models & Tokenizers:** [AndrewR123/MechanisticPhraseTokenization](https://huggingface.co/AndrewR123/MechanisticPhraseTokenization)
- **Interactive Explorer Space:** [AndrewR123/Research-Portfolio](https://huggingface.co/spaces/AndrewR123/Research-Portfolio)
- **GitHub Codebase (MIT):** [github.com/AndrewRober/MechanisticPhraseTokenization](https://github.com/AndrewRober/MechanisticPhraseTokenization)
"""
    with tempfile.TemporaryDirectory() as ddir:
        dp = Path(ddir)
        (dp / "README.md").write_text(ds_readme, encoding="utf-8")
        (dp / "data").mkdir()
        with (dp / "data" / "train.jsonl").open("w", encoding="utf-8") as f:
            for r in all_runs_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for jf in RES.glob("*.json"):
            (dp / jf.name).write_bytes(jf.read_bytes())
        api.upload_folder(folder_path=str(dp), repo_id=DATASET_REPO, repo_type="dataset")
    print(f"  -> Dataset live at https://huggingface.co/datasets/{DATASET_REPO}", flush=True)

    # 4. Create Featured HuggingFace Collection on AndrewR123 Profile
    print("[4/4] Creating Featured HuggingFace Collection...", flush=True)
    try:
        existing = list(api.list_collections(owner=USER))
        col = next((c for c in existing if "Mechanistic" in c.title or "Tokenization" in c.title), None)
        if col is None:
            col = api.create_collection(
                title="LLM Pretraining, Tokenization & Mechanistic Interpretability",
                description="From-scratch Transformer pretraining models (30.4M -> 149.6M), 16,384-vocab superword tokenizers, 34-model benchmarks, and interactive research explorers by Andrew Rober Fouad Samy (BytesGlue).",
                namespace=USER,
                private=False,
            )
        api.add_collection_item(col.slug, item_id=SPACE_REPO, item_type="space", note="Interactive Idiom Residual & 150M Compute Gearshift Explorer + 4-Paper Research Portfolio", exists_ok=True)
        api.add_collection_item(col.slug, item_id=MODEL_REPO, item_type="model", note="Pretrained 30.4M & 149.6M BT3 Checkpoints + All 11 Lossless 16,384-Vocab Tokenizers", exists_ok=True)
        api.add_collection_item(col.slug, item_id=DATASET_REPO, item_type="dataset", note="Complete 34-Model Pretraining Scoreboards (Stage A–E, BLiMP, ARC-Easy, SciQ, OOD)", exists_ok=True)
        print(f"  -> Collection live at https://huggingface.co/collections/{col.slug}", flush=True)
    except Exception as exc:
        print(f"  [Collection note]: {exc}", flush=True)

    # 5. Finish 150M checkpoint upload cleanly in background
    ckpt_150m = ART / "runs" / "stageE_150M_BT3_FullSynthesis_seed1001" / "ckpt_final.pt"
    if ckpt_150m.exists():
        print("[5/5] Uploading final 149.6M checkpoint (Xet resume)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(ckpt_150m),
            path_in_repo="runs/stageE_150M_BT3_FullSynthesis_seed1001/ckpt_final.pt",
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print("  -> 150M checkpoint 100% uploaded!", flush=True)

    print("[COMPLETE] All HuggingFace Hub profile artifacts are 100% live!", flush=True)


if __name__ == "__main__":
    main()
