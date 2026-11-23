"""
Generates a clean Markdown & JSON research summary table from `stage_b_scoreboard.json`
and `stage_a_audit.json`, ranked by Byte-Matched Test BPB, Contextual Information Gain,
Fertility-Matched BLiMP Accuracy, TinyStories OOD Transfer, and Perturbation Stability.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT_DIR = Path(os.environ.get("TOK_ROOT", Path(__file__).resolve().parent.parent))
ARTIFACTS_DIR = Path(
    os.environ.get(
        "TOK_ARTIFACTS_DIR",
        ROOT_DIR / ("results" if (ROOT_DIR / "results" / "stage_b_scoreboard.json").exists() else "artifacts"),
    )
)


def main() -> None:
    sb_path = ARTIFACTS_DIR / "stage_b_scoreboard.json"
    if not sb_path.exists():
        print("stage_b_scoreboard.json not found yet.")
        sys.exit(0)

    sb: dict[str, dict] = json.loads(sb_path.read_text(encoding="utf-8"))
    rows = sorted(sb.items(), key=lambda kv: kv[1]["test_bpb_matched_bytes"])

    print("| Method | B/tok | Test BPB (Matched) | Unigram BPB | Info Gain (ΔBPB) | BLiMP (Fert-Matched) | OOD BPB | Perturb ΔBPB |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, m in rows:
        print(
            f"| `{name}` | {m['bytes_per_token']:.3f} | **{m['test_bpb_matched_bytes']:.4f}** | "
            f"{m['unigram_bpb']:.4f} | +{m['contextual_gain_bpb']:.4f} | "
            f"{m['blimp_fertility_matched_acc']*100:.2f}% | {m['ood_tinystories_bpb']:.4f} | "
            f"+{m['perturbation_bpb_degradation']:.4f} |"
        )


if __name__ == "__main__":
    main()
