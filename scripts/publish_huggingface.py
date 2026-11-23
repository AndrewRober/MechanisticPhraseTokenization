"""
One-command publisher to upload tokenizers, JSON scoreboards, paper figures, and (optionally)
selected PyTorch model checkpoints to HuggingFace Hub.

Usage:
    export HF_TOKEN="hf_..."
    python scripts/publish_huggingface.py --repo-id <your-hf-username>/bt3-phrase-tokenization
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish BT3 research artifacts to HuggingFace Hub.")
    parser.add_argument("--repo-id", required=True, help="Target HF repo, e.g. username/bt3-phrase-tokenization")
    parser.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"), help="Optional directory with tokenizers/ and runs/")
    parser.add_argument("--include-checkpoints", action="store_true", help="Also upload final .pt model checkpoints")
    parser.add_argument("--private", action="store_true", help="Create private HF repository")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # 1. Upload HF Model Card
    api.upload_file(
        path_or_fileobj=str(ROOT / "huggingface" / "README.md"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    # 2. Upload verified JSON scoreboards, figures, and source code
    for folder in ("results", "src", "paper/figures"):
        src_p = ROOT / folder
        if src_p.exists():
            api.upload_folder(
                folder_path=str(src_p),
                path_in_repo=folder,
                repo_id=args.repo_id,
                repo_type="model",
            )

    # 3. Upload saved tokenizers & optional checkpoints if present locally or on GX10
    art_dir = Path(args.artifacts_dir)
    tok_dir = art_dir / "tokenizers"
    if tok_dir.exists():
        api.upload_folder(
            folder_path=str(tok_dir),
            path_in_repo="tokenizers",
            repo_id=args.repo_id,
            repo_type="model",
        )

    if args.include_checkpoints and (art_dir / "runs").exists():
        for ckpt_name in (
            "stageCD_BT1_FoldableResidual_seed2026",
            "stageC_IsoFLOPs_BT3_FullSynthesis",
            "stageC_300M_BT3_FullSynthesis_seed1001",
            "stageE_150M_BT3_FullSynthesis_seed1001",
        ):
            ckpt_f = art_dir / "runs" / ckpt_name / "ckpt_final.pt"
            if ckpt_f.exists():
                print(f"Uploading checkpoint {ckpt_name}...")
                api.upload_file(
                    path_or_fileobj=str(ckpt_f),
                    path_in_repo=f"runs/{ckpt_name}/ckpt_final.pt",
                    repo_id=args.repo_id,
                    repo_type="model",
                )

    print(f"[HuggingFace] Published successfully to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
