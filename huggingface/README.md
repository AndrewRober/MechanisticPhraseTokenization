---
language:
  - en
license: mit
library_name: pytorch
tags:
  - tokenization
  - superword
  - bpe
  - pretraining
  - mechanistic-interpretability
  - test-time-compute
datasets:
  - HuggingFaceFW/fineweb-edu
  - roneneldan/TinyStories
  - nyu-mll/blimp
  - allenai/ai2_arc
  - allenai/sciq
metrics:
  - accuracy
---

# BT3-FullSynthesis (`30.4M` & `149.6M`) + `BT2_AttnDemux_16384` Tokenizer

Official model checkpoints, `16,384`-entry multi-word tokenizers, and evaluation artifacts for **"What Earns a Phrase Its Own Token? Mechanistic Multi-Word Vocabulary Allocation, Compositional-Residual Folding, and Test-Time Compute Gearshifting."**

## Model & Tokenizer Highlights

- **Zero Deployment Overhead (`0.0` Float Folding Error):** Trained with **Foldable Compositional-Residual Superwords** ($W_P = \frac{1}{|P|}\sum_{t\in P} W_t + \Delta_P$, decoupled $\lambda_\Delta = 0.30$), then algebraically folded into a standard `[16384, d]` embedding matrix—requiring **zero architectural modifications** at inference.
- **Attention-Query De-Multiplexed Vocabulary (`BT2_AttnDemux_16384`):** Shares a lossless `14,336` Byte-BPE base + `2,048` boundary-guarded superwords (`96%+` active slot utilization vs. `62.8%` in `SuperBPE`), vetoing high-JSD multi-binding relational phrases ($D_{\text{bind}} > 0.0549$ bits).
- **Zero-Retraining Prefix-Cone Decoding:** Cures Superword Vote-Splitting (`+0.93%` to `+1.59%` next-word recovery with `0.00%` perturbation to subword baselines).
- **Test-Time Compute Gearshifting:** Trained with `20%` dual-granularity unmerging ($q = 0.20$), allowing a **single frozen checkpoint** to run `+5.8%` faster in Superword High Gear or downshift to Subword Low Gear on high-entropy clauses for **`+1.58%` higher accuracy** at `149.6M` scale without Chain-of-Thought tokens.

## Benchmark Summary (`300M` Unique Non-Repeated `FineWeb-Edu` Tokens)

| Model | Params | Bytes/Tok | `BLiMP` (Fert-Matched) ↑ | `ARC-Easy` ↑ | Next-Word (Cone) ↑ | Perturb $\Delta\text{BPB}$ ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `B1_ByteBPE` (3-seed) | `30.42M` | `4.322` | `85.89% ± 0.45%` | `38.13%` | `23.27%` | `+0.00019` |
| `B4_SuperBPE` (3-seed) | `30.42M` | `4.665` | `83.51% ± 0.73%` | `38.53%` | `16.22%` | `+0.00016` |
| `B5_SupraTok` (3-seed) | `30.42M` | `4.422` | `83.64% ± 1.80%` | `37.27%` | `18.75%` | `+0.00058` |
| **`BT3_FullSynthesis` (3-seed)** | **`30.42M`** | **`4.571`** | **`86.44% ± 0.31%` (#1)** | `37.93%` | **`22.83%`** | **`-0.00021` (#1)** |
| `150M_B1_ByteBPE` | `149.58M` | `4.322` | `86.20%` | `38.40%` | `25.66%` | `-0.00017` |
| `150M_B4_SuperBPE` | `149.58M` | `4.665` | `86.39%` | `38.80%` | `18.88%` | `+0.00005` |
| `150M_B5_SupraTok` | `149.58M` | `4.422` | `86.56%` | `38.40%` | `21.94%` | `+0.00042` |
| **`150M_BT3_FullSynthesis` (Zero-Shot)** | **`149.58M`** | **`4.571`** | **`86.81%` (#1 All)** | **`39.20%` (#1)** | **`24.07%`** | **`-0.00020` (#1)** |

## Usage

```python
import torch
from src.config import TokenizerSpec
from src.model import ResearchTransformerLM
from src.tokenizer_core import PhraseAugmentedTokenizer
from src.breakthroughs import PrefixConeDecoder
from scripts.run_stage_cde_mechanistic import STAGE_E_150M_SPEC

tok = PhraseAugmentedTokenizer.load("tokenizers/BT2_AttnDemux_16384", spec=TokenizerSpec())
model = ResearchTransformerLM(STAGE_E_150M_SPEC).cuda().eval()
ckpt = torch.load("runs/stageE_150M_BT3_FullSynthesis_seed1001/ckpt_final.pt", map_location="cuda")
model.load_state_dict(ckpt["model"])

cone = PrefixConeDecoder(tok, device=torch.device("cuda"))
ids = torch.tensor([[tok.eos_id] + tok.encode("In the midst of", add_eos=False)], device="cuda")
with torch.no_grad():
    logits, _ = model(ids, active_output_vocab=tok.vocab_size)
    next_id = cone.argmax_step(logits[:, -1, :])
print(tok.decode(next_id.tolist()))
```

## Author & Citation

**Andrew Rober Fouad Samy** ([GitHub](https://github.com/AndrewRober) · [LinkedIn](https://www.linkedin.com/in/andrew-rober/) · `andrew.rober@bytesglue.com`)

```bibtex
@article{samy2026phrase,
  title={What Earns a Phrase Its Own Token? Mechanistic Multi-Word Vocabulary Allocation, Compositional-Residual Folding, and Test-Time Compute Gearshifting},
  author={Samy, Andrew Rober Fouad},
  year={2026},
  url={https://github.com/AndrewRober/MechanisticPhraseTokenization}
}
```
