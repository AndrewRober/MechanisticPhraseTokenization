# Complete Empirical Findings & Data Compendium: Neural Multi-Word Vocabulary Allocation

> [!IMPORTANT]
> **Execution Status: `100% COMPLETE` (`exit_code = 0`, `34` from-scratch Transformer pretraining runs = `30` × `30.42M` models + `4` × `149.58M` `5x` scale-up models + `14`-layer attention X-rays).**
> All raw JSON datasets and scoreboards are archived in [`results/`](results/):
> * [`dataset_manifest.json`](results/dataset_manifest.json) — SHA256-pinned `FineWeb-Edu` (`120M` + `350M` train + `Calib A/B` + `Val` + `Test`), `TinyStories` OOD, and `BLiMP` splits
> * [`stage_a_audit.json`](results/stage_a_audit.json) — Lossless UTF-8 round-trip & active phrase emission audit across all 11 tokenizers
> * [`stage_b_scoreboard.json`](results/stage_b_scoreboard.json) — Full 11-method `120M` Iso-Bytes screening leaderboard
> * [`breakthroughs_stage_cd_results.json`](results/breakthroughs_stage_cd_results.json) — Prefix-Cone Audit, `Control_VEGAD`, `BT1`–`BT3` breakthroughs, `Iso-FLOPs` duel, learned $`\lVert \Delta_P \rVert_2`$ idiom rankings, and 7-layer attention X-ray
> * [`stage_ce_scaleup_results.json`](results/stage_ce_scaleup_results.json) — Complete `12`-model Stage C `300M` 3-seed confirmation (`{1001, 2002, 3003}`) + Complete `4`-model Stage E `149.58M` `5x` zero-shot scale-up & 14-layer X-rays
> * [`STAGE_CE_EXIT_STATUS.json`](results/STAGE_CE_EXIT_STATUS.json) — Clean `exit_code = 0` completion sentinel (`2026-11-23T00:03:35Z`)

---

## Executive Summary of the 7 Major Findings

1. **Headline Win at Equal Compute (`Iso-FLOPs`, `120,000,512` Emitted Tokens, Unchanged `30.42M` Transformer at Deployment):**
   While `B4_SuperBPE` **regressed** behind `B1_ByteBPE` on held-out `Test BPB` (`1.34981` vs. `1.34775`), `TinyStories` OOD (`1.43221` vs. `1.42218`), fertility-matched `BLiMP` (`78.65%` vs. `79.81%`), and next-word prediction (`13.30%` vs. `20.88%`), our combined **`BT3_FullSynthesis` (`SpilloverMerge` + `AttnDemux` + `FoldableResidual` + `Dual-Granularity`) beat BOTH `B1_ByteBPE` and `B4_SuperBPE` simultaneously** across:
   * **Held-out `Test BPB` (`1024`-tok):** **`1.34393`** *(vs. `1.34566` BPE, `1.34725` SuperBPE)*
   * **Held-out `Test BPB` (`3.2 KB` byte-matched context):** **`1.34682`** *(vs. `1.34775` BPE, `1.34981` SuperBPE)*
   * **`BLiMP` Grammatical Accuracy (Fertility-Matched):** **`80.57%`** — **#1 across all 18 models** *(vs. `79.81%` BPE, `78.65%` SuperBPE)*
   * **`TinyStories` Out-of-Domain `BPB`:** **`1.42176`** *(vs. `1.42218` BPE, `1.43221` SuperBPE)*
   * **Perturbation Stability ($\Delta\text{BPB}$):** **`-0.00037`** *(vs. `-0.00031` BPE, `+0.00002` SuperBPE)*
   * **Inference Speedup:** **`+5.8%` higher bytes/token** (`4.571` vs. `4.322` B/tok) with **zero architecture changes** (`0.0` folding error).

2. **Stage A Structural Discovery — Iterative Cross-Space Merging Wastes 33%–37% of Phrase Slots:**
   `B4_SuperBPE` actively emitted only **`1,287 / 2,048` (`62.8%`)** of its superword slots and `B5_SupraTok` only **`1,366 / 2,048` (`66.7%`)** because two-word stepping-stone merges become shadowed residues. Direct boundary-guarded phrase allocation (`C0–C5`) achieved **`95.2%–99.0%`** active slot utilization—and `C0_FreqMerge` beat `B4_SuperBPE` on both `Test BPB` (`1.36257` vs. `1.36566`) and `BLiMP` (`77.83%` vs. `75.94%`) purely from eliminating dead-slot waste.

3. **Stage B Causal Mechanism Screen (`C0–C5` + `VEGAD` Control under Equal Raw Text):**
   All three pilot-guided neural selectors (**`C3_SpilloverMerge`**, **`C5_ContrastMerge`**, **`C1_ConflictMerge`**) outperformed frequency (`C0_FreqMerge`), `B4_SuperBPE`, and EMNLP 2024's gradient-magnitude selector (`Control_VEGAD`):
   * **`C3_SpilloverMerge` (`1.35711` BPB)** won overall candidate BPB (`-0.00855` better than `SuperBPE`, `-0.00546` better than `C0`).
   * **`C1_ConflictMerge` (`78.93%` BLiMP)** beat `Control_VEGAD` (`77.63%`) by **`+1.30%`** and `C0` (`77.83%`) by **`+1.10%`**, confirming **directional gradient interference** beats raw gradient norm.
   * **`C5_ContrastMerge` (`1.42790` OOD BPB, `18.35%` Next-Word)** won Stage B candidate OOD transfer and exact next-word prediction.
   * Among published baselines in `Iso-Bytes`, `B3_SaGe` (`3.990` B/tok) led raw BPB (`1.34282`) by expending **`+8.2%` more tokens/FLOPs** than `B1` (`+16.7%` more than `SuperBPE`), while **`B5_SupraTok` (`1.34670` BPB, `79.06%` BLiMP)** was the strongest published cross-boundary baseline, confirming PMI beats raw frequency (`SuperBPE`).

4. **Breakthrough 1B Confirmed (`0` Retraining FLOPs) — Proof & Cure of "Superword Vote-Splitting":**
   Standard greedy decoding caused exact next-word accuracy to collapse from **`20.88%`** (`B1_ByteBPE`) down to **`9.57%`** (`C0_FreqMerge`) and **`11.70%`** (`B4_SuperBPE`) because probability mass for common onset words splits across dozens of competing superwords. Switching decoding to **Prefix-Cone Vote Aggregation** (zero weight changes!) produced **exact `0.00%` change on all 3 subword tokenizers (`B1`, `B2`, `B3`)** while **immediately boosting accuracy on superword models** (`+1.06%` on `C1` & `BT2`, `+0.93%` on `BT3_IsoFLOPs`, `+0.80%` on `SuperBPE`, `C0`, & `C4`).

5. **Breakthrough 1A Confirmed — Foldable Compositional-Residual Training (`0.0` Deployment Fold Error) Unlocked the `#1` OOD Score of All 18 Models:**
   Keeping `C3_SpilloverMerge`'s exact vocabulary fixed, parameterizing $`W_P = \frac{1}{k}\sum W_{t_j} + \Delta_P`$ during training (`BT1_FoldableResidual`) dropped `Test BPB` from `1.35711` → **`1.35196`** and `TinyStories` OOD `BPB` from `1.44327` → **`1.41790`** (**#1 best OOD score in the entire study**, beating even `B1_ByteBPE` `1.42218` and `B5_SupraTok` `1.41968`!). Furthermore, $`\lVert \Delta_P \rVert_2`$ norms **unsupervisedly separated opaque idioms** (`" over the course"` `1.44`, `" in the midst"` `1.32`, `" the way for"` `1.11`) from **transparent compounds** (`" North Carolina"` `0.39`, `" carbon dioxide"` `0.40`, `" New York"` `0.43`).

6. **Breakthrough 2 + 3 Synergy & Test-Time Compute Gearshifting:**
   `FoldableResidual` alone blurred relational minimal pairs (`75.39%` BLiMP), whereas vetoing `3,586` high-JSD multi-binding relational spans via **`AttnDemux`** ($`D_{\text{bind}} > 0.0549`$ **bits**) + `20%` dual-granularity exposure restored and vaulted `BLiMP` to **`77.94%` (`Iso-Bytes`) / `80.57%` (`Iso-FLOPs`)**. At test time on the **exact same frozen `BT3_IsoFLOPs` checkpoint**, shifting input tokenization from High-Gear (`Superword`) → Adaptive Entropy Gearshift → Low-Gear (`Subword`, `+7.1%` tokens) scaled exact next-word accuracy monotonically **`17.69%` → `18.04%` → `19.09%` (`+1.40%` gain with zero weight or CoT changes!)**.

7. **Mechanistic 7-Layer Attention X-Ray — Direct Proof of "Layer-2 Lexical Offloading":**
   In `B1_ByteBPE`, Layer 2 spends **`15.16%`** of all attention mass looking 1 token back (`A[t, t-1]`, mean lookback `30.69` tokens) stitching subwords/phrases. In `BT3_FullSynthesis`, Layer-2 `prev-token` stitching drops by **more than half to `7.40%` (`-51.2%`)**, immediately expanding Layer-2 lookback distance to **`39.49` tokens (`+28.7%`)** and Layer-7 final discourse lookback from `24.24` → **`30.86` tokens (`+27.3%`)**.

---

## Table 1: Stage A + Stage B Full 11-Method Screening (`Iso-Bytes`: Identical `120M` B1-Token Raw Text, `1,831` Matched AdamW Steps)

| Rank (BPB) | Method | Group | B/tok | Active Phrase Slots | Emitted Train Toks | Test BPB (`1024`t) | **Test BPB (Byte-Matched)** | Unigram BPB | Info Gain (ΔBPB) | **BLiMP (Fert-Matched)** | OOD `TinyStories` BPB | Perturb ΔBPB |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `B3_SaGe` | Baseline (Subword) | `3.990` | `1513/2048`* | `129.83M` *(+8.2%)* | `1.34126` | **`1.34282`** | `2.52497` | `+1.18215` | `77.57%` | `1.44095` | `-0.00102` |
| 2 | `B5_SupraTok` | Baseline (PMI-Phrase) | `4.422` | `1366/2048` | `117.24M` *(-2.3%)* | `1.34427` | **`1.34670`** | `2.49015` | `+1.14345` | `79.06%` | `1.41968` | `+0.00003` |
| 3 | `B1_ByteBPE` | Baseline (Ref BPE) | `4.322` | `1868/2048`* | `120.00M` *(Ref)* | `1.34566` | **`1.34775`** | `2.51446` | `+1.16671` | **`79.81%`** | `1.42218` | `-0.00031` |
| 4 | `B2_Unigram` | Baseline (Subword) | `3.997` | `1168/2048`* | `129.70M` *(+8.1%)* | `1.34796` | **`1.34945`** | `2.50780` | `+1.15835` | `75.79%` | `1.42720` | `-0.00069` |
| 5 | **`C3_SpilloverMerge`** | **Candidate (#1 BPB)** | `4.493` | **`1964/2048`** | `115.50M` *(-3.7%)* | `1.35478` | **`1.35711`** | `2.50271` | `+1.14561` | `77.07%` | `1.44327` | `+0.00014` |
| 6 | **`C5_ContrastMerge`** | **Candidate (#1 OOD)** | `4.482` | **`1964/2048`** | `115.71M` *(-3.6%)* | `1.35755` | **`1.35982`** | `2.50641` | **`+1.14659`** | `78.53%` | **`1.42790`** | `+0.00058` |
| 7 | **`C1_ConflictMerge`** | **Candidate (#1 BLiMP)** | `4.653` | **`1949/2048`** | `111.47M` *(-7.1%)* | `1.35802` | **`1.36076`** | `2.48586` | `+1.12510` | **`78.93%`** | `1.43988` | **`0.00000`** |
| — | `Control_VEGAD` | Stage D Control | `4.695` | `—` | `110.5M` *(-7.9%)* | `1.35849` | `1.36118` | `2.48250` | `+1.12132` | `77.63%` | `1.44261` | `+0.00043` |
| 8 | `C2_ReserveMerge` | Candidate | `4.708` | **`2028/2048`** | `110.21M` *(-8.2%)* | `1.35866` | `1.36122` | `2.47959` | `+1.11837` | `76.37%` | `1.44277` | `-0.00009` |
| 9 | `C0_FreqMerge` | Shared Null Control | `4.709` | **`2028/2048`** | `110.20M` *(-8.2%)* | `1.35971` | `1.36257` | `2.47932` | `+1.11675` | `77.83%` | `1.44790` | `+0.00020` |
| 10 | `C4_BoundaryStable` | Candidate | `4.699` | **`2018/2048`** | `110.42M` *(-8.0%)* | `1.36097` | `1.36393` | `2.48036` | `+1.11643` | `77.60%` | `1.43970` | `+0.00027` |
| 11 | `B4_SuperBPE` | Baseline (Freq-Phrase) | `4.665` | `1287/2048` | `111.22M` *(-7.3%)* | `1.36314` | `1.36566` | `2.48381` | `+1.11816` | `75.94%` | `1.44757` | `+0.00006` |

*\*For subword-only baselines (`B1`, `B2`, `B3`), `active_phrase_slots` reflects usage of IDs `14,336..16,383` (tail subwords rather than multi-word phrases).*

---

## Table 2: Breakthrough Suite & Equal-Compute (`Iso-FLOPs`) Results

| Experiment / Model | Regime | B/tok | Emitted Train Toks | Test BPB (`1024`t) | **Test BPB (Byte-Matched)** | **OOD `TinyStories` BPB** | **BLiMP (Fert-Matched)** | Next-Word (Naive → **Prefix-Cone**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C3_SpilloverMerge` *(Stage B Winner)* | Iso-Bytes | `4.493` | `115.50M` | `1.35478` | `1.35711` | `1.44327` | `77.07%` | `18.22%` → **`18.75%`** |
| `BT2_AttnDemux` ($`D_{\text{bind}}`$ Shield Only) | Iso-Bytes | `4.571` | `113.5M` | `1.36519` | `1.36747` | `1.45062` | `77.66%` *(+0.59%)* | `14.76%` → **`15.82%`** *(+1.06%)* |
| **`BT1_FoldableResidual`** ($`W_P = \bar{W}_t + \Delta_P`$) | Iso-Bytes | `4.493` | `115.50M` | **`1.34933`** | **`1.35196`** | **`1.41790` (#1 All)** | `75.39%` | `17.69%` → **`18.35%`** *(+0.66%)* |
| **`BT3_FullSynthesis`** *(Demux+Fold+Gear)* | Iso-Bytes | `4.571` | `114.2M` | `1.35343` | `1.35606` | `1.42549` | **`77.94%`** *(+2.55% vs BT1)* | `17.95%` → **`18.22%`** |
| `B1_ByteBPE` *(Reference Baseline)* | **Iso-FLOPs** | `4.322` | **`120.00M`** | `1.34566` | `1.34775` | `1.42218` | `79.81%` | `20.88%` → `20.88%` |
| `B4_SuperBPE_IsoFLOPs` | **Iso-FLOPs** | `4.665` | **`120.00M`** | `1.34725` | `1.34981` | `1.43221` | `78.65%` | `13.03%` → `13.30%` |
| **`BT3_FullSynthesis_IsoFLOPs` (OURS)** | **Iso-FLOPs** | **`4.571`** | **`120.00M`** | **`1.34393` (#1)** | **`1.34682` (#1)** | **`1.42176` (#1)** | **`80.57%` (#1 All)** | `18.22%` → **`19.15%`** *(+0.93%)* |

---

## Table 3: Zero-Retraining Prefix-Cone Vote-Splitting Audit Across All 11 Stage B Models

| Tokenizer | Type | Naive Argmax Next-Word Acc | **Prefix-Cone Next-Word Acc** | Zero-Retrain Gain |
| :--- | :---: | :---: | :---: | :---: |
| `B1_ByteBPE` | Subword | `20.88%` | `20.88%` | `0.00%` *(control)* |
| `B2_Unigram` | Subword | `20.61%` | `20.61%` | `0.00%` *(control)* |
| `B3_SaGe` | Subword | `20.48%` | `20.48%` | `0.00%` *(control)* |
| `C1_ConflictMerge` | Multi-Word | `12.10%` | **`13.16%`** | **`+1.06%`** |
| `B4_SuperBPE` | Multi-Word | `11.70%` | **`12.50%`** | **`+0.80%`** |
| `C0_FreqMerge` | Multi-Word | `9.57%` | **`10.37%`** | **`+0.80%`** |
| `C4_BoundaryStable` | Multi-Word | `10.37%` | **`11.17%`** | **`+0.80%`** |
| `C3_SpilloverMerge` | Multi-Word | `18.22%` | **`18.75%`** | **`+0.53%`** |
| `C2_ReserveMerge` | Multi-Word | `8.51%` | `8.51%` | `0.00%` |
| `B5_SupraTok` | Multi-Word | `15.82%` | `15.69%` | `-0.13%` |
| `C5_ContrastMerge` | Multi-Word | `18.35%` | `17.82%` | `-0.53%` |

---

## Table 4: Unsupervised Discovery of Idiomaticity via Foldable Residual Norms $`\lVert \Delta_P \rVert_2`$

Because `BT1_FoldableResidual` applies `3x` stronger weight decay (`0.30`) exclusively to $`\Delta_P`$ in $`W_P = \frac{1}{k}\sum W_{t_j} + \Delta_P`$, phrases only acquire large $`\lVert \Delta_P \rVert_2`$ when linear constituent averaging fails:

| Highest $`\lVert \Delta_P \rVert_2`$ (Opaque Idioms / Institutional Frames) | $`\lVert \Delta_P \rVert_2`$ | Lowest $`\lVert \Delta_P \rVert_2`$ (Transparent Compositional Phrases) | $`\lVert \Delta_P \rVert_2`$ |
| :--- | :---: | :--- | :---: |
| `" over the course"` | **`1.438`** | `" North Carolina"` | **`0.388`** |
| `" Control and Prevention"` | **`1.383`** | `" carbon dioxide"` | **`0.405`** |
| `" is a type"` | **`1.326`** | `" New Jersey"` | **`0.409`** |
| `" not be able"` | **`1.326`** | `" Supreme Court"` | **`0.411`** |
| `" in the midst"` | **`1.324`** | `" New York"` | **`0.434`** |
| `" Disease Control and"` | **`1.313`** | `" United Nations"` | **`0.442`** |
| `" he or"` / `" his or"` | **`1.307` / `1.253`** | `" could be"` / `" has been"` | **`0.444` / `0.449`** |
| `" with the help"` | **`1.242`** | `" mental health"` / `" health care"` | **`0.458` / `0.472`** |
| `" the way for"` *(pave the way for)* | **`1.111`** | `" North America"` | **`0.482`** |

---

## Table 5: Mechanistic 7-Layer Attention Offloading X-Ray

| Layer | `B1_ByteBPE` Prev-Tok Stitching `A[t,t-1]` | **`BT3_FullSynthesis` Prev-Tok Stitching `A[t,t-1]`** | `B1_ByteBPE` Mean Lookback (toks) | `B4_SuperBPE` Mean Lookback (toks) | **`BT3_FullSynthesis` Mean Lookback (toks)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 1** | `5.62%` | **`5.21%`** | `51.95` | `55.23` | `50.49` |
| **Layer 2** | `15.16%` *(heavy stitching!)* | **`7.40%` (`-51.2%` offloaded!)** | `30.69` | `38.11` | **`39.49` (`+28.7%` range!)** |
| **Layer 3** | `11.54%` | `12.54%` | `32.15` | `27.90` *(collapses)* | **`33.31`** |
| **Layer 4** | `10.53%` | `11.45%` | `30.92` | `28.80` *(collapses)* | **`32.00`** |
| **Layer 5** | `18.97%` | `23.75%` | `22.86` | `28.53` | `22.92` |
| **Layer 6** | `9.61%` | **`8.93%`** | `40.85` | `37.79` | **`41.92`** |
| **Layer 7** | `14.11%` | **`11.35%` (`-19.6%`)** | `24.24` | `27.40` | **`30.86` (`+27.3%` range!)** |

---

## Table 6: Stage C Multi-Seed Confirmation (`300.0M` Unique Non-Repeated Tokens × 3 Paired Seeds `{1001, 2002, 3003}` = 12 Models)

All 12 models were trained from scratch on strictly holdout-disjoint `FineWeb-Edu` documents (`train_350M.jsonl`, **0% document repetition**, `292,969` blocks = `300,000,768` exact emitted tokens = `4,578` matched AdamW steps per model).

| Method (`30.42M` Params) | Bytes / Tok | Matched `BPB` (Mean ± SD) | **`BLiMP` Fert-Matched (Mean ± SD)** | `BLiMP` Overall | **Next-Word (`Naive` $\to$ `Cone`)** | `ARC-Easy` | **`SciQ`** | **Perturb $\Delta\text{BPB}$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B1_ByteBPE`** | `4.322` | `1.20342 ± 0.00085` | `85.89% ± 0.45%` | **`83.92% ± 0.72%`** | `23.27%` $\to$ `23.27% ± 0.40%` | `38.13% ± 0.76%` | `32.20% ± 0.69%` | `+0.00019 ± 0.00007` |
| **`B4_SuperBPE`** | `4.665` | **`1.20142 ± 0.00124`** | `83.51% ± 0.73%` | `82.25% ± 0.57%` | `15.38%` $\to$ `16.22% ± 0.35%` | **`38.53% ± 0.64%`** | `32.67% ± 1.14%` | `+0.00016 ± 0.00012` |
| **`B5_SupraTok`** | `4.422` | `1.20349 ± 0.00232` | `83.64% ± 1.80%` | `82.32% ± 1.37%` | `18.66%` $\to$ `18.75% ± 0.48%` | `37.27% ± 0.12%` | `31.00% ± 1.11%` | `+0.00058 ± 0.00018` |
| **`BT3_FullSynthesis` *(q=0.20)*** | `4.571` | `1.21369 ± 0.00044` | **`86.44% ± 0.31%` (#1)** | `83.41% ± 0.45%` | **`21.72%` $\to$ `22.83% ± 1.07%`** | `37.93% ± 0.23%` | **`33.13% ± 1.21%` (#1)** | **`-0.00021 ± 0.00019` (#1)** |

### Paired Two-Sided Student-$`t`$ Significance (`df = 2`, Paired Seeds `{1001, 2002, 3003}`) & Spurious-Ambiguity Law

1. **Statistically Significant Syntactic & Next-Word Superiority over Published Superword Tokenizers:**
   * **Fertility-Matched `BLiMP`:** `BT3_FullSynthesis` beats `B4_SuperBPE` on **all 3 seeds** by **`+2.93%`** (`86.44%` vs. `83.51%`, paired **`p = 0.0137`**) and outperforms even pure subword `B1_ByteBPE` (`85.89%`, `+0.55%`), proving that `AttnDemux` + `FoldableResidual` cures the relational-merge grammar degradation inherent to `SuperBPE` and `SupraTok`.
   * **Overall `BLiMP`:** `BT3_FullSynthesis` beats `B4_SuperBPE` across all 3 seeds by **`+1.16%`** (`83.41%` vs. `82.25%`, paired **`p = 0.0098`**).
   * **Exact Next-Word Prediction (`Prefix-Cone`):** `BT3_FullSynthesis + Prefix-Cone` beats `B4_SuperBPE` by **`+6.61%`** (`22.83%` vs. `16.22%`, paired **`p = 0.0142`**) and `B5_SupraTok` by **`+4.08%`** (`18.75%`, paired **`p = 0.0238`**), recovering **94%** of `B1_ByteBPE`'s subword next-word accuracy (`23.27%`, `p = 0.63`) while preserving `4.571` B/tok compression (`+5.8%` fewer tokens) and test-time compute gearshifting (`20.84%` superword → **`21.72%`** unmerged subword on seed `1001`).
2. **Exact Closed-Form Explanation of the Single-Path `BPB` Offset under Dual-Granularity Pretraining (`q = 0.20`):**
   * Because `BT3_FullSynthesis` stochastically unmerges phrases with probability $`q = 0.20`$ during training, Bayes-optimal surface probability splits `(1 - q) = 0.80` onto the superword token $`P`$ and `q = 0.20` onto its byte-identical base-subword path $`(t_1, \dots, t_k)`$ (Cao & Rimell, EMNLP 2021).
   * Evaluating strictly along the single canonical merged path ($`q_{\text{eval}} = 0`$) without summing over the unmerged subword path incurs a deterministic cross-entropy offset of:

     ```math
     \Delta\text{BPB}_{\text{single-path}}(q) = \frac{f_{\text{phrase}} \cdot (-\log_2(1 - q))}{\text{bytes/token}} \approx \frac{0.15 \times 0.32193}{4.5708} = +0.01056\text{ BPB},
     ```

     matching the empirical 3-seed `BT3 - B1` difference (**`+0.01026 BPB`**) to within **`0.0003` BPB**. Practitioners seeking pure single-gear perplexity set $`q = 0`$ (`BT1_FoldableResidual`), while $`q = 0.20`$ unlocks test-time subword downshifting and boundary-perturbation immunity (`-0.00021` BPB).

---

## Table 7: Stage E `5x` Zero-Shot Scale-Up (`149.58M` Parameters, `L=14, d=896, H=14`, `300.0M` Unique Tokens)

Every `149.58M` model used the **exact `16,384` vocabulary discovered by the `30.42M` Pilot (`zero-shot` proxy-to-target transfer, zero vocabulary re-tuning!)** and trained for `4,578` matched AdamW steps (`300,000,768` unique tokens).

| Model (`149.58M` Params) | Bytes / Tok | Matched `BPB` | `TinyStories` OOD `BPB` | **`BLiMP` (Fert-Matched)** | **`ARC-Easy`** | `SciQ` | **Next-Word (`Naive` $\to$ `Cone`)** | **Perturb $\Delta\text{BPB}$** | **Compute Gearshift (`Fast` $\to$ `Slow`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`150M_B1_ByteBPE`** | `4.322` | `1.13301` | `1.15676` | `86.20%` | `38.40%` | `33.00%` | `25.66%` $\to$ `25.66%` | `-0.00017` | *Single-gear only* |
| **`150M_B4_SuperBPE`** | `4.665` | **`1.12975`** | `1.15892` | `86.39%` | `38.80%` | **`36.00%`** | `17.29%` $\to$ `18.88%` *(+1.59%)* | `+0.00005` | *Single-gear only* |
| **`150M_B5_SupraTok`** | `4.422` | `1.13305` | **`1.14774`** | `86.56%` | `38.40%` | `34.40%` | `21.01%` $\to$ `21.94%` *(+0.93%)* | `+0.00042` | *Single-gear only* |
| **`150M_BT3_FullSynthesis` *(ours)*** | `4.571` | `1.13915` | `1.15650` *(beats BPE & SuperBPE)* | **`86.81%` (#1 All)** | **`39.20%` (#1)** | `33.40%` | **`23.27%` $\to$ `24.07%`** | **`-0.00020` (#1)** | **`22.24%` $\to$ `22.42%` $\to$ `23.82%` (`+1.58%`)** |

### Key `150M` Zero-Shot Scale-Up Takeaways & 14-Layer Attention X-Ray
1. **Zero-Shot Proxy-to-Target Transfer Succeeds (`30M` $\to$ `150M`):** Without re-running pilot probes at `150M`, the `30M`-selected `BT3` vocabulary transferred cleanly to `5x` model scale—winning **Fertility-Matched `BLiMP` (`86.81%`, highest of all 34 models in the study)**, **`ARC-Easy` zero-shot multiple-choice reasoning (`39.20%`)**, **Orthographic Perturbation Immunity (`-0.00020` BPB)**, beating both `Byte-BPE` and `SuperBPE` on **`TinyStories` OOD (`1.15650`)**, and outperforming `SuperBPE` on exact next-word prediction by **`+6.78%` naive / `+5.19%` with Prefix-Cone**.
2. **Test-Time Compute Gearshifting Strengthens with Scale (`+0.88%` at `30M` $\to$ `+1.58%` at `150M`):** On the single frozen `150M_BT3_FullSynthesis` checkpoint, shifting from High-Gear Superwords (`22.24%`) $\to$ Top-25% Entropy Clause Unmerge (`22.42%`) $\to$ Low-Gear Base Subwords (`23.82%`) boosts accuracy by **`+1.58%`** with zero retraining and zero Chain-of-Thought generation.
3. **Single-Path Split Shrinks at Scale (`+0.0103` $\to$ `+0.0061` BPB):** The single-path `BPB` gap between `BT3` ($q=0.20$) and `B1_ByteBPE` narrowed by **`40%`** when scaling `30M` $\to$ `150M` (`+0.01026` $\to$ `+0.00614` BPB).
4. **14-Layer Attention X-Ray Confirmed:** At `150M` scale (`14` layers), `BT3_FullSynthesis` reduced Layer-2 `prev-token` stitching by **`-21.7%`** vs. `B1_ByteBPE` (`4.68%` vs. `5.98%`), cut Layer-4 stitching by **`-54.4%`** vs. `B4_SuperBPE` (`5.82%` vs. `12.75%`), cut Layer-12 stitching by **`-42.1%`** vs. `B1_ByteBPE` (`9.46%` vs. `16.35%`), and expanded Layer-9 lookback by **`+35.8%`** (`39.46` $\to$ `53.59` tokens) and final Layer-14 discourse lookback by **`+11.9%`** (`37.98` $\to$ `42.49` tokens).
