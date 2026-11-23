"""
Controlled High-Throughput Pretraining Loop (`ResearchTrainer`).

Upgrades for NVIDIA GB10 maximum throughput:
  1. Fused in-graph `cross_entropy(..., reduction="sum", return_logits=False)` inside `torch.compile`
     (elides writing 1.07 GB BF16 logits to LPDDR5x memory every micro-batch).
  2. Quantized `micro_batch_size` step boundaries (eliminates 30% remainder-row padding waste while
     maintaining 100% static `[micro_bs, 1024]` shapes, exact `total_steps`, and exact corpus coverage).
  3. Zero `.item()` CPU-GPU synchronization stalls inside the training step loop.
  4. Accepts pre-packed `blocks` array so 20-core Grace CPU can pre-tokenize all runs in parallel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Optional, Sequence

import numpy as np
import torch

from src.baselines import TokenizerInterface
from src.config import ModelSpec, TrainSpec
from src.eval_suite import evaluate_bpb_and_stats
from src.model import ResearchTransformerLM


def get_cosine_lr(step: int, total_steps: int, peak_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


def pack_documents_to_blocks(
    docs: Sequence[str],
    tokenizer: TokenizerInterface,
    seq_len: int = 1_024,
) -> np.ndarray:
    """Encodes documents with <|endoftext|> separators into contiguous uint16 blocks [N_blocks, seq_len + 1]."""
    flat_chunks: list[np.ndarray] = []
    for doc in docs:
        ids = tokenizer.encode(doc, add_eos=True)
        flat_chunks.append(np.asarray(ids, dtype=np.uint16))
    if not flat_chunks:
        raise ValueError("Empty document list passed to pack_documents_to_blocks")
    all_ids = np.concatenate(flat_chunks)
    n_blocks = (len(all_ids) - 1) // seq_len
    trimmed = all_ids[: n_blocks * seq_len + 1]
    out = np.empty((n_blocks, seq_len + 1), dtype=np.uint16)
    for idx in range(n_blocks):
        s = idx * seq_len
        out[idx] = trimmed[s : s + seq_len + 1]
    return out


def compute_quantized_step_Schedule(total_blocks: int, total_steps: int, quantum: int = 32) -> list[tuple[int, int]]:
    """
    Partitions [0, total_blocks) into `total_steps` contiguous slices whose interior boundaries
    are exact multiples of `quantum` (micro_batch_size=32), eliminating partial-batch padding waste!
    """
    bounds = [0]
    for s in range(1, total_steps):
        raw = (s * total_blocks) / total_steps
        q = int(round(raw / quantum)) * quantum
        q = max(bounds[-1] + quantum, min(q, total_blocks))
        # If corpus is smaller than total_steps * quantum, fall back to proportional
        if total_blocks < total_steps * quantum:
            q = (s * total_blocks) // total_steps
        bounds.append(q)
    bounds.append(total_blocks)
    return [(bounds[s], bounds[s + 1]) for s in range(total_steps)]


def train_language_model(
    run_name: str,
    tokenizer: TokenizerInterface,
    train_docs: Optional[Sequence[str]],
    val_docs: Sequence[str],
    output_dir: str | Path,
    prepacked_blocks: Optional[np.ndarray] = None,
    model_spec: ModelSpec = ModelSpec(),
    train_spec: TrainSpec = TrainSpec(),
    regime: str = "equal_raw_bytes",
    target_steps: int | None = None,
    target_emitted_tokens: int | None = None,
    micro_batch_size: int = 32,
    seed: int = 1337,
    eval_every_steps: int = 450,
    use_compile: bool = True,
) -> tuple[ResearchTransformerLM, dict]:
    assert regime in ("equal_raw_bytes", "equal_flops")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    t_tok0 = time.perf_counter()
    if prepacked_blocks is not None:
        blocks = prepacked_blocks
    else:
        assert train_docs is not None
        blocks = pack_documents_to_blocks(train_docs, tokenizer, seq_len=train_spec.seq_len)
    tok_encode_sec = time.perf_counter() - t_tok0

    total_avail_blocks = len(blocks)

    if regime == "equal_flops":
        assert target_emitted_tokens is not None
        needed_blocks = target_emitted_tokens // train_spec.seq_len
        assert total_avail_blocks >= needed_blocks
        blocks = blocks[:needed_blocks]
        total_avail_blocks = needed_blocks

    if target_steps is None:
        total_steps = max(1, total_avail_blocks // (micro_batch_size * 2))
    else:
        total_steps = target_steps

    step_slices = compute_quantized_step_Schedule(total_avail_blocks, total_steps, quantum=micro_batch_size)

    raw_model = ResearchTransformerLM(model_spec).to(device)
    exec_model: torch.nn.Module = raw_model
    if use_compile:
        exec_model = torch.compile(raw_model)

    decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_spec.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_spec.peak_lr,
        betas=(train_spec.beta1, train_spec.beta2),
        fused=True,
    )

    warmup_steps = max(10, int(total_steps * train_spec.warmup_fraction))
    active_vocab = tokenizer.vocab_size

    history: list[dict] = []
    emitted_tokens_processed = 0
    last_step_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
    t_train0 = time.perf_counter()

    for step, (b_start, b_end) in enumerate(step_slices):
        if b_end <= b_start:
            b_start = max(0, b_end - micro_batch_size)
        lr = get_cosine_lr(step, total_steps, train_spec.peak_lr, train_spec.min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        step_blocks = blocks[b_start:b_end]
        n_sb = len(step_blocks)
        total_step_real_tokens = max(n_sb * train_spec.seq_len, 1)
        optimizer.zero_grad(set_to_none=True)
        step_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)

        for mb0 in range(0, n_sb, micro_batch_size):
            mb = step_blocks[mb0 : mb0 + micro_batch_size]
            real_rows = len(mb)
            if real_rows < micro_batch_size:
                x_pad = np.full((micro_batch_size, train_spec.seq_len), tokenizer.pad_id, dtype=np.int64)
                y_pad = np.full((micro_batch_size, train_spec.seq_len), -100, dtype=np.int64)
                x_pad[:real_rows] = mb[:, :-1]
                y_pad[:real_rows] = mb[:, 1:]
                x = torch.from_numpy(x_pad).to(device, non_blocking=True)
                y = torch.from_numpy(y_pad).to(device, non_blocking=True)
            else:
                x = torch.from_numpy(mb[:, :-1].astype(np.int64)).to(device, non_blocking=True)
                y = torch.from_numpy(mb[:, 1:].astype(np.int64)).to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                _, mb_sum_loss = exec_model(
                    x,
                    y,
                    active_output_vocab=active_vocab,
                    loss_reduction="sum",
                    return_logits=False,
                )
                scaled_loss = mb_sum_loss / total_step_real_tokens
            scaled_loss.backward()
            step_loss_gpu = step_loss_gpu + mb_sum_loss.detach().float()
            emitted_tokens_processed += real_rows * train_spec.seq_len

        last_step_loss_gpu = step_loss_gpu / total_step_real_tokens
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_spec.grad_clip)
        optimizer.step()

        is_last = step == total_steps - 1
        if (step > 0 and step % eval_every_steps == 0) or is_last or step == 50:
            torch.cuda.synchronize(device)
            step_loss_val = float(last_step_loss_gpu.item())
            elapsed = time.perf_counter() - t_train0
            tok_per_s = emitted_tokens_processed / max(elapsed, 1e-6)
            val_metrics = evaluate_bpb_and_stats(
                raw_model, val_docs[:500], tokenizer, device=device, dtype=dtype, byte_cap_context=3200
            )
            rec = {
                "step": step + 1,
                "total_steps": total_steps,
                "lr": lr,
                "train_ce": step_loss_val,
                "val_bpb_matched_ctx": val_metrics["bpb"],
                "emitted_tokens": emitted_tokens_processed,
                "tok_per_sec": tok_per_s,
                "elapsed_sec": elapsed,
            }
            history.append(rec)
            rem_steps = total_steps - (step + 1)
            sec_per_step = elapsed / (step + 1)
            eta_min = (rem_steps * sec_per_step) / 60.0
            print(
                f"[{run_name}] step {step+1:4d}/{total_steps} | "
                f"train_ce={step_loss_val:.4f} | val_bpb={val_metrics['bpb']:.4f} | "
                f"{tok_per_s:,.0f} tok/s | ETA {eta_min:.1f}m",
                flush=True,
            )

            if is_last or (step + 1) in {total_steps // 4, total_steps // 2, (3 * total_steps) // 4}:
                ckpt_file = out_path / ("ckpt_final.pt" if is_last else f"ckpt_step_{step+1}.pt")
                torch.save({"step": step + 1, "model": raw_model.state_dict(), "spec": model_spec.__dict__}, ckpt_file)

    torch.cuda.synchronize(device)
    total_train_sec = time.perf_counter() - t_train0
    summary = {
        "run_name": run_name,
        "regime": regime,
        "seed": seed,
        "vocab_size": active_vocab,
        "encode_corpus_sec": tok_encode_sec,
        "train_wall_sec": total_train_sec,
        "emitted_tokens_processed": emitted_tokens_processed,
        "avg_tok_per_sec": emitted_tokens_processed / max(total_train_sec, 1e-6),
        "history": history,
    }
    (out_path / "train_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return raw_model, summary
