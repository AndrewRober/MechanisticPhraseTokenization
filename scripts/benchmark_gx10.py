import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import ModelSpec, TrainSpec
from src.model import ResearchTransformerLM


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = seconds / 60.0
    if mins < 60:
        return f"{mins:.1f} min"
    hrs = mins / 60.0
    return f"{hrs:.2f} hr ({mins:.0f} min)"


def benchmark_config(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_steps: int = 8,
    measure_steps: int = 25,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(measure_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    total_tokens = batch_size * seq_len * measure_steps
    tok_per_sec = total_tokens / elapsed
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return tok_per_sec, peak_gb


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 30.4M ResearchTransformerLM throughput and ETAs")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU not detected!"
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    spec = ModelSpec()
    train_spec = TrainSpec()
    base_model = ResearchTransformerLM(spec).to(device)
    breakdown = base_model.parameter_breakdown()

    print("=" * 72)
    print(f"Device       : {torch.cuda.get_device_name(device)}")
    print(f"PyTorch/CUDA : {torch.__version__} / CUDA {torch.version.cuda} / dtype={dtype}")
    print(f"Parameters   : {breakdown['total']:,} total "
          f"({breakdown['tied_embedding']:,} tied emb + {breakdown['non_embedding_backbone']:,} backbone)")
    print("=" * 72)

    modes = [("eager", base_model)]
    if not args.no_compile:
        try:
            compiled_model = torch.compile(ResearchTransformerLM(spec).to(device))
            modes.append(("torch.compile", compiled_model))
        except Exception as exc:
            print(f"[warn] torch.compile skipped: {exc}")

    best_tps = 0.0
    best_desc = ""

    for mode_name, mdl in modes:
        print(f"\n--- Mode: {mode_name} ---")
        for bs in args.batch_sizes:
            try:
                tps, mem_gb = benchmark_config(
                    mdl, bs, spec.max_seq_len, spec.vocab_size, device, dtype
                )
                flops_per_tok = 6.0 * breakdown["total"]
                achieved_tflops = (tps * flops_per_tok) / 1e12
                print(
                    f"  micro_bs={bs:3d} ({bs * spec.max_seq_len:7,d} tok/step) | "
                    f"{tps:10,.0f} tok/s | {achieved_tflops:6.2f} TFLOP/s | peak_mem={mem_gb:5.2f} GB"
                )
                if tps > best_tps:
                    best_tps = tps
                    best_desc = f"{mode_name}, micro_bs={bs}"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  micro_bs={bs:3d} | OOM")
                break

    if best_tps > 0:
        # Phrase tokenizers emit ~0.83x tokens on the same raw text in iso-byte mode
        phrase_factor = 0.84
        print("\n" + "=" * 72)
        print(f"BEST THROUGHPUT: {best_tps:,.0f} tokens/sec ({best_desc})")
        print("=" * 72)
        print("Projected Wall-Clock Training ETAs (pure training compute):")
        print(f"  1. Shared Base Pilot (100M tok)            : {format_eta(100e6 / best_tps)}")
        print(f"  2. Single 30M-tok run (old Stage B spec)   : {format_eta(30e6 / best_tps)}")
        print(f"  3. Single 120M-tok run (upgraded Stage B)  : {format_eta(120e6 / best_tps)}")
        print(f"  4. Full Stage B (11 methods @ 120M eq-tok) : {format_eta((3 * 120e6 + 8 * 120e6 * phrase_factor) / best_tps)}")
        print(f"  5. Single 300M-tok Stage C confirmation    : {format_eta(300e6 / best_tps)}")
        print(f"  6. Full Stage C (6 methods x 3 seeds @300M): {format_eta(18 * 300e6 / best_tps)}")
        print("=" * 72)


if __name__ == "__main__":
    main()
