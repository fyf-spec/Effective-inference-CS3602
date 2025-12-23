"""
Speed/Latency Benchmark for Pythia-2.8b with Multiple KV Cache Strategies

This script compares decode latency across 4 modes:
1. Baseline: Full KV cache (no eviction)
2. StreamingLLM: Sink + Recent window (O(1) eviction)
3. H2O: Heavy Hitter + Recent (attention-based eviction every step)
4. LazyH2O: Periodic H2O (attention-based eviction every N steps)

Usage:
    python evaluate/eval_speed_benchmark.py --mode baseline
    python evaluate/eval_speed_benchmark.py --mode streaming
    python evaluate/eval_speed_benchmark.py --mode h2o
    python evaluate/eval_speed_benchmark.py --mode lazy_h2o --update_interval 10
    python evaluate/eval_speed_benchmark.py --mode sepllm --separator_size 64 --local_size 256
    python evaluate/eval_speed_benchmark.py --mode unified --separator_size 64 --heavy_size 128 --local_size 256
"""

import sys
import os

# Auto-detect project root and add to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up from evaluate/ to project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import time
import argparse
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from accelerated_inference
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache, SepLLMKVCache
from accelerated_inference.kvpress.presses.unified_press import UnifiedKVCache
from accelerated_inference.utils import (
    enable_gpt_neox_pos_shift_attention,
    H2OKVCache,
    LazyH2OKVCache,
)

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Speed benchmark for Pythia with different KV cache strategies")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="pythia-2.8b-local",
        help="Path to the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/speed_benchmark",
        help="Output directory for results"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "streaming", "h2o", "lazy_h2o", "sepllm", "unified"],
        default="baseline",
        help="KV cache strategy to use"
    )
    
    # Sequence length settings
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Sequence lengths to benchmark"
    )
    parser.add_argument(
        "--num_decode_tokens",
        type=int,
        default=128,
        help="Number of decode tokens to measure average latency"
    )
    
    # Cache configuration (shared across modes)
    parser.add_argument("--start_size", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--recent_size", type=int, default=252, help="Number of recent tokens")
    
    # H2O specific
    parser.add_argument("--heavy_size", type=int, default=128, help="Number of Heavy Hitter tokens (H2O/LazyH2O)")
    
    # LazyH2O specific
    parser.add_argument("--update_interval", type=int, default=10, help="H2O update interval (LazyH2O only)")
    
    # SepLLM / Unified specific
    parser.add_argument("--separator_size", type=int, default=64, help="Max separator tokens to keep (SepLLM/Unified)")
    parser.add_argument("--local_size", type=int, default=256, help="Local window size (SepLLM/Unified)")
    
    return parser.parse_args()


def load_model(model_name_or_path):
    """Load model and tokenizer"""
    print(f"Loading model from {model_name_or_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    model.eval()
    return model, tokenizer


def setup_cache(model, tokenizer, args):
    """Setup KV cache based on mode"""
    mode = args.mode
    kv_cache = None
    
    if mode == "baseline":
        # No cache eviction, no position shift
        print("Mode: Baseline (no KV cache eviction)")
        
    elif mode == "streaming":
        # StreamingLLM: Sink + Recent
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
        kv_cache = StartRecentKVCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        print(f"Mode: StreamingLLM (start={args.start_size}, recent={args.recent_size})")
        
    elif mode == "h2o":
        # H2O: Heavy Hitter + Recent
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
        kv_cache = H2OKVCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            heavy_size=args.heavy_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        print(f"Mode: H2O (start={args.start_size}, recent={args.recent_size}, heavy={args.heavy_size})")
        
    elif mode == "lazy_h2o":
        # LazyH2O: Periodic H2O
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
        kv_cache = LazyH2OKVCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            heavy_size=args.heavy_size,
            update_interval=args.update_interval,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        print(f"Mode: LazyH2O (start={args.start_size}, recent={args.recent_size}, "
              f"heavy={args.heavy_size}, interval={args.update_interval})")
    
    elif mode == "sepllm":
        # SepLLM: Separator-aware eviction
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
        kv_cache = SepLLMKVCache(
            tokenizer=tokenizer,
            start_size=args.start_size,
            local_size=args.local_size,
            separator_size=args.separator_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        print(f"Mode: SepLLM (start={args.start_size}, local={args.local_size}, separator={args.separator_size})")
    
    elif mode == "unified":
        # Unified: Initial + Separator + Heavy Hitter + Local
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
        kv_cache = UnifiedKVCache(
            tokenizer=tokenizer,
            start_size=args.start_size,
            separator_size=args.separator_size,
            heavy_size=args.heavy_size,
            local_size=args.local_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        print(f"Mode: Unified (start={args.start_size}, sep={args.separator_size}, "
              f"heavy={args.heavy_size}, local={args.local_size})")
    
    return kv_cache


def generate_random_input(tokenizer, total_tokens, device="cuda"):
    """Generate random input tokens for benchmarking"""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        low=100,
        high=vocab_size - 100,
        size=(1, total_tokens),
        dtype=torch.long,
        device=device
    )
    return input_ids


def benchmark_at_seq_length(model, tokenizer, seq_length, num_decode_tokens, kv_cache, mode, device="cuda"):
    """
    Benchmark decode latency at a specific sequence length.
    
    Process:
    1. Prefill with (seq_length - 1) tokens
    2. Decode num_decode_tokens more tokens, measuring latency
    3. Return average decode latency
    """
    # Reset cache state if applicable
    if hasattr(kv_cache, 'reset'):
        kv_cache.reset()
    if hasattr(kv_cache, 'accumulated_scores'):
        if isinstance(kv_cache.accumulated_scores, dict):
            kv_cache.accumulated_scores = {}
        else:
            kv_cache.accumulated_scores = None
    
    # Generate random input
    total_tokens = seq_length + num_decode_tokens
    input_ids = generate_random_input(tokenizer, total_tokens, device)
    
    past_key_values = None
    
    # Prefill phase: process (seq_length - 1) tokens at once
    prefill_size = seq_length - 1
    prefill_ids = input_ids[:, :prefill_size]
    
    with torch.no_grad():
        outputs = model(
            prefill_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=(mode in ["h2o", "lazy_h2o", "unified"]),
        )
        past_key_values = outputs.past_key_values
        
        # Apply cache eviction
        if kv_cache is not None:
            if mode in ["h2o", "lazy_h2o", "unified"]:
                attn = outputs.attentions[0] if hasattr(outputs, 'attentions') and outputs.attentions else None
                if mode == "unified":
                    past_key_values = kv_cache(past_key_values, prefill_ids, attn)
                else:
                    past_key_values = kv_cache(past_key_values, attn)
            elif mode == "sepllm":
                past_key_values = kv_cache(past_key_values, prefill_ids)
            else:
                past_key_values = kv_cache(past_key_values)
    
    # First decode step to reach exactly seq_length
    next_token = input_ids[:, prefill_size:prefill_size+1]
    with torch.no_grad():
        outputs = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=(mode in ["h2o", "lazy_h2o", "unified"]),
        )
        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            if mode in ["h2o", "lazy_h2o", "unified"]:
                attn = outputs.attentions[0] if hasattr(outputs, 'attentions') and outputs.attentions else None
                if mode == "unified":
                    past_key_values = kv_cache(past_key_values, next_token, attn)
                else:
                    past_key_values = kv_cache(past_key_values, attn)
            elif mode == "sepllm":
                past_key_values = kv_cache(past_key_values, next_token)
            else:
                past_key_values = kv_cache(past_key_values)
    
    # Measure decode latency for num_decode_tokens more tokens
    decode_times = []
    current_pos = seq_length
    
    for i in range(num_decode_tokens):
        next_token = input_ids[:, current_pos:current_pos+1]
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=(mode in ["h2o", "lazy_h2o", "unified"]),
            )
            past_key_values = outputs.past_key_values
            
            if kv_cache is not None:
                if mode in ["h2o", "lazy_h2o", "unified"]:
                    attn = outputs.attentions[0] if hasattr(outputs, 'attentions') and outputs.attentions else None
                    if mode == "unified":
                        past_key_values = kv_cache(past_key_values, next_token, attn)
                    else:
                        past_key_values = kv_cache(past_key_values, attn)
                elif mode == "sepllm":
                    past_key_values = kv_cache(past_key_values, next_token)
                else:
                    past_key_values = kv_cache(past_key_values)
        
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - start_time
        decode_times.append(decode_time)
        
        current_pos += 1
    
    # Get KV cache size
    if past_key_values is not None:
        kv_size = past_key_values[0][0].size(2)
    else:
        kv_size = current_pos
    
    # Calculate statistics
    avg_latency_ms = float(np.mean(decode_times)) * 1000
    std_latency_ms = float(np.std(decode_times)) * 1000
    total_latency_ms = float(np.sum(decode_times)) * 1000
    
    return {
        "seq_length": int(seq_length),
        "kv_cache_size": int(kv_size),
        "avg_decode_latency_ms": avg_latency_ms,
        "std_decode_latency_ms": std_latency_ms,
        "total_decode_latency_ms": total_latency_ms,
        "tokens_per_sec": 1000 / avg_latency_ms if avg_latency_ms > 0 else 0,
    }


def plot_results(results, output_dir, mode, args):
    """Plot results"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation")
        return
    
    # Color scheme for each mode
    colors = {
        "baseline": "#808080",    # Gray
        "streaming": "#2E8B57",   # Sea Green
        "h2o": "#8B0000",         # Dark Red
        "lazy_h2o": "#4169E1",    # Royal Blue
    }
    
    mode_labels = {
        "baseline": "Baseline",
        "streaming": "StreamingLLM",
        "h2o": "H2O",
        "lazy_h2o": f"LazyH2O (k={args.update_interval})",
    }
    
    seq_lengths = [r["seq_length"] for r in results]
    latencies = [r["avg_decode_latency_ms"] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(seq_lengths))
    bars = ax.bar(x, latencies, color=colors.get(mode, '#808080'), 
                  width=0.6, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.annotate(f'{latency:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Decode Latency (ms)', fontsize=12)
    ax.set_title(f'Decode Latency - {mode_labels.get(mode, mode)}\n(Pythia-2.8B)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.set_ylim(0, max(latencies) * 1.25)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"speed_benchmark_{mode}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model_name_or_path)
    
    # Setup cache based on mode
    kv_cache = setup_cache(model, tokenizer, args)
    
    print(f"\n{'='*60}")
    print(f"Speed Benchmark Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Mode: {args.mode}")
    print(f"Cache Config: start={args.start_size}, recent={args.recent_size}")
    if args.mode in ["h2o", "lazy_h2o"]:
        print(f"  heavy_size: {args.heavy_size}")
    if args.mode == "lazy_h2o":
        print(f"  update_interval: {args.update_interval}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Decode tokens per test: {args.num_decode_tokens}")
    print(f"{'='*60}\n")
    
    # Run benchmarks
    results = []
    
    for seq_length in tqdm(args.seq_lengths, desc="Benchmarking"):
        print(f"\nTesting sequence length: {seq_length}")
        
        try:
            result = benchmark_at_seq_length(
                model, tokenizer, seq_length,
                args.num_decode_tokens, kv_cache, args.mode
            )
            results.append(result)
            
            print(f"  KV Cache Size: {result['kv_cache_size']}")
            print(f"  Avg Decode Latency: {result['avg_decode_latency_ms']:.2f} ± {result['std_decode_latency_ms']:.2f} ms")
            print(f"  Throughput: {result['tokens_per_sec']:.1f} tokens/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] Skipping seq_length={seq_length}")
                torch.cuda.empty_cache()
                continue
            else:
                raise
        
        torch.cuda.empty_cache()
    
    # Save results
    config_dict = {k: v for k, v in vars(args).items()}
    
    results_path = os.path.join(args.output_dir, f"speed_results_{args.mode}.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": config_dict,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot results
    plot_results(results, args.output_dir, args.mode, args)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Seq Length':>12} {'KV Cache':>12} {'Latency (ms)':>15} {'Tokens/sec':>15}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['seq_length']:>12} {r['kv_cache_size']:>12} {r['avg_decode_latency_ms']:>15.2f} {r['tokens_per_sec']:>15.1f}")
    print(f"{'='*80}")
    
    # Summary
    print(f"\n📊 Summary ({args.mode}):")
    if args.mode == "baseline":
        print(f"   KV cache grows with sequence length")
        print(f"   Decode latency INCREASES as sequence length grows")
    elif args.mode == "streaming":
        print(f"   Constant KV cache size: {args.start_size + args.recent_size}")
        print(f"   Fastest eviction (O(1) overhead)")
    elif args.mode == "h2o":
        print(f"   Constant KV cache size: {args.start_size + args.recent_size + args.heavy_size}")
        print(f"   Smart eviction but higher overhead (sorting at every step)")
    elif args.mode == "lazy_h2o":
        print(f"   Constant KV cache size: {args.start_size + args.recent_size + args.heavy_size}")
        print(f"   Hybrid: full H2O every {args.update_interval} steps, lightweight otherwise")


if __name__ == "__main__":
    main()
