"""
PPL Evaluation for Pythia on PG-19 Dataset

This script evaluates perplexity on the local PG-19 dataset with multiple KV cache strategies:
1. Baseline: Full KV cache (chunked evaluation, 512 tokens per chunk)
2. StreamingLLM: Sink + Recent window (token-by-token with KV cache eviction)
3. H2O: Heavy Hitter + Recent (attention-based eviction)
4. LazyH2O: Periodic H2O (lazy eviction between updates)

Usage:
    python evaluate/eval_ppl_pg19.py --mode baseline --context_length 512
    python evaluate/eval_ppl_pg19.py --mode streaming
    python evaluate/eval_ppl_pg19.py --mode h2o --heavy_size 128
    python evaluate/eval_ppl_pg19.py --mode lazy_h2o --update_interval 10
    python evaluate/eval_ppl_pg19.py --mode sepllm
    python evaluate/eval_ppl_pg19.py --mode unified
"""

import sys
import os

# Auto-detect project root and add to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import glob
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

# Import from accelerated_inference
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache, SepLLMKVCache
from accelerated_inference.kvpress.presses.unified_press import UnifiedKVCache
from accelerated_inference.utils import (
    enable_gpt_neox_pos_shift_attention,
    H2OKVCache,
    LazyH2OKVCache,
)

device = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="PPL evaluation on PG-19 dataset")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="pythia-2.8b-local",
        help="Path to the model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/pg19",
        help="Directory containing PG-19 txt files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ppl_pg19",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of text files to evaluate"
    )
    parser.add_argument(
        "--num_eval_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to evaluate (None = all)"
    )
    
    # Context length for chunked evaluation (baseline mode)
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Context length for chunked evaluation (baseline mode)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "streaming", "h2o", "lazy_h2o", "sepllm", "unified"],
        default="baseline",
        help="KV cache strategy to use"
    )
    
    # Cache configuration
    parser.add_argument("--start_size", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--recent_size", type=int, default=252, help="Number of recent tokens")
    
    # H2O specific
    parser.add_argument("--heavy_size", type=int, default=128, help="Number of Heavy Hitter tokens")
    
    # LazyH2O specific
    parser.add_argument("--update_interval", type=int, default=10, help="H2O update interval")
    
    # SepLLM / Unified specific
    parser.add_argument("--separator_size", type=int, default=64, help="Max separator tokens")
    parser.add_argument("--local_size", type=int, default=256, help="Local window size")
    
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
    need_attention = False
    need_input_ids = False
    
    if mode == "baseline":
        print("Mode: Baseline (chunked evaluation, no KV cache eviction)")
        
    elif mode == "streaming":
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
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
        kv_cache = H2OKVCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            heavy_size=args.heavy_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        need_attention = True
        print(f"Mode: H2O (start={args.start_size}, recent={args.recent_size}, heavy={args.heavy_size})")
        
    elif mode == "lazy_h2o":
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
        need_attention = True
        print(f"Mode: LazyH2O (start={args.start_size}, recent={args.recent_size}, "
              f"heavy={args.heavy_size}, interval={args.update_interval})")
    
    elif mode == "sepllm":
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
        need_input_ids = True
        print(f"Mode: SepLLM (start={args.start_size}, local={args.local_size}, separator={args.separator_size})")
    
    elif mode == "unified":
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
        need_attention = True
        need_input_ids = True
        print(f"Mode: Unified (start={args.start_size}, sep={args.separator_size}, "
              f"heavy={args.heavy_size}, local={args.local_size})")
    
    return kv_cache, need_attention, need_input_ids


def load_pg19_texts(data_dir, num_samples):
    """Load text files from PG-19 dataset directory"""
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    if len(txt_files) == 0:
        raise ValueError(f"No .txt files found in {data_dir}")
    
    print(f"Found {len(txt_files)} text files in {data_dir}")
    
    texts = []
    for txt_file in txt_files[:num_samples]:
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            texts.append({
                "filename": os.path.basename(txt_file),
                "text": text
            })
    
    return texts


def evaluate_ppl_chunked(model, tokenizer, texts, args):
    """
    Evaluate perplexity using chunked evaluation (for baseline mode).
    
    Process text in non-overlapping chunks of context_length tokens.
    For each chunk, compute loss on all tokens (predicting next token).
    This is the standard efficient PPL evaluation method.
    """
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    num_eval_tokens = 0
    context_length = args.context_length
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using chunked evaluation with context_length={context_length}")
    
    for sample_idx, sample in enumerate(texts):
        filename = sample["filename"]
        text = sample["text"]
        
        print(f"\n[{sample_idx + 1}/{len(texts)}] Processing: {filename}")
        
        # Tokenize
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)
        print(f"  Sequence length: {seq_len} tokens")
        
        # Calculate number of chunks
        num_chunks = (seq_len - 1) // context_length + 1
        print(f"  Number of chunks: {num_chunks}")
        
        # Process each chunk
        pbar = tqdm(range(num_chunks), desc=f"Evaluating {filename}")
        
        for chunk_idx in pbar:
            # Get chunk boundaries
            start_idx = chunk_idx * context_length
            end_idx = min(start_idx + context_length, seq_len)
            
            # Get the chunk
            chunk_ids = input_ids[:, start_idx:end_idx]
            
            if chunk_ids.size(1) < 2:
                continue  # Need at least 2 tokens for input-target pair
            
            with torch.no_grad():
                # Forward pass on the chunk
                outputs = model(
                    chunk_ids,
                    use_cache=False,
                )
                
                # logits[:-1] predicts targets[1:]
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = chunk_ids[:, 1:].contiguous()
                
                # Compute loss for each token in the chunk
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                
                chunk_nlls = loss_fn(shift_logits, shift_labels)
            
            # Accumulate NLLs
            nlls.extend(chunk_nlls.tolist())
            num_eval_tokens += chunk_nlls.size(0)
            
            # Update progress bar
            current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
            pbar.set_description(f"chunk {chunk_idx+1}/{num_chunks}, ppl: {current_ppl:.2f}")
            
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            print(f"Reached {args.num_eval_tokens} tokens limit")
            break
    
    # Compute final PPL
    final_ppl = torch.exp(torch.tensor(nlls).mean()).item()
    
    return {
        "ppl": final_ppl,
        "num_tokens": num_eval_tokens,
        "num_samples": sample_idx + 1,
        "context_length": context_length,
    }


def evaluate_ppl_streaming(model, tokenizer, texts, kv_cache, need_attention, need_input_ids, args):
    """
    Evaluate perplexity with streaming/incremental processing.
    
    Process tokens one by one, using KV cache for context.
    This is the proper way to evaluate PPL with KV cache eviction methods
    (StreamingLLM, H2O, LazyH2O, SepLLM, Unified).
    """
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    num_eval_tokens = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using streaming evaluation (token-by-token with KV cache)")
    
    for sample_idx, sample in enumerate(texts):
        filename = sample["filename"]
        text = sample["text"]
        
        print(f"\n[{sample_idx + 1}/{len(texts)}] Processing: {filename}")
        
        # Tokenize
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)
        print(f"  Sequence length: {seq_len} tokens")
        
        # Reset KV cache for each document
        past_key_values = None
        if kv_cache is not None and hasattr(kv_cache, 'reset'):
            kv_cache.reset()
        
        # Evaluate token by token
        pbar = tqdm(range(seq_len - 1), desc=f"Evaluating")
        
        for idx in pbar:
            current_token = input_ids[:, idx:idx+1]
            target_token = input_ids[:, idx+1]
            
            with torch.no_grad():
                outputs = model(
                    current_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=need_attention,
                )
                logits = outputs.logits[:, -1, :]  # [batch, vocab]
                past_key_values = outputs.past_key_values
                
                # Apply KV cache eviction with appropriate arguments
                if kv_cache is not None:
                    if need_attention and need_input_ids:
                        # Unified mode: needs both attention and input_ids
                        attn = outputs.attentions[0] if hasattr(outputs, 'attentions') and outputs.attentions else None
                        past_key_values = kv_cache(past_key_values, current_token, attn)
                    elif need_attention:
                        # H2O/LazyH2O: needs attention only
                        attn = outputs.attentions[0] if hasattr(outputs, 'attentions') and outputs.attentions else None
                        past_key_values = kv_cache(past_key_values, attn)
                    elif need_input_ids:
                        # SepLLM: needs input_ids only
                        past_key_values = kv_cache(past_key_values, current_token)
                    else:
                        # StreamingLLM: no extra args
                        past_key_values = kv_cache(past_key_values)
                
                # Compute NLL
                neg_log_likelihood = loss_fn(logits, target_token)
            
            nlls.append(neg_log_likelihood.item())
            num_eval_tokens += 1
            
            # Update progress bar
            if num_eval_tokens % 100 == 0:
                current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                pbar.set_description(f"ppl: {current_ppl:.2f}")
            
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            print(f"Reached {args.num_eval_tokens} tokens limit")
            break
    
    # Compute final PPL
    final_ppl = torch.exp(torch.tensor(nlls).mean()).item()
    
    # Get cache size info
    cache_size = args.start_size + args.recent_size
    if args.mode in ["h2o", "lazy_h2o"]:
        cache_size += args.heavy_size
    
    return {
        "ppl": final_ppl,
        "num_tokens": num_eval_tokens,
        "num_samples": sample_idx + 1,
        "cache_size": cache_size,
    }


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"PPL Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "baseline":
        print(f"Context length: {args.context_length}")
    else:
        print(f"Cache Config: start={args.start_size}, recent={args.recent_size}")
        if args.mode in ["h2o", "lazy_h2o"]:
            print(f"  heavy_size: {args.heavy_size}")
        if args.mode == "lazy_h2o":
            print(f"  update_interval: {args.update_interval}")
    
    print(f"Num samples: {args.num_samples}")
    print(f"Num eval tokens: {args.num_eval_tokens if args.num_eval_tokens else 'All'}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model(args.model_name_or_path)
    
    # Setup cache
    kv_cache, need_attention, need_input_ids = setup_cache(model, tokenizer, args)
    
    # Load PG-19 texts
    texts = load_pg19_texts(args.data_dir, args.num_samples)
    
    # Evaluate based on mode
    if args.mode == "baseline":
        # Use chunked evaluation for baseline (efficient)
        results = evaluate_ppl_chunked(model, tokenizer, texts, args)
    else:
        # Use streaming evaluation for KV cache eviction methods
        results = evaluate_ppl_streaming(model, tokenizer, texts, kv_cache, need_attention, need_input_ids, args)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_dict = {
        "mode": args.mode,
        "ppl": results["ppl"],
        "num_tokens": results["num_tokens"],
        "num_samples": results["num_samples"],
    }
    
    if args.mode == "baseline":
        results_dict["context_length"] = args.context_length
    else:
        results_dict["config"] = {
            "start_size": args.start_size,
            "recent_size": args.recent_size,
            "heavy_size": args.heavy_size if args.mode in ["h2o", "lazy_h2o"] else None,
            "update_interval": args.update_interval if args.mode == "lazy_h2o" else None,
        }
    
    output_file = f"{args.output_dir}/ppl_results_{args.mode}.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Results ({args.mode})")
    print(f"{'='*60}")
    print(f"Final PPL: {results['ppl']:.4f}")
    print(f"Tokens evaluated: {results['num_tokens']}")
    print(f"Samples processed: {results['num_samples']}")
    if args.mode == "baseline":
        print(f"Context length: {args.context_length}")
    else:
        print(f"KV Cache size: {results.get('cache_size', 'N/A')}")
    print(f"{'='*60}")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
