"""
StreamingLLM Utilities for accelerated_inference

This module contains utilities from the streaming-llm project adapted for 
GPT-NeoX (Pythia) architecture.
"""

import torch
import argparse
import types
import os
import os.path as osp
import ssl
import urllib.request
import json
from typing import Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.gpt_neox.modeling_gpt_neox import (
    apply_rotary_pos_emb,
    rotate_half,
    GPTNeoXAttention,
)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments for streaming LLM evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/llama/llama-7b"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


# =============================================================================
# Model Loading
# =============================================================================

def load(model_name_or_path):
    """Load model and tokenizer from path."""
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


# =============================================================================
# Utility Functions
# =============================================================================

def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict


# =============================================================================
# GPT-NeoX Position Shift Attention (for StreamingLLM)
# =============================================================================

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    """Apply rotary position embedding to a single tensor (query or key)."""
    # cos: [1, 1, max_seq_len, head_dim]
    # sin: [1, 1, max_seq_len, head_dim]
    # position_ids: [bs, seq_len]
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def gpt_neox_pos_shift_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    """
    Modified GPT-NeoX attention forward with position shift for StreamingLLM.
    
    This enables position shift for the KV cache so that the model can
    attend to tokens beyond its original context window.
    """
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query = apply_rotary_pos_emb_single(query_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)

    # Cache QKV values
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    present = (key, value) if use_cache else None

    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]
    key_position_ids = torch.arange(seq_len, device=position_ids.device).unsqueeze(0)
    key = apply_rotary_pos_emb_single(key_rot, cos, sin, key_position_ids)
    key = torch.cat((key, key_pass), dim=-1)

    # Compute attention
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    # Reshape outputs
    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_size
    )
    attn_output = self.dense(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def enable_gpt_neox_pos_shift_attention(model):
    """
    Enable position shift attention for GPT-NeoX model.
    
    This replaces the attention forward method in all GPTNeoXAttention modules
    with the position-shift-aware version for StreamingLLM.
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gpt_neox_pos_shift_attention(module)

        if isinstance(module, GPTNeoXAttention):
            module.forward = types.MethodType(
                gpt_neox_pos_shift_attention_forward, module
            )


# =============================================================================
# StreamingLLM KV Cache Import (from benchmark_presses)
# =============================================================================

# Import StartRecentKVCache from benchmark_presses for convenience
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache


def enable_streaming_llm(model, start_size, recent_size):
    """
    Enable StreamingLLM for a model.
    
    This function:
    1. Enables position shift attention for the model
    2. Returns a KV cache that keeps start and recent tokens
    
    Args:
        model: HuggingFace model (GPT-NeoX/Pythia, LLaMA, MPT, Falcon)
        start_size: Number of initial tokens to keep ("sink" tokens)
        recent_size: Number of recent tokens to keep
        
    Returns:
        StartRecentKVCache instance for the model
    """
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        # LLaMA position shift not included - add if needed
        raise NotImplementedError("LLaMA position shift not implemented in this module")
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        # Falcon position shift not included - add if needed
        raise NotImplementedError("Falcon position shift not implemented in this module")
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")
    
    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache


# =============================================================================
# H2O (Heavy-Hitter Oracle) KV Cache
# =============================================================================

def local_heavy_hitter_mask(attn_weights, heavy_budget, recent_budget=0):
    """
    Identify Heavy Hitter tokens based on accumulated attention scores.
    
    This is the core H2O algorithm that identifies which tokens receive
    the most attention (Heavy Hitters) and should be kept in the cache.
    
    Args:
        attn_weights: Attention weights tensor [batch, heads, query_len, key_len]
        heavy_budget: Number of Heavy Hitter tokens to keep
        recent_budget: Number of recent tokens to always keep
        
    Returns:
        mask: Boolean tensor indicating which tokens to keep
        heavy_indices: Indices of Heavy Hitter tokens (for LazyH2O)
    """
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    
    # Compute attention probabilities
    tmp_attn = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)
    
    # Accumulate attention scores: sum over queries for each key
    # This gives us the "importance" of each key position
    # Shape: [batch, heads, key_len]
    accumulated_attention_score = tmp_attn.sum(dim=-2)
    
    # Create mask for Heavy Hitters (top-k by accumulated attention)
    # Get indices of top-k tokens
    _, heavy_indices = accumulated_attention_score.topk(k=min(heavy_budget, seq_length), dim=-1)
    
    # Create boolean mask
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    
    # Mark Heavy Hitter positions
    batch_size, num_heads, q_len, k_len = attn_weights.shape
    for b in range(batch_size):
        for h in range(num_heads):
            mask_bottom[b, h, :, heavy_indices[b, h]] = True
    
    # Always keep recent tokens
    if recent_budget > 0:
        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-(recent_budget - 1))
        mask_bottom = torch.logical_or(mask_bottom, ones)
    
    # Apply causal mask (can only attend to past tokens)
    mask_bottom = torch.tril(mask_bottom, diagonal=0)
    
    return mask_bottom, heavy_indices


class H2OKVCache:
    """
    H2O (Heavy-Hitter Oracle) KV Cache.
    
    Evicts tokens based on accumulated attention scores at every step.
    Keeps Heavy Hitter (high attention) tokens + recent window.
    
    Note: This has higher overhead than StreamingLLM due to computing
    attention scores and sorting at every step.
    """
    def __init__(
        self,
        start_size=4,
        recent_size=256,
        heavy_size=128,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"H2OKVCache: start={start_size}, recent={recent_size}, heavy={heavy_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.heavy_size = heavy_size
        self.cache_size = start_size + recent_size + heavy_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        
        # Track accumulated attention scores: [k_len] 1D tensor
        self.accumulated_scores = None
        
    def update_scores(self, attn_weights):
        """Update accumulated attention scores."""
        # attn_weights: [batch, heads, q_len, k_len]
        # Sum over all dimensions except last to get importance of each key
        # Result: [k_len]
        if attn_weights.dim() == 4:
            new_scores = attn_weights.sum(dim=(0, 1, 2)).detach()  # [k_len]
        elif attn_weights.dim() == 3:
            new_scores = attn_weights.sum(dim=(0, 1)).detach()  # [k_len]
        else:
            new_scores = attn_weights.sum().detach().unsqueeze(0)  # fallback
        
        k_len = new_scores.shape[-1]
        
        if self.accumulated_scores is None:
            self.accumulated_scores = new_scores
        else:
            old_len = self.accumulated_scores.shape[-1]
            
            if k_len > old_len:
                # Pad old scores with zeros
                padding = torch.zeros(
                    k_len - old_len,
                    device=self.accumulated_scores.device,
                    dtype=self.accumulated_scores.dtype
                )
                self.accumulated_scores = torch.cat([self.accumulated_scores, padding], dim=-1)
            
            # Accumulate (only up to new_scores length)
            self.accumulated_scores[:k_len] = self.accumulated_scores[:k_len] + new_scores
    
    def get_keep_indices(self, seq_len, device):
        """Get indices of tokens to keep based on accumulated scores."""
        # Always keep start tokens (sinks)
        keep_indices = set(range(min(self.start_size, seq_len)))
        
        # Always keep recent tokens
        recent_start = max(self.start_size, seq_len - self.recent_size)
        keep_indices.update(range(recent_start, seq_len))
        
        # Select Heavy Hitters from middle region
        if self.accumulated_scores is not None and seq_len > self.start_size + self.recent_size:
            middle_start = self.start_size
            middle_end = max(middle_start, seq_len - self.recent_size)
            
            if middle_end > middle_start:
                # Get scores for middle region
                scores = self.accumulated_scores
                if scores.dim() == 0:
                    # Scalar - can't select heavy hitters
                    pass
                elif len(scores) >= middle_end:
                    middle_scores = scores[middle_start:middle_end]
                    
                    if middle_scores.numel() > 0:
                        num_heavy = min(self.heavy_size, middle_scores.numel())
                        if num_heavy > 0:
                            _, heavy_idx = middle_scores.topk(num_heavy)
                            heavy_idx = heavy_idx + middle_start  # Offset to absolute indices
                            keep_indices.update(heavy_idx.tolist())
        
        return sorted(keep_indices)
    
    def __call__(self, past_key_values, attn_weights=None):
        """
        Apply H2O eviction to past_key_values.
        
        Args:
            past_key_values: List of (key, value) tuples per layer
            attn_weights: Single attention tensor [batch, heads, q_len, k_len]
                         or list of attention tensors per layer
        """
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        # Update scores if attention weights provided
        if attn_weights is not None:
            if isinstance(attn_weights, (list, tuple)):
                # List of attention weights per layer - use first layer
                if len(attn_weights) > 0 and attn_weights[0] is not None:
                    self.update_scores(attn_weights[0])
            else:
                # Single tensor
                self.update_scores(attn_weights)
        
        # Check if eviction needed
        if seq_len <= self.cache_size:
            return past_key_values
        
        # Get indices to keep (as a list first)
        keep_list = self.get_keep_indices(seq_len, past_key_values[0][0].device)
        
        # Apply eviction to all layers - create keep_indices on each layer's device
        new_past = []
        for k, v in past_key_values:
            # Create indices on the same device as this layer's KV cache
            layer_device = k.device
            keep_indices = torch.tensor(keep_list, device=layer_device, dtype=torch.long)
            new_k = torch.index_select(k, self.k_seq_dim, keep_indices)
            new_v = torch.index_select(v, self.v_seq_dim, keep_indices)
            new_past.append((new_k, new_v))
        
        # Update accumulated scores to match new indices
        if self.accumulated_scores is not None and len(keep_list) > 0:
            score_device = self.accumulated_scores.device
            score_indices = torch.tensor(keep_list, device=score_device, dtype=torch.long)
            self.accumulated_scores = self.accumulated_scores[score_indices]
        
        return new_past
    
    def reset(self):
        """Reset cache state for new sequence."""
        self.accumulated_scores = None


class LazyH2OKVCache:
    """
    LazyH2O: Periodic Heavy-Hitter Oracle KV Cache.
    
    Combines StreamingLLM's speed with H2O's intelligent eviction by
    only running the full H2O algorithm every `update_interval` steps.
    
    During "lazy" steps, uses lightweight eviction that respects a
    protected set of indices (the previously identified Heavy Hitters).
    
    Args:
        start_size: Number of initial "sink" tokens to always keep
        recent_size: Number of recent tokens to always keep
        heavy_size: Number of Heavy Hitter tokens to keep
        update_interval: Run full H2O every N steps (default: 10)
    """
    def __init__(
        self,
        start_size=4,
        recent_size=256,
        heavy_size=128,
        update_interval=10,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"LazyH2OKVCache: start={start_size}, recent={recent_size}, "
              f"heavy={heavy_size}, update_interval={update_interval}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.heavy_size = heavy_size
        self.cache_size = start_size + recent_size + heavy_size
        self.update_interval = update_interval
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        
        # Step counter
        self.step_k = 0
        
        # Protected set: indices of Heavy Hitter tokens (relative to current cache)
        # These tokens won't be evicted during lazy steps
        self.protected_indices = set(range(start_size))  # Start with sink tokens
        
        # Accumulated attention scores for H2O computation
        self.accumulated_scores = None  # [batch, heads, seq_len]
        
    def _slice(self, x, indices, dim):
        """Helper to select indices along a dimension."""
        indices_tensor = torch.tensor(sorted(indices), device=x.device, dtype=torch.long)
        return torch.index_select(x, dim, indices_tensor)
    
    def _full_h2o_eviction(self, past_key_values):
        """
        Full H2O eviction: recompute Heavy Hitters from accumulated scores.
        Called every `update_interval` steps.
        """
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        if seq_len <= self.cache_size:
            return past_key_values
        
        # Compute which indices to keep
        keep_indices = set(range(self.start_size))  # Always keep sinks
        
        # Always keep recent tokens
        recent_start = max(self.start_size, seq_len - self.recent_size)
        keep_indices.update(range(recent_start, seq_len))
        
        # Select Heavy Hitters from middle region based on accumulated scores
        if self.accumulated_scores is not None and seq_len > self.start_size + self.recent_size:
            middle_start = self.start_size
            middle_end = seq_len - self.recent_size
            
            # Average scores across batch and heads
            avg_scores = self.accumulated_scores.mean(dim=(0, 1))  # [seq_len]
            middle_scores = avg_scores[middle_start:middle_end]
            
            num_heavy = min(self.heavy_size, len(middle_scores))
            if num_heavy > 0 and len(middle_scores) > 0:
                _, heavy_idx = middle_scores.topk(min(num_heavy, len(middle_scores)))
                heavy_idx = heavy_idx + middle_start
                keep_indices.update(heavy_idx.tolist())
        
        # Update protected set (indices relative to new cache)
        keep_list = sorted(keep_indices)
        new_protected = set()
        for new_idx, old_idx in enumerate(keep_list):
            if old_idx < self.start_size or old_idx in set(range(self.start_size, seq_len - self.recent_size)):
                # This was either a sink or a Heavy Hitter
                if old_idx < self.start_size:
                    new_protected.add(new_idx)
                elif self.accumulated_scores is not None:
                    # Check if it was identified as heavy
                    avg_scores = self.accumulated_scores.mean(dim=(0, 1))
                    middle_start = self.start_size
                    middle_end = seq_len - self.recent_size
                    if middle_start <= old_idx < middle_end:
                        new_protected.add(new_idx)
        
        self.protected_indices = new_protected
        
        # Apply eviction - create indices on each layer's device for multi-GPU
        new_past = []
        for k, v in past_key_values:
            layer_device = k.device
            keep_tensor = torch.tensor(keep_list, device=layer_device, dtype=torch.long)
            new_k = torch.index_select(k, self.k_seq_dim, keep_tensor)
            new_v = torch.index_select(v, self.v_seq_dim, keep_tensor)
            new_past.append((new_k, new_v))
        
        # Update accumulated scores
        if self.accumulated_scores is not None:
            score_device = self.accumulated_scores.device
            score_indices = torch.tensor(keep_list, device=score_device, dtype=torch.long)
            self.accumulated_scores = torch.index_select(
                self.accumulated_scores, -1, score_indices
            )
        
        return new_past
    
    def _lazy_eviction(self, past_key_values):
        """
        Lightweight eviction: evict oldest non-protected token.
        Used between full H2O updates for O(1) overhead.
        """
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        if seq_len <= self.cache_size:
            return past_key_values
        
        # Need to evict (seq_len - cache_size) tokens
        num_to_evict = seq_len - self.cache_size
        
        # Find indices to evict: oldest non-protected tokens
        evict_candidates = []
        for idx in range(self.start_size, seq_len - self.recent_size):
            if idx not in self.protected_indices:
                evict_candidates.append(idx)
        
        # If not enough candidates, evict from oldest protected (except sinks)
        if len(evict_candidates) < num_to_evict:
            protected_middle = [i for i in self.protected_indices if i >= self.start_size]
            evict_candidates.extend(protected_middle[:num_to_evict - len(evict_candidates)])
        
        # Evict oldest candidates first
        evict_candidates = sorted(evict_candidates)[:num_to_evict]
        evict_set = set(evict_candidates)
        
        # Keep everything except evicted
        keep_indices = [i for i in range(seq_len) if i not in evict_set]
        
        # Update protected indices (shift due to eviction)
        new_protected = set()
        for new_idx, old_idx in enumerate(keep_indices):
            if old_idx in self.protected_indices:
                new_protected.add(new_idx)
        self.protected_indices = new_protected
        
        # Apply eviction - create indices on each layer's device for multi-GPU
        new_past = []
        for k, v in past_key_values:
            layer_device = k.device
            keep_tensor = torch.tensor(keep_indices, device=layer_device, dtype=torch.long)
            new_k = torch.index_select(k, self.k_seq_dim, keep_tensor)
            new_v = torch.index_select(v, self.v_seq_dim, keep_tensor)
            new_past.append((new_k, new_v))
        
        # Update accumulated scores
        if self.accumulated_scores is not None:
            score_device = self.accumulated_scores.device
            score_indices = torch.tensor(keep_indices, device=score_device, dtype=torch.long)
            self.accumulated_scores = torch.index_select(
                self.accumulated_scores, -1, score_indices
            )
        
        return new_past
    
    def update_scores(self, attn_weights):
        """Update accumulated attention scores."""
        # attn_weights: [batch, heads, q_len, k_len]
        new_scores = attn_weights.sum(dim=-2).detach()
        
        if self.accumulated_scores is None:
            self.accumulated_scores = new_scores
        else:
            old_len = self.accumulated_scores.shape[-1]
            new_len = new_scores.shape[-1]
            
            if new_len > old_len:
                padding = torch.zeros(
                    (*self.accumulated_scores.shape[:-1], new_len - old_len),
                    device=self.accumulated_scores.device,
                    dtype=self.accumulated_scores.dtype
                )
                self.accumulated_scores = torch.cat([self.accumulated_scores, padding], dim=-1)
            
            self.accumulated_scores = self.accumulated_scores + new_scores
    
    def __call__(self, past_key_values, attn_weights=None):
        """
        Apply LazyH2O eviction.
        
        Args:
            past_key_values: KV cache from model
            attn_weights: Optional attention weights for score accumulation
        """
        # Update attention scores if provided
        if attn_weights is not None:
            self.update_scores(attn_weights)
        
        # Decide which eviction to use
        if self.step_k % self.update_interval == 0:
            result = self._full_h2o_eviction(past_key_values)
        else:
            result = self._lazy_eviction(past_key_values)
        
        self.step_k += 1
        return result
    
    def reset(self):
        """Reset cache state for new sequence."""
        self.step_k = 0
        self.protected_indices = set(range(self.start_size))
        self.accumulated_scores = None


def enable_h2o(model, start_size=4, recent_size=256, heavy_size=128):
    """
    Enable H2O for a model with position shift.
    
    Args:
        model: HuggingFace model
        start_size: Number of sink tokens
        recent_size: Number of recent tokens
        heavy_size: Number of Heavy Hitter tokens
        
    Returns:
        H2OKVCache instance
    """
    if "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")
    
    return H2OKVCache(
        start_size=start_size,
        recent_size=recent_size,
        heavy_size=heavy_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )


def enable_lazy_h2o(model, start_size=4, recent_size=256, heavy_size=128, update_interval=10):
    """
    Enable LazyH2O for a model with position shift.
    
    Args:
        model: HuggingFace model
        start_size: Number of sink tokens
        recent_size: Number of recent tokens
        heavy_size: Number of Heavy Hitter tokens
        update_interval: Run full H2O every N steps
        
    Returns:
        LazyH2OKVCache instance
    """
    if "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")
    
    return LazyH2OKVCache(
        start_size=start_size,
        recent_size=recent_size,
        heavy_size=heavy_size,
        update_interval=update_interval,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )

