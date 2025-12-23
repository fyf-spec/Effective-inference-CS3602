"""
Unified KV Cache Press: Combines Initial + Separator + Heavy Hitter + Local

This module implements a unified KV cache eviction strategy that combines:
- StreamingLLM: Initial sink tokens + Local window
- SepLLM: Separator token preservation
- H2O: Heavy Hitter (high attention) token retention

Cache Structure:
    [Initial/Sink] + [Separator] + [Heavy Hitter] + [Local/Recent]
"""

import torch
from typing import Optional, List, Set, Tuple

from .benchmark_presses import DIM_TO_SLICE, get_separator_ids


class UnifiedKVCache:
    """
    Unified KV Cache: Initial + Separator + Heavy Hitter + Local.
    
    Combines the best of StreamingLLM, SepLLM, and H2O approaches:
    - Initial tokens (attention sinks)
    - Separator tokens (punctuation, semantic boundaries)
    - Heavy Hitter tokens (high cumulative attention)
    - Local tokens (recent context window)
    
    Args:
        tokenizer: HuggingFace tokenizer for separator detection
        start_size: Number of initial sink tokens
        separator_size: Max separator tokens to keep
        heavy_size: Max heavy hitter tokens to keep
        local_size: Size of local (recent) window
        k_seq_dim: Key sequence dimension (2 for GPT-NeoX)
        v_seq_dim: Value sequence dimension (2 for GPT-NeoX)
        separators: Custom separator characters list
    """
    def __init__(
        self,
        tokenizer,
        start_size: int = 4,
        separator_size: int = 64,
        heavy_size: int = 128,
        local_size: int = 256,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
        separators: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.start_size = start_size
        self.separator_size = separator_size
        self.heavy_size = heavy_size
        self.local_size = local_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        
        # Total cache capacity
        self.cache_size = start_size + separator_size + heavy_size + local_size
        
        # Get separator token IDs
        self.separator_ids = get_separator_ids(tokenizer, separators)
        
        # Slice functions
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        # Track token history for separator detection
        self.token_history: List[int] = []
        
        # Accumulated attention scores for Heavy Hitter detection
        self.accumulated_scores: Optional[torch.Tensor] = None
        
        print(f"UnifiedKVCache: start={start_size}, sep={separator_size}, "
              f"heavy={heavy_size}, local={local_size}")
        print(f"  Total capacity: {self.cache_size}")
        print(f"  Separator token IDs: {len(self.separator_ids)}")
    
    def _is_separator(self, token_id: int) -> bool:
        """Check if token is a separator."""
        return int(token_id) in self.separator_ids
    
    def update_scores(self, attn_weights: torch.Tensor):
        """Update accumulated attention scores for Heavy Hitter detection."""
        # attn_weights: [batch, heads, q_len, k_len]
        if attn_weights.dim() == 4:
            new_scores = attn_weights.sum(dim=(0, 1, 2)).detach()  # [k_len]
        elif attn_weights.dim() == 3:
            new_scores = attn_weights.sum(dim=(0, 1)).detach()
        else:
            return
        
        k_len = new_scores.shape[-1]
        
        if self.accumulated_scores is None:
            self.accumulated_scores = new_scores
        else:
            old_len = self.accumulated_scores.shape[-1]
            if k_len > old_len:
                padding = torch.zeros(k_len - old_len, 
                                     device=self.accumulated_scores.device,
                                     dtype=self.accumulated_scores.dtype)
                self.accumulated_scores = torch.cat([self.accumulated_scores, padding])
            self.accumulated_scores[:k_len] = self.accumulated_scores[:k_len] + new_scores
    
    def get_keep_indices(self, seq_len: int, device: torch.device) -> List[int]:
        """
        Compute indices to keep based on unified policy.
        
        Priority order:
        1. Initial tokens (always keep)
        2. Local tokens (always keep)  
        3. Separator tokens (keep most recent up to separator_size)
        4. Heavy Hitter tokens (keep top-k by attention score)
        """
        keep_indices: Set[int] = set()
        
        # 1. Always keep initial tokens (sinks)
        keep_indices.update(range(min(self.start_size, seq_len)))
        
        # 2. Always keep local (recent) tokens
        local_start = max(self.start_size, seq_len - self.local_size)
        keep_indices.update(range(local_start, seq_len))
        
        # Middle region: between initial and local
        middle_start = self.start_size
        middle_end = local_start
        
        if middle_end <= middle_start:
            return sorted(keep_indices)
        
        middle_indices = list(range(middle_start, middle_end))
        
        # 3. Find separator tokens in middle region
        separator_indices = []
        for idx in middle_indices:
            if idx < len(self.token_history) and self._is_separator(self.token_history[idx]):
                separator_indices.append(idx)
        
        # Keep most recent separators (up to separator_size)
        separator_indices = separator_indices[-self.separator_size:]
        keep_indices.update(separator_indices)
        
        # 4. Find Heavy Hitter tokens (excluding already-kept tokens)
        remaining_middle = set(middle_indices) - keep_indices
        
        if self.accumulated_scores is not None and len(remaining_middle) > 0:
            remaining_list = sorted(remaining_middle)
            
            # Get scores for remaining tokens
            if len(self.accumulated_scores) >= max(remaining_list) + 1:
                remaining_scores = self.accumulated_scores[remaining_list]
                
                # Select top-k heavy hitters
                heavy_budget = min(self.heavy_size, len(remaining_list))
                if heavy_budget > 0 and remaining_scores.numel() > 0:
                    _, top_k_indices = remaining_scores.topk(heavy_budget)
                    heavy_indices = [remaining_list[i] for i in top_k_indices.tolist()]
                    keep_indices.update(heavy_indices)
        
        return sorted(keep_indices)
    
    def __call__(
        self, 
        past_key_values, 
        input_ids=None, 
        attn_weights=None
    ):
        """
        Apply unified eviction to past_key_values.
        
        Args:
            past_key_values: HuggingFace model past_key_values
            input_ids: Current input token IDs (for separator detection)
            attn_weights: Attention weights (for Heavy Hitter detection)
            
        Returns:
            Updated past_key_values
        """
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        # Update token history
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                new_tokens = input_ids.view(-1).tolist()
            else:
                new_tokens = list(input_ids)
            self.token_history.extend(new_tokens)
        
        # Update attention scores
        if attn_weights is not None:
            if isinstance(attn_weights, (list, tuple)):
                if len(attn_weights) > 0 and attn_weights[0] is not None:
                    self.update_scores(attn_weights[0])
            else:
                self.update_scores(attn_weights)
        
        # Check if eviction needed
        if seq_len <= self.cache_size:
            return past_key_values
        
        # Get indices to keep
        keep_list = self.get_keep_indices(seq_len, past_key_values[0][0].device)
        
        # Apply eviction to all layers
        new_past = []
        for k, v in past_key_values:
            layer_device = k.device
            keep_indices = torch.tensor(keep_list, device=layer_device, dtype=torch.long)
            new_k = torch.index_select(k, self.k_seq_dim, keep_indices)
            new_v = torch.index_select(v, self.v_seq_dim, keep_indices)
            new_past.append([new_k, new_v])
        
        # Update accumulated scores
        if self.accumulated_scores is not None and len(keep_list) > 0:
            score_device = self.accumulated_scores.device
            score_indices = torch.tensor(keep_list, device=score_device, dtype=torch.long)
            self.accumulated_scores = self.accumulated_scores[score_indices]
        
        # Update token history
        self.token_history = [self.token_history[i] for i in keep_list if i < len(self.token_history)]
        
        return new_past
    
    def reset(self):
        """Reset cache state for new sequence."""
        self.token_history = []
        self.accumulated_scores = None


def enable_unified_cache(model, tokenizer, start_size=4, separator_size=64, 
                         heavy_size=128, local_size=256):
    """
    Enable Unified KV Cache for a model.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        start_size: Number of sink tokens
        separator_size: Max separator tokens
        heavy_size: Max heavy hitter tokens
        local_size: Recent window size
        
    Returns:
        UnifiedKVCache instance
    """
    # Import here to avoid circular imports
    from accelerated_inference.utils import enable_gpt_neox_pos_shift_attention
    
    if "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")
    
    return UnifiedKVCache(
        tokenizer=tokenizer,
        start_size=start_size,
        separator_size=separator_size,
        heavy_size=heavy_size,
        local_size=local_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
