from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn


# =============================================================================
# StreamingLLM KV Cache (Standalone, works with past_key_values directly)
# =============================================================================

def slice2d(x, start, end):
    """Slice tensor on dimension 2 (typical KV cache sequence dimension)."""
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    """Slice tensor on dimension 3."""
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    """Slice tensor on dimension 1."""
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    """
    StreamingLLM KV Cache that keeps initial tokens ("sinks") and recent tokens.
    
    This is a standalone implementation that works directly with past_key_values
    from HuggingFace models. Use with enable_gpt_neox_pos_shift_attention for
    best results with long sequences.
    
    Args:
        start_size: Number of initial tokens to keep (attention sinks)
        recent_size: Number of recent tokens to keep (sliding window)
        k_seq_dim: Sequence dimension in key tensors (2 for GPT-NeoX/LLaMA)
        v_seq_dim: Sequence dimension in value tensors (2 for GPT-NeoX/LLaMA)
    
    Example:
        >>> kv_cache = StartRecentKVCache(start_size=4, recent_size=252)
        >>> # During generation loop:
        >>> outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        >>> past_key_values = kv_cache(outputs.past_key_values)
    """
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        """
        Apply cache eviction to past_key_values.
        
        Keeps start_size initial tokens and recent_size recent tokens,
        evicting tokens in the middle when cache exceeds cache_size.
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        """
        Evict tokens to make space for num_coming new tokens.
        
        Use this to proactively evict before adding new tokens, rather than
        after (which is what __call__ does).
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        """
        Evict a specific range of tokens from the cache.
        
        Removes tokens from index start to end (exclusive).
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]


# =============================================================================
# SepLLM KV Cache - Separator-aware eviction (preserves punctuation)
# =============================================================================

def get_separator_ids(tokenizer, separators=None):
    """
    Get token IDs for separator characters.
    
    Args:
        tokenizer: HuggingFace tokenizer
        separators: List of separator strings. Defaults to common punctuation.
        
    Returns:
        set of token IDs that are separators
    """
    if separators is None:
        separators = ['\n', '.', ',', '?', '!', ';', ':', '\n\n']
    
    separator_ids = set()
    
    for sep in separators:
        # Try encoding with and without space prefix
        for text in [sep, f' {sep}', f'{sep} ']:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
                separator_ids.update(ids)
            except:
                pass
        
        # Also try direct vocab lookup
        if hasattr(tokenizer, 'vocab'):
            for token, idx in tokenizer.vocab.items():
                # Check if token contains separator (handles Ġ prefix etc.)
                clean_token = token.replace('Ġ', ' ').replace('Ċ', '\n').strip()
                if clean_token == sep or clean_token == sep.strip():
                    separator_ids.add(idx)
    
    return separator_ids


class SepLLMKVCache:
    """
    SepLLM KV Cache: Separator-aware eviction policy.
    
    Improves on StreamingLLM by preserving separator tokens (punctuation, newlines)
    in addition to attention sink tokens. This maintains semantic boundaries in
    the context.
    
    Cache Structure (4 blocks):
        [Initial/Sink] + [Separator Cache] + [Past Window] + [Local Window]
    
    Eviction Policy:
        When tokens leave the local window:
        - If token is a separator: Move to Separator Cache
        - If token is not a separator: Drop it
        - If Separator Cache is full: Evict oldest separator
    
    Args:
        tokenizer: HuggingFace tokenizer for separator detection
        start_size: Number of initial sink tokens to keep
        local_size: Size of local (recent) window
        separator_size: Max separators to keep in separator cache
        k_seq_dim: Sequence dimension in key tensors
        v_seq_dim: Sequence dimension in value tensors
        separators: List of separator characters
        
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("pythia-2.8b-local")
        >>> kv_cache = SepLLMKVCache(tokenizer, start_size=4, local_size=256, separator_size=64)
        >>> # During generation:
        >>> outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        >>> past_key_values = kv_cache(outputs.past_key_values, input_ids)
    """
    def __init__(
        self,
        tokenizer,
        start_size=4,
        local_size=256,
        separator_size=64,
        k_seq_dim=2,
        v_seq_dim=2,
        separators=None,
    ):
        self.tokenizer = tokenizer
        self.start_size = start_size
        self.local_size = local_size
        self.separator_size = separator_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        
        # Total cache capacity
        self.cache_size = start_size + separator_size + local_size
        
        # Get separator token IDs
        self.separator_ids = get_separator_ids(tokenizer, separators)
        
        # Slice functions
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        # Separator KV buffer: list of (key, value) for each layer
        # Each layer stores list of (k_tensor, v_tensor) for separator tokens
        self.separator_buffer = None  # Will be initialized on first call
        
        # Track which tokens we've seen (for separator detection)
        self.token_history = []
        
        print(f"SepLLMKVCache: start={start_size}, local={local_size}, "
              f"separator={separator_size}, total_capacity={self.cache_size}")
        print(f"  Detected {len(self.separator_ids)} separator token IDs")
    
    def _is_separator(self, token_id):
        """Check if token_id is a separator."""
        return int(token_id) in self.separator_ids
    
    def _init_separator_buffer(self, past_key_values):
        """Initialize empty separator buffer matching layer structure."""
        num_layers = len(past_key_values)
        self.separator_buffer = [[] for _ in range(num_layers)]
    
    def _get_separator_kv(self, layer_idx):
        """Get concatenated separator KV for a layer."""
        if self.separator_buffer is None or len(self.separator_buffer[layer_idx]) == 0:
            return None, None
        
        layer_separators = self.separator_buffer[layer_idx]
        keys = torch.cat([s[0] for s in layer_separators], dim=self.k_seq_dim)
        values = torch.cat([s[1] for s in layer_separators], dim=self.v_seq_dim)
        return keys, values
    
    def _add_to_separator_buffer(self, past_key_values, position):
        """Add token at position to separator buffer (all layers)."""
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Extract single token KV
            k_token = self.k_slice(k, position, position + 1)
            v_token = self.v_slice(v, position, position + 1)
            
            self.separator_buffer[layer_idx].append((k_token, v_token))
            
            # Evict oldest if separator buffer is full
            if len(self.separator_buffer[layer_idx]) > self.separator_size:
                self.separator_buffer[layer_idx].pop(0)
    
    def __call__(self, past_key_values, input_ids=None):
        """
        Apply SepLLM eviction to past_key_values.
        
        The cache is structured as:
        [Initial Tokens] [Separator Cache] [Local Window]
        
        Args:
            past_key_values: HuggingFace model past_key_values
            input_ids: Current input token IDs (for separator detection)
            
        Returns:
            Updated past_key_values with separator-aware eviction
        """
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        # Initialize separator buffer if needed
        if self.separator_buffer is None:
            self._init_separator_buffer(past_key_values)
        
        # Track tokens for separator detection
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                new_tokens = input_ids.view(-1).tolist()
            else:
                new_tokens = list(input_ids)
            self.token_history.extend(new_tokens)
        
        # Current separator count
        num_separators = len(self.separator_buffer[0]) if self.separator_buffer else 0
        
        # Check if eviction needed
        # Effective local window = total - start - separators
        effective_local = self.cache_size - self.start_size - num_separators
        
        if seq_len <= self.start_size + effective_local:
            return past_key_values
        
        # Calculate how many tokens to process for eviction
        # Tokens between start_size and (seq_len - effective_local) need to be processed
        evict_start = self.start_size
        evict_end = seq_len - effective_local
        
        # Process tokens being evicted: check if they're separators
        for pos in range(evict_start, min(evict_end, len(self.token_history))):
            if pos < len(self.token_history):
                token_id = self.token_history[pos]
                if self._is_separator(token_id):
                    # Add to separator buffer
                    self._add_to_separator_buffer(past_key_values, pos)
        
        # Now construct the new cache: [Initial] + [Local Window]
        # (Separator tokens are stored separately and will be stitched during forward)
        new_past = []
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Keep initial tokens
            k_initial = self.k_slice(k, 0, self.start_size)
            v_initial = self.v_slice(v, 0, self.start_size)
            
            # Keep local window
            local_start = seq_len - effective_local
            k_local = self.k_slice(k, local_start, seq_len)
            v_local = self.v_slice(v, local_start, seq_len)
            
            # Concatenate: Initial + Local (separator buffer is separate)
            new_k = torch.cat([k_initial, k_local], dim=self.k_seq_dim)
            new_v = torch.cat([v_initial, v_local], dim=self.v_seq_dim)
            
            new_past.append([new_k, new_v])
        
        # Update token history to match new cache positions
        self.token_history = (
            self.token_history[:self.start_size] + 
            self.token_history[seq_len - effective_local:]
        )
        
        return new_past
    
    def get_full_kv(self, past_key_values):
        """
        Get full KV cache including separator tokens for attention computation.
        
        Use this to get [Initial + Separator + Local] for the forward pass.
        This ensures separator tokens are included in attention computation.
        
        Returns:
            past_key_values with separator tokens stitched in
        """
        if past_key_values is None:
            return None
        
        if self.separator_buffer is None or len(self.separator_buffer[0]) == 0:
            return past_key_values  # No separators to add
        
        new_past = []
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Get separator KV for this layer
            sep_k, sep_v = self._get_separator_kv(layer_idx)
            
            if sep_k is None:
                new_past.append([k, v])
                continue
            
            # Split into initial and local
            k_initial = self.k_slice(k, 0, self.start_size)
            v_initial = self.v_slice(v, 0, self.start_size)
            
            k_local = self.k_slice(k, self.start_size, k.size(self.k_seq_dim))
            v_local = self.v_slice(v, self.start_size, v.size(self.v_seq_dim))
            
            # Stitch: Initial + Separator + Local
            new_k = torch.cat([k_initial, sep_k, k_local], dim=self.k_seq_dim)
            new_v = torch.cat([v_initial, sep_v, v_local], dim=self.v_seq_dim)
            
            new_past.append([new_k, new_v])
        
        return new_past
    
    def reset(self):
        """Reset cache state for new sequence."""
        self.separator_buffer = None
        self.token_history = []


# =============================================================================
# KVPress-based Press Implementations
# =============================================================================

# Import BasePress here (after standalone StartRecentKVCache) to avoid circular imports
try:
    from accelerated_inference.kvpress.base_press import BasePress
    HAS_BASE_PRESS = True
except ImportError:
    # Fallback: create a dummy BasePress if kvpress is not available
    HAS_BASE_PRESS = False
    @dataclass
    class BasePress:
        """Dummy BasePress for when kvpress is not available."""
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            return keys, values

@dataclass
class StreamLLMPress(BasePress):
    """
    StreamingLLM: Keep initial tokens (sinks) and recent window.
    """
    compression_ratio: float = 0.0
    num_sinks: int = 4

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        seq_len = keys.shape[2]
        n_kept = int(seq_len * (1 - self.compression_ratio))
        
        if n_kept >= seq_len:
            return keys, values
            
        if n_kept <= self.num_sinks:
             # Edge case: kept is smaller than sinks, just keep recent 
             return keys[:, :, -n_kept:], values[:, :, -n_kept:]
        
        # Keep sinks
        sinks_k = keys[:, :, :self.num_sinks]
        sinks_v = values[:, :, :self.num_sinks]
        
        # Keep recent
        window_size = n_kept - self.num_sinks
        recent_k = keys[:, :, -window_size:]
        recent_v = values[:, :, -window_size:]
        
        return torch.cat([sinks_k, recent_k], dim=2), torch.cat([sinks_v, recent_v], dim=2)

@dataclass
class SnapKVPress(BasePress):
    """
    SnapKV: Select important KV pairs based on attention scores from a 'window' of observation.
    Simplified implementation for benchmarking.
    """
    compression_ratio: float = 0.0
    window_size: int = 32 # Observation window size
    kernel_size: int = 5 
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values
            
        seq_len = keys.shape[2]
        n_kept = int(seq_len * (1 - self.compression_ratio))
        
        # If we don't have attention scores (e.g. prefill), we can't prune effectively using SnapKV logic usually
        # But here 'attentions' might be passed?
        # In base_press, `output[1]` is passed as `attentions`.
        # If None, we cannot compress based on attention.
        
        if attentions is None:
            # Fallback to recent window (StreamingLLM style) if no attention scores
            return keys[:, :, -n_kept:], values[:, :, -n_kept:]
            
        # attentions shape: (bsz, num_heads, q_len, k_len)
        # We perform pruning based on the last few tokens' attention to the past
        
        # Take average attention over the observation window (last `window_size` queries)
        # We need to be careful with shapes.
        # If q_len is small (generation), we use it.
        
        # Sum attention over query dimension
        # attention_score: (bsz, num_heads, k_len)
        attention_score = attentions.sum(dim=-2) 
        
        # Select top-k
        indices = attention_score.topk(n_kept, dim=-1).indices
        indices = indices.sort(dim=-1).values # Sort to keep temporal order if needed/preferred
        
        # Gather
        # indices: (bsz, num_heads, n_kept)
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        
        keys = keys.gather(2, expanded_indices).contiguous()
        values = values.gather(2, expanded_indices).contiguous()
        
        return keys, values
