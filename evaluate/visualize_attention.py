"""
Attention Heatmap Visualization for Pythia-2.8B

This script visualizes attention patterns across all layers of the model.
It generates heatmaps showing which tokens attend to which.

Usage:
    python evaluate/visualize_attention.py \
        --model_name_or_path pythia-2.8b-local \
        --text "Hello, how are you? I am fine, thank you!" \
        --output_dir outputs/attention_viz
"""

import sys
import os

# Auto-detect project root and add to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, cannot generate plots")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="pythia-2.8b-local",
        help="Path to the model"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog. What a beautiful day, isn't it?",
        help="Input text to visualize attention for (should contain punctuation)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/attention_viz",
        help="Output directory for heatmaps"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to visualize (default: all layers)"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Which attention head to visualize (default: 0, use -1 for average)"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        default=12,
        help="Figure size (width=height)"
    )
    
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


def get_attention_weights(model, tokenizer, text):
    """
    Get attention weights for all layers.
    
    Returns:
        attentions: tuple of tensors, each [batch, heads, seq_len, seq_len]
        tokens: list of token strings
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    
    # Get token strings for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            return_dict=True,
        )
    
    # outputs.attentions is tuple of (batch, heads, seq, seq) tensors
    attentions = outputs.attentions
    
    return attentions, tokens


def plot_attention_heatmap(attention, tokens, layer_idx, head_idx, output_path, figsize=12):
    """
    Plot a single attention heatmap.
    
    Args:
        attention: [seq_len, seq_len] tensor
        tokens: list of token strings
        layer_idx: layer index
        head_idx: head index (or "avg" for average)
        output_path: path to save the figure
        figsize: figure size
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Convert to numpy
    attn = attention.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    # Plot heatmap
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    seq_len = len(tokens)
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    
    # Clean up token names for display
    display_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in tokens]
    
    ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(display_tokens, fontsize=8)
    
    # Labels
    ax.set_xlabel('Key (attended to)', fontsize=12)
    ax.set_ylabel('Query (attending from)', fontsize=12)
    
    head_str = f"Head {head_idx}" if isinstance(head_idx, int) else "Average"
    ax.set_title(f'Attention Heatmap - Layer {layer_idx}, {head_str}', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_all_layers_summary(attentions, tokens, output_path, figsize=16):
    """
    Plot a summary grid showing all layers' attention patterns.
    """
    if not HAS_MATPLOTLIB:
        return
    
    num_layers = len(attentions)
    
    # Determine grid size
    cols = 8
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize, figsize * rows / cols))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        # Get attention for this layer, averaged over heads
        attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()  # [seq, seq]
        
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'L{layer_idx}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Attention Patterns Across All {num_layers} Layers\n(averaged over heads)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Summary saved: {output_path}")


def main():
    args = parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Attention Visualization Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Text: \"{args.text}\"")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model(args.model_name_or_path)
    
    # Get attention weights
    print("\nComputing attention weights...")
    attentions, tokens = get_attention_weights(model, tokenizer, args.text)
    
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = len(tokens)
    
    print(f"\nModel info:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Tokens: {tokens}")
    
    # Determine which layers to visualize
    if args.layers is not None:
        layers_to_viz = args.layers
    else:
        # Visualize selected layers: first, middle layers, last
        layers_to_viz = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        layers_to_viz = sorted(set(layers_to_viz))
    
    print(f"\nVisualizing layers: {layers_to_viz}")
    print(f"Visualizing head: {args.head if args.head >= 0 else 'average'}")
    
    # Generate individual layer heatmaps
    print("\nGenerating heatmaps...")
    for layer_idx in layers_to_viz:
        if layer_idx >= num_layers:
            print(f"  Skipping layer {layer_idx} (out of range)")
            continue
        
        # Get attention for this layer: [batch, heads, seq, seq]
        layer_attn = attentions[layer_idx][0]  # [heads, seq, seq]
        
        if args.head == -1:
            # Average over all heads
            attn = layer_attn.mean(dim=0)  # [seq, seq]
            head_str = "avg"
        else:
            head_idx = min(args.head, num_heads - 1)
            attn = layer_attn[head_idx]  # [seq, seq]
            head_str = f"h{head_idx}"
        
        output_path = os.path.join(args.output_dir, f"layer_{layer_idx:02d}_{head_str}.png")
        plot_attention_heatmap(
            attn, tokens, layer_idx, 
            args.head if args.head >= 0 else "avg",
            output_path, args.figsize
        )
    
    # Generate summary plot (all layers)
    summary_path = os.path.join(args.output_dir, "all_layers_summary.png")
    plot_all_layers_summary(attentions, tokens, summary_path)
    
    # Save token info
    info_path = os.path.join(args.output_dir, "info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Text: {args.text}\n")
        f.write(f"Layers: {num_layers}\n")
        f.write(f"Heads: {num_heads}\n")
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"\nTokens:\n")
        for i, token in enumerate(tokens):
            f.write(f"  {i}: {repr(token)}\n")
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files generated:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
