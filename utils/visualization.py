import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

def plot_attention_map(attention_weights: torch.Tensor, 
                      title: str = "Attention Map",
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor of shape (seq_len_q, seq_len_k)
        title: Title for the plot
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
    """
    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar=True,
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": .8}
    )
    
    plt.title(title)
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_attention_maps_grid(attention_weights: torch.Tensor,
                            titles: Optional[List[str]] = None,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Plot multiple attention maps in a grid.
    
    Args:
        attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        titles: Titles for each subplot
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
    """
    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    batch_size, num_heads, seq_len_q, seq_len_k = attention_weights.shape
    
    # Determine grid size
    n_plots = min(batch_size * num_heads, 16)  # Limit to 16 plots for readability
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots > 1:
        axes = axes.flatten()
    
    # Plot each attention map
    for i in range(n_plots):
        if n_plots > 1:
            ax = axes[i]
        else:
            ax = axes
        
        batch_idx = i // num_heads
        head_idx = i % num_heads
        
        # Get attention weights
        weights = attention_weights[batch_idx, head_idx]
        
        # Plot heatmap
        im = ax.imshow(weights, cmap='viridis', aspect='auto')
        ax.set_title(f"Batch {batch_idx}, Head {head_idx}" if not titles else titles[i])
        ax.set_xlabel("Key Positions")
        ax.set_ylabel("Query Positions")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        if n_plots > 1:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves(metrics_history: Dict[str, List[float]],
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot training curves (loss, accuracy, etc.).
    
    Args:
        metrics_history: Dictionary mapping metric names to lists of values
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
    """
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot metrics
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_history)))
    
    for i, (metric_name, values) in enumerate(metrics_history.items()):
        ax1.plot(values, label=metric_name, color=colors[i], linewidth=2)
    
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Metric Values")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.title("Training Curves")
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_spectrogram(spectrogram: torch.Tensor,
                    title: str = "Spectrogram",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot spectrogram as a heatmap.
    
    Args:
        spectrogram: Spectrogram tensor of shape (time_steps, freq_bins)
        title: Title for the plot
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
    """
    # Convert to numpy if needed
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot spectrogram
    plt.imshow(
        spectrogram.T,  # Transpose for time on x-axis, frequency on y-axis
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    
    plt.colorbar(label="Magnitude")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_length_distribution(lengths: List[int],
                           title: str = "Sequence Length Distribution",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot distribution of sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        title: Title for the plot
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    plt.axvline(mean_len, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_len:.1f}')
    plt.axvline(median_len, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_len:.1f}')
    plt.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Example: Plot attention map
    attention = torch.randn(10, 15)  # (query_len=10, key_len=15)
    plot_attention_map(attention, title="Sample Attention Map")
    
    # Example: Plot spectrogram
    spectrogram = torch.randn(100, 80)  # (time_steps=100, freq_bins=80)
    plot_spectrogram(spectrogram, title="Sample Spectrogram")