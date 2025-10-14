import torch
import numpy as np
import os

def get_model(vocab_size, encoder_hidden=256, encoder_head=4, encoder_layer=4,
              decoder_hidden=256, decoder_layer=4, decoder_head=4,
              encoder_dropout=0.1, decoder_dropout=0.1, mel_channels=80,
              emotion_channels=256, speaker_channels=32, multi_speaker=False, n_speaker=0,
              use_alibi=False, alibi_alpha=1.0, activation='relu',
              vq_token_mode=False, vq_vocab_size=1024,
              encoder_kv_heads=None, decoder_kv_heads=None,
              emotion_input_size=768, emotion_hidden_sizes=[512, 384], emotion_dropout=0.1):
    """
    Create an Echolancer model with specified parameters.
    """
    try:
        # Try relative import first (when used as a package)
        from .model.echolancer import Echolancer
    except ImportError:
        # Fall back to absolute import (when running scripts directly)
        from model.echolancer import Echolancer
    
    model = Echolancer(
        vocab_size=vocab_size,
        encoder_hidden=encoder_hidden,
        encoder_head=encoder_head,
        encoder_layer=encoder_layer,
        decoder_hidden=decoder_hidden,
        decoder_layer=decoder_layer,
        decoder_head=decoder_head,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        mel_channels=mel_channels,
        emotion_channels=emotion_channels,
        speaker_channels=speaker_channels,
        multi_speaker=multi_speaker,
        n_speaker=n_speaker,
        use_alibi=use_alibi,
        alibi_alpha=alibi_alpha,
        activation=activation,
        vq_token_mode=vq_token_mode,
        vq_vocab_size=vq_vocab_size,
        encoder_kv_heads=encoder_kv_heads,
        decoder_kv_heads=decoder_kv_heads,
        emotion_input_size=emotion_input_size,
        emotion_hidden_sizes=emotion_hidden_sizes,
        emotion_dropout=emotion_dropout
    )
    return model

def get_param_num(model):
    """
    Count the number of parameters in a model.
    """
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def load_checkpoint(model, optimizer, checkpoint_path, device, scaler=None):
    """
    Load model and optimizer state from a checkpoint file.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scaler state if provided and available in checkpoint
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    return model, optimizer

def save_checkpoint(model, optimizer, checkpoint_path, step, scaler=None):
    """
    Save model and optimizer state to a checkpoint file.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    
    # Add scaler state if provided
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    torch.save(
        checkpoint,
        f"{checkpoint_path}_{step}.pth"
    )
    print(f"Saved checkpoint to {checkpoint_path}_{step}.pth")

def sequence_mask(lengths, max_len=None):
    """
    Create a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= lengths.unsqueeze(1)

def to_device(data, device):
    """
    Move data to the specified device.
    """
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

# Import utility modules
from .config import load_config, save_config, set_seed, get_device, update_config
from .scheduler import get_scheduler
from .early_stopping import EarlyStopping
from .metrics import compute_accuracy, compute_mae, compute_rmse, compute_metrics
from .visualization import plot_attention_map, plot_attention_maps_grid, plot_training_curves, plot_spectrogram
from .augmentation import DataAugmenter, add_noise_to_spectrogram, time_stretch_spectrogram, pitch_shift_spectrogram
from .profiling import Profiler, get_global_profiler, profile_function, profile_block, print_profiling_stats, reset_profiling
from .char_tokenizer import CharTokenizer