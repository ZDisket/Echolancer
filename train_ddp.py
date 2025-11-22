"""
Why do I have to write machine learning code? I should be eating ice cream and playing Hearts of Iron IV.

DDP-enabled training script for multi-GPU training on a single node.
"""
import os
# TF32 on ROCm is disabled unless we do this.
os.environ["HIPBLASLT_ALLOW_TF32"] = "1"
# CUDAgraphs prevent us from using grad acc and torch.compile
os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
torch.backends.cuda.enable_cudagraph_trees = False
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from utils.char_tokenizer import CharTokenizer

import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
import math
from functools import reduce
from operator import mul
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np
from tqdm import tqdm
from data import PreprocessedVQAudioDataset
from concat_dataset import ConcatTextAudioSharded, ShardAwareRandomSampler, collate_stack, DistributedShardAwareSampler
from paired_dataset_concat import TTSPairedVQConcatDataset, tts_paired_vq_concat_collate
from model.echolancer import Echolancer
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import wandb
from utils.logger import WandbLogger
import random



def is_main_process():
    """
    Check if this is the main process (rank 0).
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """
    Print only on the main process (rank 0) to avoid duplicate output in multi-GPU training.
    
    Args:
        *args, **kwargs: Same arguments as built-in print()
    """
    if is_main_process():
        print(*args, **kwargs)


# Import Transformer Engine if available
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print_rank0("WARNING: Transformer Engine not available. FP8 training will not be possible.")

# Import NeuCodecFE if available, otherwise set a flag
try:
    from neucodecfe import NeuCodecFE
    NEUCODEC_AVAILABLE = True
except ImportError:
    print_rank0("Warning: neucodec module not available. Testing with audio generation will be disabled.")
    NEUCODEC_AVAILABLE = False



def setup_distributed():
    """
    Initialize the distributed environment for single-node multi-GPU training.
    Uses environment variables set by torchrun.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Initialize process group for single-node training
    if world_size > 1:
        # Set the device for this process BEFORE init_process_group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',  # NCCL is best for GPU training
            init_method='env://',  # Use environment variables
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f'cuda:{local_rank}')  # Explicitly specify device to avoid NCCL warning
        )
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()



def get_model(model):
    """
    Get the underlying model, unwrapping DDP if necessary.
    
    Args:
        model: The model, potentially wrapped with DistributedDataParallel
    
    Returns:
        The underlying model
    """
    model_unwrapped = model
    # Unwrap torch.compile wrapper (OptimizedModule)
    if hasattr(model_unwrapped, "_orig_mod"):
        model_unwrapped = model_unwrapped._orig_mod
    
    # Unwrap DDP wrapper
    if isinstance(model_unwrapped, DDP):
        model_unwrapped = model_unwrapped.module
        
    # Unwrap torch.compile wrapper again (in case of DDP(Compile(model)))
    if hasattr(model_unwrapped, "_orig_mod"):
        model_unwrapped = model_unwrapped._orig_mod
        
    return model_unwrapped




class TrainingConfig:
    """Configuration object to hold all training parameters"""
    def __init__(self, **kwargs):
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.grad_acc_step = kwargs.get('grad_acc_step', 1)
        self.log_interval = kwargs.get('log_interval', 10)
        self.val_interval_steps = kwargs.get('val_interval_steps', 500)
        self.test_interval_steps = kwargs.get('test_interval_steps', 1000)
        self.save_interval_steps = kwargs.get('save_interval_steps', 1000)
        self.max_val_steps = kwargs.get('max_val_steps', 50)
        self.val_split_ratio = kwargs.get('val_split_ratio', 0.01)
        self.max_val_samples = kwargs.get('max_val_samples', 2500)
        self.n_test_audios = kwargs.get('n_test_audios', 10)
        self.gpu_flops = kwargs.get('gpu_flops', 0)
        self.eos_bos_id = kwargs.get('eos_bos_id', (-1, -2))
        
        # Model parameters
        self.seq_len = kwargs.get('seq_len', 2048)
        self.text_vocab_size = kwargs.get('text_vocab_size', 256)
        self.audio_vocab_size = kwargs.get('audio_vocab_size', 64000)
        
        # LoRA parameters
        self.lora_rank = kwargs.get('lora_rank', 0)
        self.lora_alpha = kwargs.get('lora_alpha', 16)
        self.lora_dropout = kwargs.get('lora_dropout', 0.0)
        self.lora_scale = kwargs.get('lora_scale', 1.0)
        
        # Paths and directories
        self.shards_dir = kwargs.get('shards_dir', '')
        self.output_dir = kwargs.get('output_dir', 'pretrain_checkpoints')
        self.tokenizer_path = kwargs.get('tokenizer_path', None)
        self.shard_type = kwargs.get('shard_type', 0)

        self.test_texts = kwargs.get('test_texts', None)

        if self.test_texts is None:
            self.test_texts = ["The rain stopped just as the train pulled into the station, leaving the air smelling like wet concrete and cold metal",
                               "She looked out over the ocean, wondering how something so vast could feel so calm and familiar",
                               "The old clock in the hallway still chimed every hour, even though nobody in the house listened anymore",
                               "I took a wrong turn somewhere, but the road was quiet and the sunset made it hard to care",
                               "Some nights, the city feels alive, lights flicker, windows hum, and the wind carries a thousand whispered stories",
                               "When the sun sets behind the hills, the quiet hum of the city turns into a soft, endless melody"]
        
        # Device and mixed precision
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = kwargs.get('mixed_precision', 'none')

        self.test_speakers = kwargs.get('test_speakers', None)
        
        # Model specific
        self.model_config = kwargs.get('model_config', {})
        self.training_params = kwargs.get('training_params', {})


def get_mixed_precision_config(mixed_precision):
    if mixed_precision in ["fp8", "fp8_autocast"]:
        fp8_recipe = recipe.DelayedScaling(
            fp8_format=recipe.Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max",
        )
        # Return a factory that creates a NEW fp8_autocast each time
        return lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe), None

    elif mixed_precision in ["fp16", "float16"]:
        # torch.autocast requires device_type in recent PyTorch
        return lambda: torch.autocast(device_type="cuda", dtype=torch.float16), GradScaler()

    elif mixed_precision in ["bf16", "bfloat16"]:
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16), None

    else:
        # Disabled mixed precision
        return lambda: nullcontext(), None


def cross_entropy_with_disabled_autocast(logits, targets, **kwargs):
    """Calculate cross entropy loss with autocast disabled and inputs cast to float."""
    # Use the newer torch.amp.autocast API to avoid deprecation warning
    if hasattr(torch.amp, 'autocast'):  # For newer PyTorch versions
        with torch.amp.autocast('cuda', enabled=False):
            # Cast logits to float32 for numerical stability in loss calculation
            # Cross-entropy expects targets as long type, so ensure they're long
            return F.cross_entropy(
                logits.float(),
                targets.long(),  # Convert targets to long (int64) which is expected by cross_entropy
                **kwargs
            )
    else:  # For older PyTorch versions
        with autocast(enabled=False):
            # Cast logits to float32 for numerical stability in loss calculation
            # Cross-entropy expects targets as long (int64) which is expected by cross_entropy
            return F.cross_entropy(
                logits.float(),
                targets.long(),  # Convert targets to long (int64) which is expected by cross_entropy
                **kwargs
            )


# We are using a very manual implementation of cross entropy
# This is because F.cross_entropy was causing ROCm to shit the bed during pretraining
# TODO: Fix.
def cross_entropy_text_audio(
    logits,                 # (B, L, V)
    targets,                # (B, L)
    text_vocab_size,
    true_vocab_size=65583,
    ignore_index=None,
    target_lengths=None,    # (B,), optional
    input_lengths=None,     # (B,), optional
    token_id_weights=None   # dict: {token_id: weight}, e.g. {audio_eos_id: 3.0}
):
    logits  = logits.to(torch.float32).contiguous()
    targets = targets.to(torch.long).contiguous()

    B, L, V = logits.shape
    assert targets.shape[:2] == (B, L)

    if true_vocab_size < V:
        logits = logits[..., :true_vocab_size]
        V = true_vocab_size

    if ignore_index is None:
        ignore_index = -100

    device = logits.device

    if target_lengths is not None:
        target_lengths = target_lengths.to(device)
    if input_lengths is not None:
        input_lengths = input_lengths.to(device)

    if target_lengths is None and input_lengths is None:
        len_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    else:
        eff_len = torch.full((B,), L, device=device)
        if target_lengths is not None:
            eff_len = torch.minimum(eff_len, target_lengths)
        if input_lengths is not None:
            eff_len = torch.minimum(eff_len, input_lengths)
        t = torch.arange(L, device=device).unsqueeze(0)
        len_mask = t < eff_len.unsqueeze(1)

    class_pad = targets >= true_vocab_size
    valid = len_mask & (~class_pad)
    user_ignored = (targets == ignore_index)
    valid = valid & (~user_ignored)

    safe_tg = torch.where(valid, targets, torch.zeros_like(targets))

    logp   = F.log_softmax(logits, dim=-1)
    picked = logp.gather(-1, safe_tg.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # ----- per-position weights -----
    weights = torch.ones(B, L, device=device)
    if token_id_weights:
        for tid, w in token_id_weights.items():
            mask_tid = (safe_tg == int(tid)) & valid
            if isinstance(w, torch.Tensor):
                w = w.to(device)
            weights[mask_tid] = w

    # zero-out invalids so they don't contribute to sums
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    denom = weights.sum()
    if denom > 0:
        total_ce = (-(picked) * weights).sum() / denom
    else:
        total_ce = torch.zeros((), device=device)

    text_mask  = valid & (safe_tg < text_vocab_size)
    audio_mask = valid & (safe_tg >= text_vocab_size) & (safe_tg < true_vocab_size)

    # For monitoring, keep unweighted class-specific CEs (change to weighted if you prefer)
    text_ce  = (-(picked[text_mask])).mean()  if text_mask.any()  else torch.zeros((), device=device)
    audio_ce = (-(picked[audio_mask])).mean() if audio_mask.any() else torch.zeros((), device=device)

    return total_ce, text_ce, audio_ce




def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following a cosine curve after a warmup period.
    
    Args:
        optimizer: The optimizer to apply the learning rate schedule to
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of the initial learning rate (default 0.1)
        last_epoch: The index of last epoch (default -1)
    
    Returns:
        A learning rate scheduler object
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            if num_warmup_steps == 0:
                return 1.0
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def load_configs(train_config_path, model_config_path, pretrain_config_path=None):
    """Load training, model, and pretraining configurations from YAML files."""
    with open(train_config_path, 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # Load pretrain config if provided, otherwise use defaults
    if pretrain_config_path:
        with open(pretrain_config_path, 'r', encoding='utf-8') as f:
            pretrain_config = yaml.safe_load(f)
    else:
        # Default pretrain configuration
        pretrain_config = {
            'pretraining': {
                'vq_chunk_length': 512,
                'log_interval': 10,
                'save_interval_steps': 1000,
                'val_interval_steps': 500,  # New: validation interval in steps
                'max_val_steps': 50,
                'val_split_ratio': 0.01,  # 1% for validation by default
                'max_val_samples': 2500,  # Maximum 2500 samples in validation
                'output_dir': 'pretrain_checkpoints',
                'testing': {
                    'test_interval_steps': 1000,
                    'n_test_audios': 10
                }
            },
            'training': {
                'epochs': 10
            },
            'model': {
                'pretraining_mode': True,
                'multi_speaker': False,
                'n_speaker': 0
            }
        }
    
    return train_config, model_config, pretrain_config


def estimate_flops_forward_pass(batch_size, seq_len, model_params):
    """
    Estimate TRAINING FLOPs for a single optimizer step of a decoder-only transformer
    (decoder, self-attention only; no cross-attn). Uses 2·m·k·n for matmul FLOPs.

    NOTE: Despite the name, this returns TRAINING FLOPs (≈ forward + backward wrt weights
    + backward wrt activations). If activation checkpointing is enabled, FLOPs increase.

    Args:
        batch_size: int
        seq_len: int OR {'decoder': Sd}
        model_params: dict with:
            - 'decoder': {hidden:int, layer:int, forward_expansion:int?}
            - optional 'checkpointing': bool (default False)
            - optional 'train_flop_factor': float override (default 3.0 or 4.0 if checkpointing)

    Returns:
        training_flops_per_step (float)
    """

    # Resolve sequence length for decoder
    if isinstance(seq_len, dict):
        dec_S = int(seq_len.get('decoder', 0))
        if dec_S <= 0:
            raise ValueError("When seq_len is a dict, provide positive 'decoder' length.")
    else:
        dec_S = int(seq_len)

    B = int(batch_size)

    # Decoder params (only decoder for decoder-only training)
    dec_d  = int(model_params['decoder']['hidden'])
    dec_L  = int(model_params['decoder']['layer'])
    dec_ff = int(model_params['decoder'].get('forward_expansion', 4))

    # Train cost factor
    use_ckpt = bool(model_params.get('checkpointing', False))
    train_factor = float(model_params.get('train_flop_factor', 4.0 if use_ckpt else 3.0))

    # FLOPs per transformer layer (self-attn only)
    # Proj (Q,K,V,O): 4 matmuls → 8·B·S·d²
    # Attn matmuls (QK^T and A·V): → 4·B·S²·d
    # MLP (up & down): 4·B·S·d·(ff_exp·d)
    def flops_block_self_attn(B, S, d, ff_exp):
        proj = 8.0 * B * S * (d ** 2)
        attn = 4.0 * B * (S ** 2) * d
        mlp  = 4.0 * B * S * d * (ff_exp * d)
        return proj + attn + mlp

    forward_dec = dec_L * flops_block_self_attn(B, dec_S, dec_d, dec_ff)

    # Convert to TRAINING FLOPs per step (forward + backward for decoder only)
    return forward_dec * train_factor


def calculate_mfu(total_flops, time_elapsed, gpu_flops):
    """
    MFU (%) = (Achieved FLOPs/s) / (GPU peak FLOPs/s) * 100
    Pass TRAINING FLOPs for 'total_flops' (e.g., output of estimate_flops_forward_pass).
    """
    if time_elapsed <= 0 or gpu_flops <= 0:
        return 0.0
    achieved = float(total_flops) / float(time_elapsed)  # FLOPs/s
    return (achieved / float(gpu_flops)) * 100.0


def train_epoch(model, dataloader, optimizer, config, global_step, text_tokenizer, epoch_num=0, validation_fn=None,
                val_dataloader=None, wandb_logger=None, autocast_ctx=None, grad_scaler=None, scheduler=None, has_compiled=False, skip_steps=0):
    """
    Train for one epoch with concatenated text-audio sequences.
    The model learns to predict the next token in the concatenated sequence.
    """
    model.train()
    total_loss = 0
    total_steps = 0
    
    # Determine the number of steps based on the dataset
    steps_per_epoch = len(dataloader)
    
    # Initialize variables for MFU calculation
    log_start_time = time.time()
    
    # Create tqdm progress bar for the epoch (only on main process)
    if is_main_process():
        epoch_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch_num + 1}", leave=False)
    else:
        epoch_pbar = None
    
    # Create iterator
    data_iter = iter(dataloader)
    
    for step in range(steps_per_epoch):
        # Update progress bar on main process
        if epoch_pbar is not None:
            epoch_pbar.update(1)
        # Skip steps if resuming from a checkpoint in the middle of an epoch
        if step < skip_steps:
            # Skip this step by fetching the batch but not processing it
            try:
                next(data_iter)
            except StopIteration:
                break
            continue
            
        # Process concatenated sequence data


        try:
            # Get batch of concatenated sequences (B, seq_len)
            batch = next(data_iter)

            batch_size, seq_len, inputs, targets, sequence_lengths, speakers = extract_batch(batch, config)

            # For the decoder, we want to predict the next token in the concatenated sequence
            decoder_input = inputs.clone()  # Use all but last token as input
            decoder_target = targets.clone()  # Use all but first token as target

            # Forward pass for the model
            with autocast_ctx():
                torch.compiler.cudagraph_mark_step_begin()
                decoder_logits = model(decoder_input, sequence_lengths, speakers, None)

            # Calculate decoder loss - predict the next token in sequence
            total_ce_loss, text_loss, audio_loss = cross_entropy_text_audio(
                decoder_logits,
                decoder_target,
                config.text_vocab_size,
                ignore_index=-200,
                target_lengths=sequence_lengths,
                token_id_weights={get_model(model).decoder.eos_token_id: 3.0},
            )

        except StopIteration:
            break

        # Only decoder loss for decoder-only training (using total_loss from cross_entropy_text_audio)
        total_batch_loss = total_ce_loss

        # Apply gradient accumulation
        # Scale the loss by grad_acc_step to maintain the same effective batch size
        scaled_loss = total_batch_loss / config.grad_acc_step

        if step % config.grad_acc_step == 0:
            # Zero gradients at the beginning of accumulation cycle
            optimizer.zero_grad(set_to_none=True)

        if grad_scaler is not None:
            # Scale the loss and perform backward pass with gradient scaling
            grad_scaler.scale(scaled_loss).backward()
        else:
            # Standard backward pass without gradient scaling
            scaled_loss.backward()

        # Step optimizer and scheduler every grad_acc_step
        if (step + 1) % config.grad_acc_step == 0 or (step + 1) == steps_per_epoch:
            if grad_scaler is not None:
                # Unscaled gradients for gradient clipping
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                # Step with scaler
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                # Standard gradient clipping and step without gradient scaling
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            # Step the learning rate scheduler if it exists
            if scheduler is not None:
                scheduler.step()

        total_loss += total_batch_loss.item()
        total_steps += 1

        # Update global step
        current_global_step = global_step + step + 1

        if step % config.log_interval == 0:
            # Calculate perplexity from total loss only
            total_perplexity = torch.exp(torch.tensor(total_batch_loss.item() if hasattr(total_batch_loss, 'item') else total_batch_loss))
            text_perplexity = torch.exp(torch.tensor(text_loss.item() if hasattr(text_loss, 'item') else text_loss))
            audio_perplexity = torch.exp(torch.tensor(audio_loss.item() if hasattr(audio_loss, 'item') else audio_loss))
            
            # Calculate MFU if gpu_flops is available
            mfu = None
            if config.gpu_flops > 0 and config.model_config is not None:
                log_elapsed_time = time.time() - log_start_time
                sample_batch_size = config.batch_size
                seq_len_param = seq_len  # Using full sequence length now
                
                # Calculate FLOPs for concatenated sequence processing in each step
                concat_flops_per_step = estimate_flops_forward_pass(sample_batch_size, seq_len_param, config.model_config['model'])
                flops_per_step = concat_flops_per_step  # Only concatenated processing now
                
                # Calculate FLOPs for the steps since the last log
                total_flops_since_last_log = flops_per_step * config.log_interval
                
                # Calculate MFU for the time period since the last log
                mfu = calculate_mfu(total_flops_since_last_log, log_elapsed_time, config.gpu_flops)
                
                # Reset the timer for the next log interval
                log_start_time = time.time()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else config.learning_rate
            print_str = f"Global Step {current_global_step}/{global_step + steps_per_epoch}, "
            print_str += f"Total Loss: {total_batch_loss.item():.4f}, "
            print_str += f"Text Loss: {text_loss.item():.4f}, "
            print_str += f"Audio Loss: {audio_loss.item():.4f}, "
            print_str += f"Total PPL: {total_perplexity:.2f}, "
            print_str += f"Text PPL: {text_perplexity:.2f}, "
            print_str += f"Audio PPL: {audio_perplexity:.2f}, "
            print_str += f"LR: {current_lr:.2e}"
            
            if mfu is not None:
                print_str += f", MFU: {mfu:.2f}%"
            
            # Only print on main process
            print_rank0(print_str)
            
            # Log training metrics to wandb (with text and audio losses separately) - only on main process
            if wandb_logger is not None and is_main_process():
                train_metrics = {
                    "train/loss": total_batch_loss.item(),
                    "train/text_loss": text_loss.item() if hasattr(text_loss, 'item') else text_loss,
                    "train/audio_loss": audio_loss.item() if hasattr(audio_loss, 'item') else audio_loss,
                    "train/total_perplexity": total_perplexity.item(),
                    "train/text_perplexity": text_perplexity.item(),
                    "train/audio_perplexity": audio_perplexity.item(),
                    "train/lr": current_lr,
                }
                
                # Add MFU to the logged metrics if available
                if mfu is not None:
                    train_metrics["train/mfu"] = mfu
                    
                wandb_logger.log_metrics(train_metrics, current_global_step)

        # Update progress bar description with current loss (only on main process)
        if epoch_pbar is not None:
            epoch_pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'text_loss': f'{text_loss.item():.4f}',
                'audio_loss': f'{audio_loss.item():.4f}',
                'step': f'{step}/{steps_per_epoch}'
            })

        # Perform validation if interval is set and current step matches the validation interval
        if config.val_interval_steps > 0 and current_global_step % config.val_interval_steps == 0 and validation_fn is not None and config.shard_type != 0 and is_main_process():
            print_rank0(f"Performing validation at step {current_global_step}")
            val_loss = validation_fn(model, val_dataloader, config, autocast_ctx, wandb_logger, current_global_step, optimizer)
            print_rank0(f"Validation Loss at step {current_global_step}: {val_loss:.4f}")
            torch.cuda.empty_cache()


        # Perform testing if interval is set and current step matches the test interval (only on main process)
        if config.test_interval_steps > 0 and current_global_step % config.test_interval_steps == 0 and is_main_process():
            print_rank0(f"Generating audio test samples at step {current_global_step}")
            test_output_dir = os.path.join(config.output_dir, f"test_audio_step_{current_global_step}")
            generate_audio_test(model, config, text_tokenizer, wandb_logger, current_global_step)
            torch.cuda.empty_cache()

        # Save checkpoint every save_interval_steps (only on main process)
        if config.save_interval_steps > 0 and current_global_step % config.save_interval_steps == 0 and is_main_process():
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_step_{current_global_step}.pt")
            
            # Get the actual model (unwrap DDP if necessary)
            model_to_save = get_model(model)
            
            if config.lora_rank > 0:
                # For LoRA, save both LoRA parameters and potentially base model state
                # We can save just the LoRA parameters for efficiency
                lora_state_dict = {k: v for k, v in model_to_save.state_dict().items() 
                                   if 'lora_A' in k or 'lora_B' in k}
                torch.save({
                    'global_step': current_global_step,
                    'model_state_dict': lora_state_dict,  # Only save LoRA parameters
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lora_config': {
                        'lora_rank': config.lora_rank,
                        'lora_alpha': config.lora_alpha,
                        'lora_dropout': config.lora_dropout,
                        'lora_scale': config.lora_scale
                    }
                }, checkpoint_path)
            else:
                torch.save({
                    'global_step': current_global_step,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
            print_rank0(f"Checkpoint saved to {checkpoint_path}")
    
    if epoch_pbar is not None:
        epoch_pbar.close()

    return total_loss / max(total_steps, 1) if total_steps > 0 else 0, total_steps

def prepare_text_tokens(text, text_tokenizer, model, device, add_bos=True, add_eos=True):
    # 1) Encode text into token IDs
    inp_seq = text_tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    # 2) Append decoder BOS token for the audio segment
    inp_seq.append(int(get_model(model).decoder.bos_token_id))

    # 3) Convert to tensor and batch it (1, L)
    x_in = torch.tensor(inp_seq, dtype=torch.long, device=device).unsqueeze(0)

    return x_in



def generate_audio_test(model, config, text_tokenizer, wandb_logger, step=None):
    """
    Generate audio samples using unconditional inference for testing during pretraining.

    Args:
        model: The Echolancer model
        config: TrainingConfig object containing all configuration parameters
        step: Current training step for logging
    """
    if not NEUCODEC_AVAILABLE:
        print_rank0("NeuCodec not available. Skipping audio generation test.")
        return

    print_rank0(f"Generating {config.n_test_audios} audio samples for testing...")

    # Create output directory
    output_dir = os.path.join(config.output_dir, f"test_audio_step_{step}" if step is not None else "test_audio")
    os.makedirs(output_dir, exist_ok=True)

    # Set model to evaluation mode
    model.eval()
    torch.cuda.empty_cache()

    # Initialize NeuCodecFE for this test
    neu_codec_fe = NeuCodecFE(is_cuda=config.device.type == 'cuda', offset=get_model(model).decoder.vocab_offset)

    # Generate speaker embeddings for testing (if applicable)
    # Using random speaker embeddings for testing purposes
    n_seqs = min(config.n_test_audios, 4)  # Keep batch size reasonable
    if get_model(model).speaker_channels > 0:
        speakers = config.test_speakers
    else:
        speakers = None

    if speakers is None:
        speakers = [0] * n_seqs  # just use first speaker

    for n in range(0, n_seqs):
        with torch.no_grad(), torch.compiler.set_stance("force_eager"):

            if config.shard_type == 2:
                spk = speakers[n]
                spk = torch.tensor([spk], dtype=torch.long, device=config.device) # (B,)
            elif config.shard_type == 1:
                spk = speakers[n].to(config.device).unsqueeze(0) # (1, spk_hidden)
            elif config.shard_type == 0:
                spk = None # Pretraining has no speaker cond

            text_input = prepare_text_tokens(config.test_texts[n], text_tokenizer, model, device=config.device)
            generated_tokens = get_model(model).infer(text_input, spk, max_length=768, top_p=0.92, temperature=0.8) # (1, T)

            # Decode and save with NeuCodec

            # first remove audio EOS
            generated_tokens = generated_tokens[:, :-1]

            generated_tokens = generated_tokens.unsqueeze(1) # (1, 1, T) as codec expects

            vq_vocab_size = config.model_config['model']['vq_vocab_size']
            generated_tokens = generated_tokens.clamp(0, vq_vocab_size - 1) # clamp to valid range, just in case.

            reconstructed_audio = neu_codec_fe.decode_codes(generated_tokens)

            # Process audio for saving and logging
            audio_to_save = reconstructed_audio[0, 0, :].cpu()  # (T,) format for saving
            audio_for_logging = audio_to_save.numpy()  # Convert to numpy for wandb logging

            # Normalize audio to avoid clipping
            if audio_to_save.abs().max() > 0:
                audio_to_save = audio_to_save / audio_to_save.abs().max() * 0.9  # 0.9 to avoid clipping
                audio_for_logging = audio_for_logging / np.max(
                    np.abs(audio_for_logging)) * 0.9  # Normalize for logging as well

            # Save audio file
            audio_file_path = os.path.join(output_dir, f"generated_audio_sample_{n}.wav")
            torchaudio.save(audio_file_path, audio_to_save.unsqueeze(0), 24000)
            print_rank0(f"Saved generated audio to: {audio_file_path}")

            # Log audio to wandb if logger is provided
            if step is not None and wandb_logger is not None:
                wandb_logger.log_audio(
                    audio=audio_for_logging,
                    sample_rate=24000,
                    name=f"generated_audio_sample_{n}",
                    step=step
                )

    print_rank0(f"Completed generating audio samples.")


def validate(model, dataloader, config, autocast_ctx=None, wandb_logger=None, step=None, optimizer=None):
    """
    Validate the model on concatenated text-audio sequences only.
    The model learns to predict the next token in the concatenated sequence.
    """
    model.eval()
    torch.cuda.empty_cache()
    total_loss = 0.0  # This will be accumulated as a scalar value, then converted to tensor when needed
    total_steps = 0
    total_decoder_loss = 0.0  # This will be accumulated from tensor .item() values
    total_text_loss = 0.0
    total_audio_loss = 0.0

    with torch.no_grad():

        # Determine the number of validation steps based on the dataset
        steps_per_epoch = len(dataloader)
        max_steps = min(steps_per_epoch, config.max_val_steps)

        # Create iterator
        data_iter = iter(dataloader)

        # Create tqdm progress bar for validation (only on main process)
        if is_main_process():
            val_pbar = tqdm(range(max_steps), desc="Validation", leave=False)
        else:
            val_pbar = None

        for step_idx in val_pbar:
            # Validate on concatenated sequences only
            try:
                # Get batch of concatenated sequences (B, seq_len)
                batch = next(data_iter)
                batch_size, seq_len, inputs, targets, sequence_lengths, speakers = extract_batch(batch, config)

                # For the decoder, we want to predict the next token in the concatenated sequence
                decoder_input = inputs  # Use all but last token as input
                decoder_target = targets  # Use all but first token as target


                # Forward pass for the model
                with autocast_ctx():
                    decoder_logits = model(decoder_input, sequence_lengths, speakers, None)

                # Calculate decoder loss - predict the next token in sequence
                total_ce_loss, text_loss, audio_loss = cross_entropy_text_audio(
                    decoder_logits,
                    decoder_target,
                    config.text_vocab_size,
                    ignore_index=-200,
                    target_lengths=sequence_lengths,
                    token_id_weights={get_model(model).decoder.eos_token_id: 3.0},
                )



            except StopIteration:
                break

            # Use total loss from cross_entropy_text_audio for validation
            total_batch_loss = total_ce_loss
            total_loss += total_batch_loss.item()
            total_decoder_loss += total_batch_loss.item() if hasattr(total_batch_loss, 'item') else total_batch_loss
            total_text_loss += text_loss.item() if hasattr(text_loss, 'item') else text_loss
            total_audio_loss += audio_loss.item() if hasattr(audio_loss, 'item') else audio_loss
            total_steps += 1

            # If we're in zero shot mode and have no test speakers, supply test speaker embeddings from the
            # validation dataloader.
            if config.shard_type == 1 and config.test_speakers is None and is_main_process():
                config.test_speakers = speakers
                print_rank0(f"Eval::Zero shot mode detected, but no test speakers configured. Supplying with {speakers.size(0)} speakers from the validation set.")

            # Update progress bar description with current loss (only on main process)
            if val_pbar is not None:
                val_pbar.set_postfix({
                    'loss': f'{total_batch_loss.item():.4f}',
                    'text_loss': f'{text_loss.item():.4f}',
                    'audio_loss': f'{audio_loss.item():.4f}',
                    'step': f'{step_idx + 1}/{max_steps}'
                })

        # Close the validation progress bar
        if val_pbar is not None:
            val_pbar.close()

    avg_loss = total_loss / max(total_steps, 1) if total_steps > 0 else 0
    avg_decoder_loss = total_decoder_loss / max(total_steps, 1) if total_steps > 0 else 0
    avg_text_loss = total_text_loss / max(total_steps, 1) if total_steps > 0 else 0
    avg_audio_loss = total_audio_loss / max(total_steps, 1) if total_steps > 0 else 0
    
    # Calculate perplexity from average loss
    avg_decoder_perplexity = torch.exp(torch.tensor(avg_decoder_loss))
    avg_text_perplexity = torch.exp(torch.tensor(avg_text_loss))
    avg_audio_perplexity = torch.exp(torch.tensor(avg_audio_loss))
    
    # Log validation metrics to wandb if logger is provided (with text and audio losses separately) - only on main process
    if wandb_logger is not None and step is not None and is_main_process():
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0.0
        validation_metrics = {
            "val/loss": avg_loss,
            "val/total_loss": avg_decoder_loss,
            "val/text_loss": avg_text_loss,
            "val/audio_loss": avg_audio_loss,
            "val/total_perplexity": avg_decoder_perplexity.item(),
            "val/text_perplexity": avg_text_perplexity.item(),
            "val/audio_perplexity": avg_audio_perplexity.item(),
            "val/lr": current_lr,
        }
        wandb_logger.log_metrics(validation_metrics, step)
    
    return avg_loss


def extract_batch(batch, config):
    # Pretraining shard
    if config.shard_type == 0:
        inputs, targets = batch
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        batch_size, seq_len = inputs.shape

        return batch_size, seq_len, inputs, targets, None, None
    else: # Finetuning shard.
        inputs, targets, sequence_lengths, speakers = batch["input_ids"], batch["target_ids"], batch["seq_lens"], batch["speakers"]

        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        sequence_lengths = sequence_lengths.to(config.device)
        speakers = speakers.to(config.device)

        batch_size, seq_len = inputs.shape
        return batch_size, seq_len, inputs, targets, sequence_lengths, speakers



def param_groups_no_decay(model):
    decay, no_decay = [], []
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    norm_like = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    tied = set()
    # if you tie embedding and lm_head: ensure they share storage (common in LMs)
    # e.g., model.lm_head.weight = model.embed.weight

    for module_name, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = "%s.%s" % (module_name, name) if module_name else name

            if name.endswith("bias"):
                no_decay.append(param);  continue

            if isinstance(module, norm_like):
                # layernorm/groupnorm/batchnorm weights
                no_decay.append(param);  continue

            if isinstance(module, (nn.Embedding,)):
                no_decay.append(param);  continue

            if isinstance(module, whitelist_weight_modules):
                decay.append(param);     continue

            if isinstance(module, blacklist_weight_modules):
                no_decay.append(param);  continue

            # Fallback: if it looks like a weight but not in whitelist, be conservative
            if name.endswith("weight"):
                decay.append(param)
            else:
                no_decay.append(param)

    # Deduplicate while preserving order
    def uniq(params):
        seen = set()
        out = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    return [
        {"params": uniq(decay), "weight_decay": 0.01},
        {"params": uniq(no_decay), "weight_decay": 0.0},
    ]

def create_collate_fn(shard_type):
    """Create a collate function with the proper BOS/EOS tokens."""

    def collate_fn(batch):
        return tts_paired_vq_concat_collate(
            batch,
            pad_id=0,
            shard_type=shard_type,
        )

    return collate_fn


def load_pretrained(model, checkpoint_path, device="cpu", lora_enabled=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Detect LoRA presence
    is_lora_checkpoint = any(("lora_A" in k) or ("lora_B" in k) for k in state_dict.keys())

    # ---- 1) Clean prefixes ---------------------------------------------------------
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod."):]] = v
        elif k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    # ---- 2) Map keys between LoRA and non-LoRA layouts ----------------------------
    target_keys = set(model.state_dict().keys())

    def try_map_key(k):
        if k in target_keys:
            return k
        for leaf in (".weight", ".bias"):
            if k.endswith(leaf):
                candidate = k[:-len(leaf)] + ".original_layer" + leaf
                if candidate in target_keys:
                    return candidate
        if ".original_layer." in k:
            candidate = k.replace(".original_layer.", ".")
            if candidate in target_keys:
                return candidate
        return None

    remapped = {}
    for k, v in cleaned.items():
        new_k = try_map_key(k)
        if new_k is not None:
            remapped[new_k] = v
        else:
            # keep LoRA adapter keys if checkpoint has them and model expects them
            if is_lora_checkpoint and k in target_keys:
                remapped[k] = v

    if is_lora_checkpoint and not lora_enabled:
        print_rank0("Warning: LoRA checkpoint detected, but lora_enabled=False. Ignoring LoRA adapters.")

    # ---- 3) Load model ------------------------------------------------------------
    load_info = model.load_state_dict(remapped, strict=False)
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)

    # ---- 4) Compute parameter counts ---------------------------------------------
    def param_count(key_list):
        total = 0
        for k in key_list:
            if k in model.state_dict():
                total += reduce(mul, model.state_dict()[k].shape, 1)
        return total

    total_params = sum(reduce(mul, p.shape, 1) for p in model.state_dict().values())
    missing_params = param_count(missing)
    unexpected_params = sum(reduce(mul, remapped[k].shape, 1) for k in unexpected if k in remapped)

    pct_missing = 100 * missing_params / total_params if total_params > 0 else 0
    pct_unexpected = 100 * unexpected_params / total_params if total_params > 0 else 0

    print_rank0(f"Loaded checkpoint: {checkpoint_path}")
    print_rank0(f"Missing params: {missing_params:,}  ({pct_missing:.4f}%)")
    print_rank0(f"Unexpected params: {unexpected_params:,}  ({pct_unexpected:.4f}%)")
    print_rank0(f"Total params in model: {total_params:,}")

    if is_lora_checkpoint:
        print_rank0("Detected LoRA checkpoint.")
    else:
        print_rank0("Detected base checkpoint.")

    if lora_enabled and hasattr(model, "enable_lora"):
        try:
            model.enable_lora()
        except Exception:
            pass

    return model



import torch.nn as nn

def unfreeze_last_n_layers_with_tied_head(model, n_last):
    # Unwrap DDP if necessary
    actual_model = get_model(model)
    
    # 0) freeze everything
    for p in actual_model.parameters():
        p.requires_grad = False

    # 1) unfreeze last N transformer layers
    dec_layers = actual_model.decoder.dec.layers
    n = len(dec_layers)
    if n_last > 0:
        for layer in dec_layers[max(0, n - n_last):]:
            for p in layer.parameters():
                p.requires_grad = True

    # 2) re-tie head to embedding (defensive) and unfreeze both
    if hasattr(actual_model, "combined_emb") and hasattr(actual_model, "combined_head"):
        if hasattr(actual_model.combined_head, "weight") and hasattr(actual_model.combined_emb, "weight"):
            if actual_model.combined_head.weight.data.shape == actual_model.combined_emb.weight.data.shape:
                # share the same tensor to keep tying strict
                actual_model.combined_head.weight = actual_model.combined_emb.weight
        for p in actual_model.combined_emb.parameters():
            p.requires_grad = True
        for p in actual_model.combined_head.parameters():
            p.requires_grad = True  # includes bias if present

    # 3) unfreeze all LayerNorms globally (includes layer.norm1, layer.norm3)
   # for m in actual_model.modules():
    #    if isinstance(m, nn.LayerNorm):
     #       for p in m.parameters():
      #          p.requires_grad = True

    print_rank0(f"Only trainable params are last {n_last} layers")
    return model


def find_latest_checkpoint(output_dir):
    """
    Find the latest checkpoint file in the output directory.
    
    Args:
        output_dir (str): Directory to search for checkpoints
        
    Returns:
        str or None: Path to the latest checkpoint file, or None if no checkpoints found
    """
    import re
    import os
    from pathlib import Path
    
    # Pattern to match checkpoint files: checkpoint_step_<number>.pt
    checkpoint_pattern = re.compile(r'checkpoint_step_(\d+)\.pt')
    
    checkpoint_files = []
    for file_path in Path(output_dir).glob("checkpoint_step_*.pt"):
        match = checkpoint_pattern.search(file_path.name)
        if match:
            step_num = int(match.group(1))
            checkpoint_files.append((step_num, str(file_path)))
    
    if not checkpoint_files:
        return None
    
    # Sort by step number in descending order and return the first (highest step)
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_files[0][1]



def main():
    parser = argparse.ArgumentParser(description="Pretrain Echolancer decoder only with concatenated data")

    # Configuration paths
    parser.add_argument("--train_config", type=str, required=True,
                        help="Path to the training configuration YAML file")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to the model configuration YAML file")
    parser.add_argument("--pretrain_config", type=str, default=None,
                        help="Path to the pretraining configuration YAML file (optional)")

    # Data configuration - using concatenated text-audio shards
    parser.add_argument("--shards_dir", type=str, required=True,
                        help="Directory containing preprocessed .pt shard files with text-audio pairs")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to saved tokenizer")

    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model path for transfer learning")
    parser.add_argument("--shard_type", type=int, default=0,
                        help="Type of shard. 0 is pretraining, 1 and 2 are finetuning shards")

    # Additional configuration overrides (optional)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")

    parser.add_argument("--out_dir", type=str, default="out_default1",
                        help="Output directory. Override")
    
    parser.add_argument("--load_checkpoint", action="store_true",
                        help="Load the latest checkpoint from the output folder and continue training")

    args = parser.parse_args()

    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device based on local rank
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device)
    
    # Only print on main process
    print_rank0(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    print_rank0(f"Device: {device}")

    # Load configurations from YAML files
    train_config, model_config, pretrain_config = load_configs(args.train_config, args.model_config, args.pretrain_config)

    # Extract parameters from configs
    # Model config extraction
    model_params = model_config['model']
    decoder_params = model_params['decoder']

    # Training config extraction
    training_params = train_config['training']
    optimizer_params = train_config['optimizer']

    # Pretraining config extraction
    pretraining_params = pretrain_config['pretraining']
    pretrain_model_params = pretrain_config.get('model', {})

    # Extract mixed precision setting from training config
    mixed_precision = train_config.get('mixed_precision', 'none')
    print_rank0(f"Mixed precision setting: {mixed_precision}")

    # Initialize Weights & Biases logging (only on main process)
    wandb_logger = None
    if is_main_process():
        train_config['wandb']['name'] = f"echolancer-pretrainv2-{int(time.time())}"
        wandb_logger = WandbLogger(train_config)

    # Initialize training configuration object
    config = TrainingConfig(
        batch_size=training_params.get('batch_size', 16),
        num_epochs=training_params.get('epochs', 50),
        learning_rate=optimizer_params.get('learning_rate', 1e-4),
        weight_decay=optimizer_params.get('weight_decay', 0.01),
        max_grad_norm=optimizer_params.get('grad_clip_thresh', 1.0),
        grad_acc_step=optimizer_params.get('grad_acc_step', 1),
        log_interval=training_params.get('log_step', 10) * optimizer_params.get('grad_acc_step', 1),
        val_interval_steps=training_params.get('val_step', 999999999999),
        test_interval_steps=training_params.get('test_step', 5000),
        save_interval_steps=training_params.get('save_step', 1000),
        max_val_steps=pretraining_params.get('max_val_steps', 50),
        val_split_ratio=pretraining_params.get('val_split_ratio', 0.01),
        max_val_samples=pretraining_params.get('max_val_samples', 2500),
        n_test_audios=pretraining_params.get('testing', {}).get('n_test_audios', 10),
        gpu_flops=train_config.get('gpu_flops', 0),
        seq_len=training_params.get('seq_len', 2048),
        text_vocab_size=len(CharTokenizer()) if hasattr(CharTokenizer(), '__len__') else CharTokenizer().get_vocab_size(),
        audio_vocab_size=model_params.get('vq_vocab_size', 64000),
        lora_rank=train_config.get('lora_rank', 0),
        lora_alpha=train_config.get('lora_alpha', 16),
        lora_dropout=train_config.get('lora_dropout', 0.0),
        lora_scale=train_config.get('lora_scale', 1.0),
        shards_dir=args.shards_dir,
        output_dir=args.out_dir,
        tokenizer_path=args.tokenizer_path,
        device=device,  # Use the distributed device
        mixed_precision=mixed_precision,
        model_config=model_config,
        training_params=training_params,
        shard_type=args.shard_type,
    )

    # Set device in global scope for use in audio generation
    #torch.global_device = config.device

    # Initialize mixed precision components based on config
    autocast_ctx, grad_scaler = get_mixed_precision_config(config.mixed_precision)
    if autocast_ctx is not None and is_main_process():
        print_rank0(f"Using mixed precision: {config.mixed_precision}")
    elif is_main_process():
        print_rank0("Mixed precision disabled")

    # Create output directory (only on main process)
    if is_main_process():
        os.makedirs(config.output_dir, exist_ok=True)
    
    # Synchronize all processes before continuing
    if dist.is_initialized():
        dist.barrier()


    # Create model with pretraining mode enabled (cross-attention disabled)
    print_rank0("Creating model...")

    # Initialize text tokenizer
    text_tokenizer = CharTokenizer()
    
    # Use vocab size that accommodates both text and audio tokens
    # Text tokens + audio offset (which is handled by the dataset)
    text_vocab_size = len(text_tokenizer) if hasattr(text_tokenizer, '__len__') else text_tokenizer.get_vocab_size()
    audio_vocab_size = model_params.get('vq_vocab_size', 64000)  # This should match the audio token range
    total_vocab_size = text_vocab_size + audio_vocab_size + 2  # +2 for audio BOS/EOS tokens

    # Check if we're using FP8 precision and set use_te accordingly
    use_te = config.mixed_precision == "fp8" or config.mixed_precision == "fp8_autocast"

    model = Echolancer(
        vocab_size=text_vocab_size,
        decoder_hidden=decoder_params['hidden'],
        decoder_layer=decoder_params['layer'],
        decoder_head=decoder_params['head'],
        decoder_dropout=decoder_params['dropout'],
        emotion_channels=0,
        speaker_channels=model_params['speaker_channels'],
        multi_speaker=model_params['multi_speaker'],
        n_speaker=model_params['n_speaker'],
        alibi_alpha=model_params['alibi_alpha'],
        use_alibi=model_params['use_alibi'],
        activation=model_params.get('activation', 'relu'),
        vq_token_mode=False,
        vq_vocab_size=model_params['vq_vocab_size'],
        decoder_kv_heads=model_params['decoder_kv_heads'],
        decoder_start_i=0,
        emotion_input_size=0,  # This could be configurable too
        emotion_hidden_sizes=[512, 384],  # This could be configurable too
        emotion_dropout=0.1,  # This could be configurable too
        pretraining_mode=pretrain_model_params.get('pretraining_mode', True),  # Enable pretraining mode to disable cross-attention
        use_te=use_te,  # Enable Transformer Engine if using FP8 precision
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_scale=config.lora_scale,
        zero_shot_mode=model_params['zero_shot_mode'],
        use_macaron=model_params['use_macaron'],
    )

    if args.pretrained is not None and is_main_process():
        print_rank0(f"Loading pretrained model from {args.pretrained}")
        model = load_pretrained(model, args.pretrained, lora_enabled=config.lora_rank > 0)

    if train_config.get('freeze_to', 0) > 0 and is_main_process():
        print_rank0("Doing partial training.")
        unfreeze_last_n_layers_with_tied_head(model, train_config['freeze_to'])

    model = model.to(config.device)

    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print_rank0(f"Model wrapped with DistributedDataParallel")

    # Create concatenated text-audio dataset
    print_rank0("Creating concatenated text-audio dataset...")

    if config.shard_type == 0:
        concat_dataset = ConcatTextAudioSharded(
            shards_dir=config.shards_dir,
            text_tokenizer=text_tokenizer,
            seq_len=config.seq_len,
            audio_bos_id=get_model(model).decoder.bos_token_id,
            audio_eos_id=get_model(model).decoder.eos_token_id,
            max_shard_cache=16,
        )

        # Training data loader setup with sampler for concatenated data
        # Use DistributedSampler for multi-GPU, otherwise use ShardAwareRandomSampler
        # Use DistributedShardAwareSampler for multi-GPU to preserve IO locality
        if world_size > 1:
            train_sampler = DistributedShardAwareSampler(
                concat_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=42,
                drop_last=True,
                block_utterances=4096
            )
        else:
            train_sampler = DistributedShardAwareSampler(
                concat_dataset,
                num_replicas=1, # For single GPU, treat as 1 replica
                rank=0,         # For single GPU, rank is 0
                shuffle=True,
                seed=42,
                drop_last=True,
                block_utterances=4096
            )
        collate_fun = collate_stack
    else:
        concat_dataset = TTSPairedVQConcatDataset(config.shards_dir,preload=True,
                                                  text_tokenizer=text_tokenizer,shard_type=config.shard_type,
                                                  audio_bos_id=get_model(model).decoder.bos_token_id,
                                                  audio_eos_id=get_model(model).decoder.eos_token_id,
                                                  max_shards=None)
        # Use DistributedSampler for multi-GPU, otherwise use default sampler
        if world_size > 1:
            train_sampler = DistributedSampler(
                concat_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=42
            )
        else:
            train_sampler = None  # Use default sampler
        collate_fun = create_collate_fn(config.shard_type)
        train_config["enable_torch_compile"] = False # Turn off torch.compile for varseqlen.
        torch.backends.cudnn.benchmark = False


    print_rank0(f"Concatenated dataset size: {len(concat_dataset)}")
    
    # Calculate validation size
    val_size = min(int(len(concat_dataset) * config.val_split_ratio), config.max_val_samples)
    train_size = len(concat_dataset) - val_size

    print_rank0(f"Dataset split: {train_size} train, {val_size} validation")

    # Create data loaders using config values
    n_workers = ((os.cpu_count() // 2) + (os.cpu_count() // 4)) // world_size
    

    train_dataloader = DataLoader(
        concat_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fun,
        prefetch_factor=2,
    )
    
    # For validation, use DistributedSampler for multi-GPU or SubsetRandomSampler for single GPU
    # since we don't necessarily need shard-aware sampling for validation
    if world_size > 1:
        # Create a subset of indices for validation
        val_indices = list(range(train_size, len(concat_dataset)))
        val_subset = torch.utils.data.Subset(concat_dataset, val_indices)
        val_sampler = DistributedSampler(
            val_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Don't shuffle validation data
            drop_last=False
        )
        val_dataset = val_subset
    else:
        val_sampler = torch.utils.data.SubsetRandomSampler(list(range(train_size, len(concat_dataset)))) if config.shard_type == 0 else None
        val_dataset = concat_dataset
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size // 2,
        sampler=val_sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fun,
    )


    # Create optimizer using config values
    # If LoRA is enabled, only optimize LoRA parameters
    if config.lora_rank > 0:
        get_model(model).enable_lora()  # Enable LoRA training
        print_rank0(f"LoRA enabled with rank={config.lora_rank}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
        optimizer = AdamW(
            get_model(model).get_lora_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True
        )
        print_rank0(f"Training only {len(get_model(model).get_lora_parameters())} LoRA parameters")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True
        )
        print_rank0(f"Training all {sum(p.numel() for p in model.parameters())} parameters")

    # Load checkpoint if --load_checkpoint is enabled and a checkpoint exists
    if args.load_checkpoint:
        latest_checkpoint = find_latest_checkpoint(config.output_dir)
        if latest_checkpoint:
            print_rank0(f"Loading checkpoint from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=config.device)
            
            # Load model state dict
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Detect if this is a LoRA checkpoint and adjust loading accordingly
            is_lora_checkpoint = any(("lora_A" in k) or ("lora_B" in k) for k in model_state_dict.keys())
            if config.lora_rank > 0 and not is_lora_checkpoint:
                # If model uses LoRA but checkpoint doesn't have LoRA weights, try to load base weights
                model_state_dict = {k: v for k, v in model_state_dict.items() if 'lora' not in k}
            
            get_model(model).load_state_dict(model_state_dict, strict=False)
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Resume from the global step stored in checkpoint
            global_step = checkpoint.get('global_step', 0)
            print_rank0(f"Resuming training from step {global_step}")
        else:
            print_rank0(f"No checkpoint found in {config.output_dir}, starting from scratch")
    else:
        print_rank0("Not loading any checkpoint, starting from scratch")

    # Training loop
    # Create learning rate scheduler if specified in config
    scheduler_config = train_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'constant')
    warmup_steps = scheduler_config.get('warmup_steps', 0)

    if config.grad_acc_step > 1:
        warmup_steps = warmup_steps // config.grad_acc_step
    
    # Calculate total training steps for the scheduler
    steps_per_epoch = len(train_dataloader)
    total_training_steps = config.num_epochs * steps_per_epoch
    
    # Adjust steps for gradient accumulation
    if config.grad_acc_step > 1:
        # Calculate effective steps accounting for gradient accumulation
        effective_steps_per_epoch = (steps_per_epoch + config.grad_acc_step - 1) // config.grad_acc_step  # Round up division
        total_training_steps = config.num_epochs * effective_steps_per_epoch
        # Warmup steps have already been adjusted earlier
    
    scheduler = None
    if scheduler_type == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        print_rank0(f"Using warmup cosine scheduler: {warmup_steps} warmup steps, {total_training_steps} total steps")
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps)
        print_rank0(f"Using cosine annealing scheduler: {total_training_steps} total steps")
    else:
        print_rank0("Using constant learning rate (no scheduler)")

    # Log model information to wandb
    if wandb_logger is not None:
        # Calculate and log model parameters in millions (M)
        decoder_param = sum(p.numel() for p in get_model(model).decoder.parameters())
        total_param = sum(p.numel() for p in model.parameters())
        
        model_metrics = {
            "model/decoder_params_M": decoder_param / 1e6,
            "model/total_params_M": total_param / 1e6,
            "model/decoder_params": decoder_param,
            "model/total_params": total_param
        }
        wandb_logger.log_metrics(model_metrics)
        
        print_rank0(f"Number of Decoder Parameters: {decoder_param / 1e6:.2f}M") 
        print_rank0(f"Total Number of Parameters: {total_param / 1e6:.2f}M")

    print_rank0("Starting training...")


    # Check if torch.compile is available and enabled in config
    enable_torch_compile = train_config.get('enable_torch_compile', False)
    if enable_torch_compile and hasattr(torch, 'compile'):
        print_rank0("Compiling whole model with torch.compile...")
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        print_rank0("Model compilation completed.")
    elif enable_torch_compile:
        print_rank0("torch.compile requested but not available in this PyTorch version.")
    else:
        print_rank0("torch.compile not enabled in configuration.")
    
    # Calculate total steps for the entire training based on VQ training dataset (larger dataset)
    steps_per_epoch = len(train_dataloader)
    total_steps = config.num_epochs * steps_per_epoch
    
    # Initialize global_step to 0
    global_step = 0
    
    # Load checkpoint if --load_checkpoint is enabled and a checkpoint exists
    if args.load_checkpoint:
        latest_checkpoint = find_latest_checkpoint(config.output_dir)
        if latest_checkpoint:
            print_rank0(f"Loading checkpoint from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=config.device)
            
            # Load model state dict
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Detect if this is a LoRA checkpoint and adjust loading accordingly
            is_lora_checkpoint = any(("lora_A" in k) or ("lora_B" in k) for k in model_state_dict.keys())
            if config.lora_rank > 0 and not is_lora_checkpoint:
                # If model uses LoRA but checkpoint doesn't have LoRA weights, try to load base weights
                model_state_dict = {k: v for k, v in model_state_dict.items() if 'lora' not in k}
            
            get_model(model).load_state_dict(model_state_dict, strict=False)
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Resume from the global step stored in checkpoint
            checkpoint_global_step = checkpoint.get('global_step', 0)
            
            # Calculate epoch and step within epoch to resume from
            steps_per_epoch = len(train_dataloader)
            resumed_epoch = checkpoint_global_step // steps_per_epoch
            step_in_epoch = checkpoint_global_step % steps_per_epoch
            
            print_rank0(f"Resuming training from global step {checkpoint_global_step}")
            print_rank0(f"This corresponds to epoch {resumed_epoch}, step {step_in_epoch} in that epoch")
            print_rank0(f"Will skip {step_in_epoch} steps in the resumed epoch's dataloader")
            
            # Set the starting global step
            global_step = checkpoint_global_step
        else:
            print_rank0(f"No checkpoint found in {config.output_dir}, starting from scratch")
            resumed_epoch = 0
            step_in_epoch = 0
            global_step = 0
    else:
        print_rank0("Not loading any checkpoint, starting from scratch")
        resumed_epoch = 0
        step_in_epoch = 0
        global_step = 0
        
    for epoch in range(resumed_epoch, config.num_epochs):
        # Set epoch for distributed sampler to ensure proper shuffling
        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        
        print_rank0(f"Epoch {epoch + 1}/{config.num_epochs}")

        # Determine if this is the first epoch being resumed, and how many steps to skip
        skip_steps = 0
        if epoch == resumed_epoch and step_in_epoch > 0:
            skip_steps = step_in_epoch
            print_rank0(f"Resuming from step {step_in_epoch} in epoch {epoch}, will skip first {skip_steps} steps")

        start_time = time.time()
        train_loss, steps_completed = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            config, 
            global_step,  # Pass current global step to train_epoch
            text_tokenizer,
            epoch,  # Pass current epoch number
            validate,  # Pass validation function  
            val_dataloader,
            wandb_logger,  # Pass wandb logger
            autocast_ctx, 
            grad_scaler,
            scheduler,  # Pass learning rate scheduler
            skip_steps=skip_steps  # Pass number of steps to skip at the beginning of epoch
        )
        epoch_time = time.time() - start_time

        # Calculate epoch-level perplexity from average loss
        epoch_perplexity = torch.exp(torch.tensor(train_loss))
        print_rank0(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train PPL: {epoch_perplexity:.2f}")
        # Increment global steps after each epoch
        global_step += steps_completed
        # Reshuffle preloaded varlen dataset after every epoch.
        if config.shard_type > 0:
            concat_dataset.shuffle_order()
    
    # Log final metrics and finish wandb run (only on main process)
    if wandb_logger is not None and is_main_process():
        wandb_logger.finish()
    
    # Clean up distributed training
    cleanup_distributed()
    
    print_rank0("Training completed!")


if __name__ == "__main__":
    main()