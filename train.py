import argparse
import os
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import numpy as np
import wandb

# Import our standalone modules
from model import Echolancer, EcholancerLoss
from utils import (
    get_model, get_param_num, save_checkpoint, load_checkpoint,
    load_config, set_seed, get_device, get_scheduler, EarlyStopping,
    CharTokenizer
)
from utils.logger import WandbLogger
from data import create_dummy_data_files, get_data_loader

# Import pipeline components
from bertfe import BERTFrontEnd
from scripted_preencoder import ScriptedPreEncoder
from istftnetfe import ISTFTNetFE

# Check for mixed precision support
def get_amp_dtype():
    """Get the appropriate dtype for mixed precision training."""
    if torch.cuda.is_available():
        # Check if bfloat16 is supported (available on Ampere and newer GPUs)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            # Use float16 as fallback
            return torch.float16
    else:
        # CPU doesn't benefit from mixed precision
        return torch.float32

# Global variables for AMP
AMP_DTYPE = get_amp_dtype()
USE_AMP = AMP_DTYPE != torch.float32
USE_GRAD_SCALER = AMP_DTYPE == torch.float16  # Only use grad scaler with float16

print(f"AMP enabled: {USE_AMP}")
print(f"AMP dtype: {AMP_DTYPE}")
print(f"Grad scaler needed: {USE_GRAD_SCALER}")

def validate(model, val_loader, loss_fn, device, epoch):
    """
    Validate the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False):
            # Move batch to device
            speakers = batch[2].to(device)
            texts = batch[3].to(device)
            src_lens = batch[4].to(device)
            mels = batch[5].to(device)
            mel_lens = batch[6].to(device)
            em_hidden = batch[7].to(device)
            
            # Forward pass with mixed precision if enabled
            if USE_AMP:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    model_out = model(speakers, texts, src_lens, mels, mel_lens, em_hidden)
                    
                    # Compute loss
                    batch_for_loss = (batch[0], batch[1], speakers, texts, src_lens, mels, mel_lens, em_hidden)
                    losses = loss_fn(batch_for_loss, model_out)
            else:
                model_out = model(speakers, texts, src_lens, mels, mel_lens, em_hidden)
                
                # Compute loss
                batch_for_loss = (batch[0], batch[1], speakers, texts, src_lens, mels, mel_lens, em_hidden)
                losses = loss_fn(batch_for_loss, model_out)
            
            val_loss += losses[0].item()
            val_steps += 1
    
    model.train()
    return val_loss / val_steps


def test_model(model, bert_model, pre_encoder, istftnet, test_sentences, test_speakers, device, step, wandb_logger=None, test_vq_tokens=None, test_vq_token_lens=None):
    """
    Test the model on novel sentences using the full pipeline:
    Echolancer -> Pre-encoder -> ISTFTNet
    
    Args:
        model: Trained Echolancer model
        bert_model: BERT model for emotion encoding
        pre_encoder: Scripted pre-encoder for VQGAN decoding
        istftnet: ISTFTNet for waveform generation
        test_sentences: List of test sentences
        test_speakers: List of speaker IDs for test sentences (or indices for VQ tokens)
        device: Device to run inference on
        step: Current training step
        wandb_logger: Logger for wandb integration
        test_vq_tokens: List of VQ tokens for zero-shot testing (optional)
        test_vq_token_lens: List of VQ token lengths for zero-shot testing (optional)
    """
    model.eval()
    
    print(f"Testing model at step {step}...")
    
    # Initialize tokenizer
    tokenizer = CharTokenizer()
    
    with torch.no_grad():
        for idx, (sentence, speaker_id) in enumerate(zip(test_sentences, test_speakers)):
            try:
                print(f"  Testing sentence {idx+1}/{len(test_sentences)}: '{sentence}' with speaker {speaker_id}")
                
                # 1. Get emotion encoding from BERT
                em_blocks, em_hidden = bert_model.infer(sentence)
                
                # 2. Tokenize text using our CharTokenizer
                text_tokens = tokenizer.encode(sentence)
                text_tokens = torch.tensor([text_tokens], dtype=torch.long, device=device)
                src_lens = torch.tensor([len(text_tokens[0])], dtype=torch.long, device=device)
                
                # 3. Prepare speaker information based on mode
                if model.vq_token_mode:
                    # Zero-shot mode: use VQ tokens if provided, otherwise skip this test
                    if test_vq_tokens is not None and idx < len(test_vq_tokens):
                        vq_tokens = test_vq_tokens[idx].unsqueeze(0).to(device)  # Add batch dimension
                        vq_token_lens = test_vq_token_lens[idx].unsqueeze(0).to(device) if test_vq_token_lens else None
                        speakers = None  # Not used in zero-shot mode
                    else:
                        print(f"    Skipping zero-shot test for sentence {idx+1} (no VQ tokens provided)")
                        continue
                else:
                    # Traditional mode: use speaker ID
                    speakers = torch.tensor([speaker_id], dtype=torch.long, device=device)
                    vq_tokens = None
                    vq_token_lens = None
                
                # 4. Run Echolancer inference with mixed precision if enabled
                if USE_AMP:
                    with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                        if model.vq_token_mode:
                            # Zero-shot mode: provide VQ tokens
                            token_outputs = model.infer(
                                speakers, text_tokens, src_lens, 
                                em_hidden=em_hidden,
                                vq_tokens=vq_tokens, 
                                vq_token_lens=vq_token_lens
                            )
                        else:
                            # Traditional mode: use speaker IDs
                            token_outputs = model.infer(
                                speakers, text_tokens, src_lens, 
                                em_hidden=em_hidden
                            )
                else:
                    if model.vq_token_mode:
                        # Zero-shot mode: provide VQ tokens
                        token_outputs = model.infer(
                            speakers, text_tokens, src_lens, 
                            em_hidden=em_hidden,
                            vq_tokens=vq_tokens, 
                            vq_token_lens=vq_token_lens
                        )
                    else:
                        # Traditional mode: use speaker IDs
                        token_outputs = model.infer(
                            speakers, text_tokens, src_lens, 
                            em_hidden=em_hidden
                        )
                
                # 5. Decode indices to mel spectrogram using pre-encoder
                # Note: token_outputs may need to be adjusted based on the actual model output
                mels = pre_encoder.decode(token_outputs)
                
                # 6. Convert mel spectrogram to waveform using ISTFTNet
                # Note: The exact format may need to be adjusted based on ISTFTNet requirements
                waveform = istftnet.infer(mels)
                
                # 7. Log results
                if wandb_logger is not None:
                    # Log the generated waveform
                    wandb_logger.log_audio(
                        waveform, 
                        sample_rate=istftnet.sampling_rate if hasattr(istftnet, 'sampling_rate') else 22050,
                        name=f"test_sentence_{idx+1}_step_{step}",
                        step=step
                    )
                    
                print(f"    Successfully generated audio for sentence {idx+1}")
                
            except Exception as e:
                print(f"    Error processing sentence {idx+1}: {str(e)}")
                continue
    
    model.train()
    print("Testing completed.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Echolancer Training")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--train_config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--val_data_dir", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--create_dummy", action="store_true", help="Create dummy data files")
    args = parser.parse_args()
    
    # Load configurations
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    # Create dummy data if requested
    if args.create_dummy:
        print("Creating dummy training dataset...")
        create_dummy_data_files(args.train_data_dir, num_samples=1000)
        print("Creating dummy validation dataset...")
        create_dummy_data_files(args.val_data_dir, num_samples=100)
    
    # Set random seed for reproducibility
    set_seed(train_config['training']['seed'])
    
    # Device configuration
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    train_config['wandb']['name'] = f"echolancer-{train_config['training']['seed']}"
    wandb_logger = WandbLogger(train_config)
    
    # Prepare model
    model = get_model(
        vocab_size=model_config['model']['vocab_size'],
        encoder_hidden=model_config['model']['encoder']['hidden'],
        encoder_head=model_config['model']['encoder']['head'],
        encoder_layer=model_config['model']['encoder']['layer'],
        decoder_hidden=model_config['model']['decoder']['hidden'],
        decoder_layer=model_config['model']['decoder']['layer'],
        decoder_head=model_config['model']['decoder']['head'],
        encoder_dropout=model_config['model']['encoder']['dropout'],
        decoder_dropout=model_config['model']['decoder']['dropout'],
        mel_channels=model_config['model']['mel_channels'],
        emotion_channels=model_config['model']['emotion_channels'],
        speaker_channels=model_config['model']['speaker_channels'],
        multi_speaker=model_config['model']['multi_speaker'],
        n_speaker=model_config['model']['n_speaker'] if model_config['model']['multi_speaker'] else 1,
        use_alibi=model_config['model'].get('use_alibi', False),
        alibi_alpha=model_config['model'].get('alibi_alpha', 1.0),
        activation=model_config['model'].get('activation', 'relu'),
        vq_token_mode=model_config['model'].get('vq_token_mode', False),
        vq_vocab_size=model_config['model'].get('vq_vocab_size', 1024),
        encoder_kv_heads=model_config['model'].get('encoder_kv_heads', None),
        decoder_kv_heads=model_config['model'].get('decoder_kv_heads', None),
        emotion_input_size=model_config['model'].get('emotion_input_size', 768),
        emotion_hidden_sizes=model_config['model'].get('emotion_hidden_sizes', [512, 384]),
        emotion_dropout=model_config['model'].get('emotion_dropout', 0.1)
    )
    
    model = model.to(device)
    
    # Print model parameters
    encoder_param = get_param_num(model.encoder)
    decoder_param = get_param_num(model.decoder)
    total_param = get_param_num(model)
    
    print("Number of Encoder Parameters: {:.2f}M".format(encoder_param / 1e6))
    print("Number of Decoder Parameters: {:.2f}M".format(decoder_param / 1e6))
    print("Total Number of Echolancer Parameters: {:.2f}M".format(total_param / 1e6))
    
    # Log model info to wandb
    wandb_logger.log_metrics({
        "model/encoder_params": encoder_param,
        "model/decoder_params": decoder_param,
        "model/total_params": total_param
    })
    
    # Prepare optimizer
    optimizer_config = train_config['optimizer']
    if optimizer_config['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay'],
            eps=optimizer_config['eps'],
            betas=optimizer_config['betas']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    
    # Initialize gradient scaler for mixed precision training (only needed with float16)
    if USE_AMP and AMP_DTYPE == torch.float16:
        scaler = torch.cuda.amp.GradScaler()
        print(f"Using mixed precision training with gradient scaling ({AMP_DTYPE})")
    elif USE_AMP and AMP_DTYPE == torch.bfloat16:
        scaler = None
        print(f"Using mixed precision training without gradient scaling ({AMP_DTYPE})")
    else:
        scaler = None
        print("Using full precision training")
    
    # Prepare scheduler
    total_steps = 10000  # This should be calculated based on your dataset
    scheduler = get_scheduler(optimizer, train_config['scheduler'], total_steps)
    
    loss_fn = EcholancerLoss()
    
    # Prepare early stopping
    early_stopping = EarlyStopping(
        patience=train_config['early_stopping']['patience'],
        min_delta=train_config['early_stopping']['min_delta']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if args.resume:
        model, optimizer, start_step = load_checkpoint(model, optimizer, args.resume, device, scaler)
        print(f"Resumed from checkpoint: {args.resume} at step {start_step}")
    
    # Load pipeline components for testing
    print("Loading pipeline components...")
    try:
        # Get testing configuration
        testing_config = train_config.get('testing', {})
        if testing_config.get('enabled', True):
            bert_model_name = testing_config.get('bert_model', 'answerdotai/ModernBERT-base')
            pre_encoder_dir = testing_config.get('pre_encoder_dir', './pre_encoder')
            istftnet_dir = testing_config.get('istftnet_dir', './istftnet')
            test_sentences = testing_config.get('test_sentences', [
                "This is a test sentence.",
                "Another example sentence."
            ])
            test_speakers = testing_config.get('test_speakers', [0, 0])
            
            bert_model = BERTFrontEnd(is_cuda=(device.type == 'cuda'), model_name=bert_model_name)
            pre_encoder = ScriptedPreEncoder(pre_encoder_dir, device)
            istftnet = ISTFTNetFE(None, None)
            istftnet.load_ts(istftnet_dir, device.type)
            print("Pipeline components loaded successfully.")
        else:
            print("Testing disabled in configuration.")
            bert_model = None
            pre_encoder = None
            istftnet = None
            test_sentences = []
            test_speakers = []
    except Exception as e:
        print(f"Warning: Failed to load pipeline components for testing: {e}")
        print("Testing will be skipped.")
        bert_model = None
        pre_encoder = None
        istftnet = None
        test_sentences = []
        test_speakers = []
    
    # Prepare dataset and dataloader
    print(f"Loading training dataset from {args.train_data_dir}...")
    train_loader = get_data_loader(
        args.train_data_dir, 
        batch_size=train_config['training']['batch_size']
    )
    
    print(f"Loading validation dataset from {args.val_data_dir}...")
    val_loader = get_data_loader(
        args.val_data_dir, 
        batch_size=train_config['training']['batch_size']
    )
    
    # Update total steps based on dataset size
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_config['training']['epochs']
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    
    # Reinitialize scheduler with correct total steps
    scheduler = get_scheduler(optimizer, train_config['scheduler'], total_steps)
    
    # Training configuration
    training_config = train_config['training']
    grad_acc_step = optimizer_config['grad_acc_step']
    grad_clip_thresh = optimizer_config['grad_clip_thresh']
    
    # Training loop
    best_val_loss = float('inf')
    step = start_step
    
    for epoch in range(start_epoch, training_config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']}")
        
        for batch_idx, batch in enumerate(epoch_bar):
            # Move batch to device
            speakers = batch[2].to(device)
            texts = batch[3].to(device)
            src_lens = batch[4].to(device)
            mels = batch[5].to(device)
            mel_lens = batch[6].to(device)
            em_hidden = batch[7].to(device)
            
            # Forward pass with autocast for mixed precision
            if USE_AMP:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    model_out = model(speakers, texts, src_lens, mels, mel_lens, em_hidden)
                    
                    # Compute loss
                    batch_for_loss = (batch[0], batch[1], speakers, texts, src_lens, mels, mel_lens, em_hidden)
                    losses = loss_fn(batch_for_loss, model_out)
                    total_loss = losses[0] / grad_acc_step
            else:
                model_out = model(speakers, texts, src_lens, mels, mel_lens, em_hidden)
                
                # Compute loss
                batch_for_loss = (batch[0], batch[1], speakers, texts, src_lens, mels, mel_lens, em_hidden)
                losses = loss_fn(batch_for_loss, model_out)
                total_loss = losses[0] / grad_acc_step
            
            # Backward pass
            if USE_AMP and scaler is not None:
                # Scale loss for gradient scaling (only needed with float16)
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_acc_step == 0:
                # Unscale gradients if using float16 AMP
                if USE_AMP and scaler is not None:
                    scaler.unscale_(optimizer)
                
                # Gradient clipping (always applied)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                
                # Optimizer step
                if USE_AMP and scaler is not None:
                    # Optimizer step with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Optimizer step (full precision or bfloat16)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Scheduler step
                if hasattr(scheduler, 'step'):
                    scheduler.step()
            
            # Update epoch loss
            epoch_loss += losses[0].item()
            epoch_steps += 1
            step += 1
            
            # Update progress bar
            epoch_bar.set_postfix({
                'loss': losses[0].item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Log training metrics
            if step % training_config['log_step'] == 0:
                train_metrics = {
                    "train/loss": losses[0].item(),
                    "train/token_loss": losses[4].item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "step": step
                }
                wandb_logger.log_metrics(train_metrics, step)
            
            # Save checkpoint
            if step % training_config['save_step'] == 0:
                checkpoint_path = os.path.join(args.output_dir, "ckpt", f"echolancer_checkpoint_{step}")
                save_checkpoint(model, optimizer, checkpoint_path, step, scaler)
                wandb_logger.log_model(f"{checkpoint_path}_{step}.pth", f"echolancer-checkpoint-{step}")
            
            # Validation
            if step % training_config['val_step'] == 0:
                val_loss = validate(model, val_loader, loss_fn, device, epoch)
                
                # Log validation metrics
                val_metrics = {
                    "val/loss": val_loss,
                    "step": step
                }
                wandb_logger.log_metrics(val_metrics, step)
                
                print(f"Validation Loss at step {step}: {val_loss:.4f}")
                
                # Early stopping
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
            
            # Testing on novel sentences
            if step % training_config.get('test_step', training_config['val_step'] * 2) == 0:
                if bert_model is not None and pre_encoder is not None and istftnet is not None:
                    # Load VQ tokens for zero-shot testing if in zero-shot mode
                    test_vq_tokens = testing_config.get('test_vq_tokens', None)
                    test_vq_token_lens = testing_config.get('test_vq_token_lens', None)
                    
                    # Convert paths to tensors if provided
                    if test_vq_tokens is not None and isinstance(test_vq_tokens, str):
                        try:
                            # Load VQ tokens from file
                            test_vq_tokens_data = torch.load(test_vq_tokens)
                            test_vq_tokens = test_vq_tokens_data.get('vq_tokens', None)
                            test_vq_token_lens = test_vq_tokens_data.get('vq_token_lens', None)
                        except Exception as e:
                            print(f"Warning: Failed to load VQ tokens for testing: {e}")
                            test_vq_tokens = None
                            test_vq_token_lens = None
                    
                    test_model(
                        model, bert_model, pre_encoder, istftnet,
                        test_sentences, test_speakers,
                        device, step, wandb_logger,
                        test_vq_tokens, test_vq_token_lens
                    )
        
        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Validate at the end of each epoch
        val_loss = validate(model, val_loader, loss_fn, device, epoch)
        print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
        
        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch+1,
            "epoch/train_loss": avg_epoch_loss,
            "epoch/val_loss": val_loss,
            "epoch/learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb_logger.log_metrics(epoch_metrics, step)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "ckpt", "echolancer_best_model")
            save_checkpoint(model, optimizer, best_model_path, epoch+1, scaler)
            wandb_logger.log_model(f"{best_model_path}_{epoch+1}.pth", "echolancer-best-model")
            
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best weights if early stopping was used
    early_stopping.load_best_weights(model)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "ckpt", "echolancer_final")
    save_checkpoint(model, optimizer, final_model_path, step, scaler)
    wandb_logger.log_model(f"{final_model_path}_{step}.pth", "echolancer-final-model")
    
    # Finish wandb run
    wandb_logger.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    main()