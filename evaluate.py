import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Import our standalone modules
from model import Echolancer
from utils import get_model, load_checkpoint, get_device, compute_metrics
from data import get_data_loader
from utils.metrics import compute_accuracy, compute_mae, compute_rmse

def evaluate_model(model, data_loader, device, criterion=None):
    \"\"\"
    Evaluate the Echolancer model on a dataset.
    
    Args:
        model: Trained Echolancer model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        criterion: Loss function (optional)
        
    Returns:
        Dictionary of evaluation metrics
    \"\"\"
    model.eval()
    
    # Initialize metrics accumulators
    total_loss = 0.0
    total_token_accuracy = 0.0
    total_mel_mae = 0.0
    total_mel_rmse = 0.0
    total_samples = 0
    
    # Progress bar
    eval_bar = tqdm(data_loader, desc=\"Evaluating\")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_bar):
            # Move batch to device
            speakers = batch[2].to(device)
            texts = batch[3].to(device)
            src_lens = batch[4].to(device)
            mels = batch[5].to(device)
            mel_lens = batch[6].to(device)
            em_hidden = batch[7].to(device)
            
            # Forward pass
            model_out = model(speakers, texts, src_lens, mels, mel_lens, em_hidden)
            
            # Extract outputs
            mel_pred, text_mask, mel_mask, _, x_mask_in, logits, indices_gt = model_out
            
            # Compute loss if criterion is provided
            if criterion is not None:
                batch_for_loss = (batch[0], batch[1], speakers, texts, src_lens, mels, mel_lens, em_hidden)
                losses = criterion(batch_for_loss, model_out)
                batch_loss = losses[0].item()
                total_loss += batch_loss
            
            # Compute metrics
            # Token accuracy (using logits vs indices)
            if logits is not None and indices_gt is not None:
                # Mask padded positions
                valid_positions = ~x_mask_in  # True for valid positions
                token_acc = compute_accuracy(logits, indices_gt[:, 1:], valid_positions)
                total_token_accuracy += token_acc
            
            # Mel spectrogram MAE and RMSE
            if mel_pred is not None and mels is not None:
                mel_mae = compute_mae(mel_pred, mels, mel_mask)
                mel_rmse = compute_rmse(mel_pred, mels, mel_mask)
                total_mel_mae += mel_mae
                total_mel_rmse += mel_rmse
            
            total_samples += 1
            
            # Update progress bar
            eval_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1) if criterion is not None else 0,
                'token_acc': total_token_accuracy / (batch_idx + 1),
                'mel_mae': total_mel_mae / (batch_idx + 1)
            })
    
    # Compute final metrics
    metrics = {
        'samples_evaluated': total_samples,
        'average_loss': total_loss / total_samples if criterion is not None else 0,
        'token_accuracy': total_token_accuracy / total_samples,
        'mel_mae': total_mel_mae / total_samples,
        'mel_rmse': total_mel_rmse / total_samples
    }
    
    return metrics

def run_evaluation(args):
    \"\"\"
    Run model evaluation.
    
    Args:
        args: Command line arguments
    \"\"\"
    print(\"Starting Echolancer Evaluation\")
    print(\"=\" * 40)
    
    # Device configuration
    device = get_device()
    print(f\"Using device: {device}\")
    
    # Load model configuration
    try:
        import yaml
        with open(args.model_config, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config['model']
        print(f\"Loaded model configuration from {args.model_config}\")
    except Exception as e:
        print(f\"Error loading model configuration: {e}\")
        return
    
    # Create model
    print("Creating model...")
    model = get_model(
        vocab_size=model_config['vocab_size'],
        encoder_hidden=model_config['encoder']['hidden'],
        encoder_head=model_config['encoder']['head'],
        encoder_layer=model_config['encoder']['layer'],
        decoder_hidden=model_config['decoder']['hidden'],
        decoder_layer=model_config['decoder']['layer'],
        decoder_head=model_config['decoder']['head'],
        encoder_dropout=model_config['encoder']['dropout'],
        decoder_dropout=model_config['decoder']['dropout'],
        mel_channels=model_config['mel_channels'],
        emotion_channels=model_config['emotion_channels'],
        speaker_channels=model_config['speaker_channels'],
        multi_speaker=model_config['multi_speaker'],
        n_speaker=model_config['n_speaker'] if model_config['multi_speaker'] else 1,
        use_alibi=model_config.get('use_alibi', False),
        alibi_alpha=model_config.get('alibi_alpha', 1.0),
        activation=model_config.get('activation', 'relu'),
        vq_token_mode=model_config.get('vq_token_mode', False),
        vq_vocab_size=model_config.get('vq_vocab_size', 1024),
        encoder_kv_heads=model_config.get('encoder_kv_heads', None),
        decoder_kv_heads=model_config.get('decoder_kv_heads', None)
    )
    
    model = model.to(device)
    print(f\"Model created successfully\")
    
    # Load checkpoint
    print(f\"Loading checkpoint from {args.checkpoint}\")
    model, _ = load_checkpoint(model, None, args.checkpoint, device)
    print(\"Checkpoint loaded successfully\")
    
    # Prepare data loader
    print(f\"Loading evaluation data from {args.eval_data_dir}\")
    eval_loader = get_data_loader(
        args.eval_data_dir,
        batch_size=args.batch_size,
        shuffle=False  # Don't shuffle for evaluation
    )
    print(f\"Loaded {len(eval_loader.dataset)} evaluation samples\")
    
    # Create criterion if specified
    criterion = None
    if args.compute_loss:
        from model.loss import EcholancerLoss
        # Load training config for loss parameters
        try:
            with open(args.train_config, 'r') as f:
                train_config = yaml.safe_load(f)
            loss_config = train_config['loss']
            
            criterion = EcholancerLoss()
            criterion = criterion.to(device)
            print(\"Loss function created for evaluation\")
        except Exception as e:
            print(f\"Warning: Could not create loss function: {e}\")
            criterion = None
    
    # Run evaluation
    print(\"\\nRunning evaluation...\")
    metrics = evaluate_model(model, eval_loader, device, criterion)
    
    # Print results
    print(\"\\n\" + \"=\" * 50)
    print(\"EVALUATION RESULTS\")
    print(\"=\" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f\"{key.replace('_', ' ').title()}: {value:.6f}\")
        else:
            print(f\"{key.replace('_', ' ').title()}: {value}\")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f\"\\nResults saved to {args.output_file}\")
    
    print(\"\\nEvaluation completed!\")

def main():
    parser = argparse.ArgumentParser(description=\"Evaluate Echolancer Model\")
    parser.add_argument(\"--model_config\", type=str, required=True, 
                        help=\"Path to model configuration file\")
    parser.add_argument(\"--train_config\", type=str, 
                        help=\"Path to training configuration file (for loss)\")
    parser.add_argument(\"--checkpoint\", type=str, required=True,
                        help=\"Path to model checkpoint\")
    parser.add_argument(\"--eval_data_dir\", type=str, required=True,
                        help=\"Path to evaluation data directory\")
    parser.add_argument(\"--batch_size\", type=int, default=16,
                        help=\"Batch size for evaluation\")
    parser.add_argument(\"--compute_loss\", action=\"store_true\",
                        help=\"Compute loss during evaluation\")
    parser.add_argument(\"--output_file\", type=str, default=\"evaluation_results.json\",
                        help=\"Path to save evaluation results\")
    
    args = parser.parse_args()
    run_evaluation(args)

if __name__ == \"__main__\":
    main()