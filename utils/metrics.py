import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute accuracy for classification/prediction tasks.
    
    Args:
        predictions: Predicted values (logits or probabilities)
        targets: Ground truth labels
        mask: Optional mask for valid positions
        
    Returns:
        Accuracy as a float value
    """
    # Convert logits to predictions if needed
    if predictions.dim() > 1 and predictions.size(-1) > 1:
        preds = torch.argmax(predictions, dim=-1)
    else:
        preds = (predictions > 0.5).long()
    
    # Compute correct predictions
    correct = (preds == targets).float()
    
    # Apply mask if provided
    if mask is not None:
        correct = correct * mask
        total = mask.sum()
    else:
        total = targets.numel()
    
    accuracy = correct.sum() / total
    return accuracy.item()

def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return np.exp(loss)

def compute_mae(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid positions
        
    Returns:
        MAE as a float value
    """
    abs_error = torch.abs(predictions - targets)
    
    # Apply mask if provided
    if mask is not None:
        abs_error = abs_error * mask
        total = mask.sum()
    else:
        total = targets.numel()
    
    mae = abs_error.sum() / total
    return mae.item()

def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Root Mean Square Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid positions
        
    Returns:
        RMSE as a float value
    """
    squared_error = (predictions - targets) ** 2
    
    # Apply mask if provided
    if mask is not None:
        squared_error = squared_error * mask
        total = mask.sum()
    else:
        total = targets.numel()
    
    rmse = torch.sqrt(squared_error.sum() / total)
    return rmse.item()

def compute_spectral_convergence(predictions: torch.Tensor, targets: torch.Tensor, 
                                mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Spectral Convergence for audio spectrograms.
    
    Args:
        predictions: Predicted spectrogram
        targets: Ground truth spectrogram
        mask: Optional mask for valid positions
        
    Returns:
        Spectral convergence as a float value
    """
    # Apply mask if provided
    if mask is not None:
        predictions = predictions * mask
        targets = targets * mask
    
    # Compute spectral convergence
    numerator = torch.norm(targets - predictions, p='fro')
    denominator = torch.norm(targets, p='fro')
    
    if denominator == 0:
        return 0.0
    
    sc = numerator / denominator
    return sc.item()

def compute_length_accuracy(predicted_lengths: torch.Tensor, 
                           target_lengths: torch.Tensor) -> float:
    """
    Compute accuracy of predicted sequence lengths.
    
    Args:
        predicted_lengths: Predicted sequence lengths
        target_lengths: Ground truth sequence lengths
        
    Returns:
        Length accuracy as a float value
    """
    correct = (predicted_lengths == target_lengths).float()
    accuracy = correct.mean()
    return accuracy.item()

def compute_bleu_score(predictions: List[List[str]], 
                      references: List[List[List[str]]], 
                      max_n: int = 4) -> float:
    """
    Compute BLEU score for text generation (simplified implementation).
    
    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences (multiple references per prediction)
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score as a float value
    """
    # This is a simplified implementation - in practice, you'd use nltk or sacrebleu
    # For now, we'll return a placeholder
    return 0.0

def compute_word_error_rate(predictions: List[str], 
                           references: List[str]) -> float:
    """
    Compute Word Error Rate (WER) for text generation.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        WER as a float value
    """
    # This is a simplified implementation - in practice, you'd use jiwer or similar
    # For now, we'll return a placeholder
    return 0.0

def compute_metrics(predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor],
                    masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
    """
    Compute comprehensive metrics for the Echolancer model.
    
    Args:
        predictions: Dictionary of model predictions
        targets: Dictionary of ground truth targets
        masks: Optional dictionary of masks for valid positions
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Compute token prediction accuracy
    if 'tokens' in predictions and 'tokens' in targets:
        token_mask = masks.get('tokens') if masks else None
        metrics['token_accuracy'] = compute_accuracy(
            predictions['tokens'], targets['tokens'], token_mask
        )
    
    # Compute mel spectrogram MAE
    if 'mels' in predictions and 'mels' in targets:
        mel_mask = masks.get('mels') if masks else None
        metrics['mel_mae'] = compute_mae(
            predictions['mels'], targets['mels'], mel_mask
        )
        metrics['mel_rmse'] = compute_rmse(
            predictions['mels'], targets['mels'], mel_mask
        )
        metrics['spectral_convergence'] = compute_spectral_convergence(
            predictions['mels'], targets['mels'], mel_mask
        )
    
    # Compute gate prediction accuracy
    # Gate outputs have been removed from the model
    
    return metrics

# Example usage:
if __name__ == "__main__":
    # Example: Compute accuracy
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = torch.tensor([1, 0, 1])
    mask = torch.tensor([1.0, 1.0, 0.0])  # Mask last position
    
    acc = compute_accuracy(preds, targets, mask)
    print(f"Accuracy: {acc:.4f}")
    
    # Example: Compute MAE
    pred_vals = torch.randn(10, 20, 80)
    target_vals = torch.randn(10, 20, 80)
    mae = compute_mae(pred_vals, target_vals)
    print(f"MAE: {mae:.4f}")