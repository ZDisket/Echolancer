import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropy(nn.Module):
    """
    A masked cross-entropy loss for discrete token prediction.
    """

    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: Tensor of shape (B, T, vocab_size) containing logits.
            targets: Tensor of shape (B, T) containing ground truth token indices.
            mask: Optional Float Tensor of shape (B, T) with 1.0 for valid positions and 0.0 for padded positions.

        Returns:
            Scalar cross-entropy loss normalized by the number of valid (non-padded) elements.
        """
        # Reshape for cross-entropy
        B, T, V = logits.size()
        logits_flat = logits.reshape(-1, V)  # (B*T, vocab_size)
        targets_flat = targets.reshape(-1)    # (B*T,)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)  # (B*T,)
            # Apply mask to targets and logits
            logits_flat = logits_flat[mask_flat.bool()]
            targets_flat = targets_flat[mask_flat.bool()]
        
        # Compute cross-entropy loss
        if logits_flat.size(0) == 0:  # No valid elements
            return torch.tensor(0.0, device=logits.device)
            
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        return loss


class EcholancerLoss(nn.Module):
    """
    Loss function for the TTS model (Echolancer) - Discrete Token Prediction
    """

    def __init__(self):
        super(EcholancerLoss, self).__init__()
        self.token_loss_fn = MaskedCrossEntropy()

    def forward(self, batch, model_out, epoch=0):
        """
        Compute loss for Echolancer model with discrete token prediction.
        
        Args:
            batch: Tuple containing (ids, raw_texts, speakers, texts, src_lens, mels, mel_lens, em_hidden)
            model_out: Tuple containing (text_mask, mel_mask, attn_logprob, x_mask_in, logits, indices_gt)
            epoch: Current epoch number (for scheduling)
            
        Returns:
            List of losses: [total_loss, dummy_attn_loss, token_loss]
        """
        # Extract model outputs
        text_mask, mel_mask, attn_logprob, x_mask_in, logits, indices_gt = model_out

        # Extract targets
        token_targets = indices_gt[:, 1:]  # Shifted targets for autoregressive prediction
        
        # Create sequence mask for token loss (True for valid positions)
        seq_mask = ~x_mask_in  # Invert: True for valid positions, False for padded

        # Compute token loss
        token_loss = self.token_loss_fn(logits, token_targets, seq_mask)
        
        # For compatibility with existing code that expects a list of losses
        dummy_attn_loss = torch.tensor(0.0, device=token_loss.device)
        total_loss = token_loss  # Only token loss now
        
        return [total_loss, dummy_attn_loss, token_loss]