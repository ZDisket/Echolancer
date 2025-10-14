import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

class DataAugmenter:
    """
    Data augmentation for Echolancer training data.
    """
    
    def __init__(self, 
                 time_mask_prob: float = 0.1,
                 time_mask_width: int = 10,
                 freq_mask_prob: float = 0.1,
                 freq_mask_width: int = 5,
                 text_dropout_prob: float = 0.05,
                 speaker_perturbation: bool = False):
        """
        Initialize data augmenter.
        
        Args:
            time_mask_prob: Probability of applying time masking
            time_mask_width: Width of time masks
            freq_mask_prob: Probability of applying frequency masking
            freq_mask_width: Width of frequency masks
            text_dropout_prob: Probability of dropping out text tokens
            speaker_perturbation: Whether to perturb speaker embeddings
        """
        self.time_mask_prob = time_mask_prob
        self.time_mask_width = time_mask_width
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_width = freq_mask_width
        self.text_dropout_prob = text_dropout_prob
        self.speaker_perturbation = speaker_perturbation
    
    def augment_spectrogram(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time and frequency masking to mel spectrogram.
        
        Args:
            mel_spectrogram: Mel spectrogram tensor of shape (time_steps, freq_bins)
            
        Returns:
            Augmented mel spectrogram
        """
        augmented = mel_spectrogram.clone()
        
        # Time masking
        if random.random() < self.time_mask_prob:
            time_steps = augmented.size(0)
            mask_start = random.randint(0, max(0, time_steps - self.time_mask_width))
            mask_end = min(time_steps, mask_start + self.time_mask_width)
            augmented[mask_start:mask_end, :] = 0  # Zero out time segment
        
        # Frequency masking
        if random.random() < self.freq_mask_prob:
            freq_bins = augmented.size(1)
            mask_start = random.randint(0, max(0, freq_bins - self.freq_mask_width))
            mask_end = min(freq_bins, mask_start + self.freq_mask_width)
            augmented[:, mask_start:mask_end] = 0  # Zero out frequency band
        
        return augmented
    
    def augment_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to text tokens.
        
        Args:
            text_tokens: Text token tensor of shape (seq_len,)
            
        Returns:
            Augmented text tokens
        """
        if self.text_dropout_prob <= 0:
            return text_tokens
        
        # Create dropout mask
        mask = torch.rand_like(text_tokens.float()) > self.text_dropout_prob
        augmented = text_tokens * mask.long()
        
        return augmented
    
    def augment_speaker(self, speaker_embedding: torch.Tensor, 
                       noise_level: float = 0.01) -> torch.Tensor:
        """
        Apply noise to speaker embedding.
        
        Args:
            speaker_embedding: Speaker embedding tensor
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Perturbed speaker embedding
        """
        if not self.speaker_perturbation:
            return speaker_embedding
        
        # Add Gaussian noise
        noise = torch.randn_like(speaker_embedding) * noise_level
        perturbed = speaker_embedding + noise
        
        return perturbed
    
    def augment_emotion(self, emotion_embedding: torch.Tensor,
                       noise_level: float = 0.01) -> torch.Tensor:
        """
        Apply noise to emotion embedding.
        
        Args:
            emotion_embedding: Emotion embedding tensor
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Perturbed emotion embedding
        """
        # Add Gaussian noise
        noise = torch.randn_like(emotion_embedding) * noise_level
        perturbed = emotion_embedding + noise
        
        return perturbed
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to a batch.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Augmented batch data
        """
        augmented_batch = {}
        
        # Copy unchanged items
        for key, value in batch.items():
            augmented_batch[key] = value
        
        # Augment mel spectrograms
        if 'mel' in batch:
            augmented_batch['mel'] = self.augment_spectrogram(batch['mel'])
        
        # Augment text tokens
        if 'text' in batch:
            augmented_batch['text'] = self.augment_text(batch['text'])
        
        # Augment speaker embeddings if present
        if 'speaker' in batch and self.speaker_perturbation:
            augmented_batch['speaker'] = self.augment_speaker(batch['speaker'])
        
        # Augment emotion embeddings
        if 'emotion' in batch:
            augmented_batch['emotion'] = self.augment_emotion(batch['emotion'])
        
        return augmented_batch

def add_noise_to_spectrogram(mel_spectrogram: torch.Tensor, 
                            noise_level: float = 0.01) -> torch.Tensor:
    """
    Add Gaussian noise to mel spectrogram.
    
    Args:
        mel_spectrogram: Mel spectrogram tensor
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        Noisy mel spectrogram
    """
    noise = torch.randn_like(mel_spectrogram) * noise_level
    noisy_spectrogram = mel_spectrogram + noise
    return noisy_spectrogram

def time_stretch_spectrogram(mel_spectrogram: torch.Tensor,
                            stretch_factor: float = 1.0) -> torch.Tensor:
    """
    Apply time stretching to mel spectrogram (simplified implementation).
    
    Args:
        mel_spectrogram: Mel spectrogram tensor of shape (time_steps, freq_bins)
        stretch_factor: Stretch factor (e.g., 1.1 for 10% slower)
        
    Returns:
        Time-stretched mel spectrogram
    """
    # This is a simplified implementation
    # In practice, you'd use librosa or similar for proper time stretching
    if stretch_factor == 1.0:
        return mel_spectrogram
    
    # Simple linear interpolation for demonstration
    time_steps, freq_bins = mel_spectrogram.shape
    new_time_steps = int(time_steps / stretch_factor)
    
    # Create new time indices
    old_indices = torch.linspace(0, time_steps - 1, time_steps)
    new_indices = torch.linspace(0, time_steps - 1, new_time_steps)
    
    # Interpolate
    augmented = torch.zeros(new_time_steps, freq_bins, device=mel_spectrogram.device)
    for i in range(freq_bins):
        augmented[:, i] = torch.interp(new_indices, old_indices, mel_spectrogram[:, i])
    
    return augmented

def pitch_shift_spectrogram(mel_spectrogram: torch.Tensor,
                           shift_bins: int = 0) -> torch.Tensor:
    """
    Apply pitch shifting to mel spectrogram.
    
    Args:
        mel_spectrogram: Mel spectrogram tensor of shape (time_steps, freq_bins)
        shift_bins: Number of frequency bins to shift (positive = up, negative = down)
        
    Returns:
        Pitch-shifted mel spectrogram
    """
    if shift_bins == 0:
        return mel_spectrogram
    
    time_steps, freq_bins = mel_spectrogram.shape
    
    # Create shifted spectrogram
    shifted = torch.zeros_like(mel_spectrogram)
    
    if shift_bins > 0:
        # Shift up (drop high frequencies)
        shifted[:, shift_bins:] = mel_spectrogram[:, :-shift_bins]
    else:
        # Shift down (drop low frequencies)
        shifted[:, :shift_bins] = mel_spectrogram[:, -shift_bins:]
    
    return shifted

# Example usage:
if __name__ == "__main__":
    # Example: Create sample data
    mel_spec = torch.randn(100, 80)  # (time_steps=100, freq_bins=80)
    text_tokens = torch.randint(0, 100, (20,))  # (seq_len=20)
    emotion_emb = torch.randn(768)  # BERT-sized emotion embedding
    
    # Create augmenter
    augmenter = DataAugmenter(
        time_mask_prob=0.2,
        freq_mask_prob=0.2,
        text_dropout_prob=0.05
    )
    
    # Create sample batch
    batch = {
        'mel': mel_spec,
        'text': text_tokens,
        'emotion': emotion_emb
    }
    
    # Apply augmentation
    augmented_batch = augmenter(batch)
    
    print("Original mel shape:", batch['mel'].shape)
    print("Augmented mel shape:", augmented_batch['mel'].shape)
    print("Original text shape:", batch['text'].shape)
    print("Augmented text shape:", augmented_batch['text'].shape)