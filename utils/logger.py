import wandb
import torch
import numpy as np
from typing import Dict, Any, List

class WandbLogger:
    """
    A logger class for Weights & Biases integration.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Weights & Biases logger.
        
        Args:
            config: Configuration dictionary containing WANDB settings
        """
        self.config = config
        self.is_initialized = False
        
        # Initialize Weights & Biases
        if 'wandb' in config:
            wandb_config = config['wandb']
            wandb.init(
                project=wandb_config.get('project', 'echolancer'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('name', 'echolancer-run'),
                notes=wandb_config.get('notes', ''),
                tags=wandb_config.get('tags', []),
                group=wandb_config.get('group', ''),
                config=config
            )
            self.is_initialized = True
            
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """
        Log metrics to Weights & Biases.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step
        """
        if self.is_initialized:
            wandb.log(metrics, step=step)
            
    def log_audio(self, audio: np.ndarray, sample_rate: int, name: str, step: int = None):
        """
        Log audio to Weights & Biases.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            name: Name of the audio sample
            step: Training step
        """
        if self.is_initialized:
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Normalize if needed
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
                
            wandb.log({
                name: wandb.Audio(audio, sample_rate=sample_rate)
            }, step=step)
            
    def log_model(self, model_path: str, name: str = "model"):
        """
        Log model artifact to Weights & Biases.
        
        Args:
            model_path: Path to the model file
            name: Name of the model artifact
        """
        if self.is_initialized:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
    def log_histogram(self, values: torch.Tensor, name: str, step: int = None):
        """
        Log histogram of values to Weights & Biases.
        
        Args:
            values: Tensor of values to log as histogram
            name: Name of the histogram
            step: Training step
        """
        if self.is_initialized:
            wandb.log({
                name: wandb.Histogram(values.cpu().numpy())
            }, step=step)
            
    def finish(self):
        """
        Finish the Weights & Biases run.
        """
        if self.is_initialized:
            wandb.finish()