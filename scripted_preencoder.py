import torch
torch.backends.cudnn.benchmark = False

import os
import yaml
from typing import Optional, Union, List

def sequence_mask(max_length: int, x_lengths: torch.Tensor) -> torch.Tensor:
    """
    Creates a boolean mask from sequence lengths.
    The mask is True for padded positions and False for valid positions.

    Args:
        max_length (int): The maximum length of the sequences in the batch.
        x_lengths (torch.Tensor): A tensor of shape (batch_size,) containing the original length of each sequence.

    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, max_length).
    """
    mask = torch.arange(max_length, device=x_lengths.device).expand(len(x_lengths), max_length)
    mask = mask >= x_lengths.unsqueeze(1)
    return mask

class ScriptedPreEncoder:
    """
    A comprehensive wrapper for a TorchScript-exported PreEncoder model directory.

    This class provides a high-level interface to load a scripted PreEncoder model
    from a directory containing CPU and CUDA versions and a config file.
    It intelligently selects the best model based on the requested device.

    Attributes:
        model (torch.jit.ScriptModule): The loaded TorchScript model.
        device (torch.device): The device on which the model is running.
        config (dict): The model configuration loaded from model_config.yaml.
    """

    def __init__(self, model_dir: str, device: Optional[str] = 'cpu'):
        """
        Initializes the ScriptedPreEncoder by loading the model and config from a directory.

        Args:
            model_dir (str): The path to the directory containing the exported model files.
            device (Optional[str]): The desired device ('cpu', 'cuda'). Defaults to 'cpu'.

        Raises:
            FileNotFoundError: If the directory or required files are not found.
            RuntimeError: If the model or config fails to load.
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.device = torch.device(device)

        # Load configuration
        config_path = os.path.join(model_dir, 'model_config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"model_config.yaml not found in: {model_dir}")
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse config file: {e}")

        # Determine the correct model file to load
        model_path = self._get_model_path(model_dir)

        # Load the TorchScript model
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"Successfully loaded model from {os.path.basename(model_path)} onto {self.device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load TorchScript model from {model_path}: {e}")

    def _get_model_path(self, model_dir: str) -> str:
        """Determines the appropriate model file path based on the device."""
        cuda_path = os.path.join(model_dir, 'model_cuda.pt')
        cpu_path = os.path.join(model_dir, 'model_cpu.pt')

        if self.device.type == 'cuda':
            if os.path.exists(cuda_path):
                return cuda_path
            elif os.path.exists(cpu_path):
                print(f"Warning: CUDA device requested, but model_cuda.pt not found. Falling back to CPU model.")
                self.device = torch.device('cpu') # Update device to reflect reality
                return cpu_path
            else:
                raise FileNotFoundError("No CUDA or CPU model found in the specified directory.")
        else: # CPU
            if os.path.exists(cpu_path):
                return cpu_path
            else:
                raise FileNotFoundError(f"model_cpu.pt not found in: {model_dir}")

    @property
    def mel_channels(self) -> int:
        """Returns the number of mel channels from the model config."""
        return self.config.get('model', {}).get('mel_channels', 0)

    @property
    def fsq_levels(self) -> List[int]:
        """Returns the FSQ levels from the model config."""
        return self.config.get('model', {}).get('generator', {}).get('fsq_levels', [])

    def _prepare_mask(self, max_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """Creates and prepares the mask for the model."""
        mask = sequence_mask(max_len, lengths.to(self.device))
        # The model expects the mask to be of shape (B, 1, T)
        return mask.unsqueeze(1)

    def encode(self, spectrogram: torch.Tensor, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        """
        Encodes a spectrogram tensor into a sequence of discrete tokens.

        Args:
            spectrogram (torch.Tensor): A spectrogram tensor with the shape
                                      (batch_size, sequence_length, mel_channels).
            lengths (Optional[Union[List[int], torch.Tensor]]): A list or tensor of the original,
                un-padded lengths of each spectrogram in the batch. If provided, a mask
                will be created to handle padded data correctly.

        Returns:
            torch.Tensor: A tensor of discrete indices with the shape
                          (batch_size, sequence_length, num_quantizers).
        Raises:
            ValueError: If the input tensor does not have the expected 3 dimensions.
        """
        if spectrogram.ndim != 3:
            raise ValueError(f"Input spectrogram must be a 3D tensor (B, T, C), but got shape {spectrogram.shape}")

        spectrogram = spectrogram.to(self.device)
        mask = None
        if lengths is not None:
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)
            mask = self._prepare_mask(spectrogram.shape[1], lengths)

        with torch.no_grad():
            try:
                indices = self.model.encode(spectrogram, mask)
                return indices
            except Exception as e:
                raise RuntimeError(f"An error occurred during the encode operation: {e}")

    def decode(self, indices: torch.Tensor, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        """
        Decodes a tensor of discrete indices back into a spectrogram.

        Args:
            indices (torch.Tensor): A tensor of discrete indices with the shape
                                    (batch_size, sequence_length, num_quantizers).
            lengths (Optional[Union[List[int], torch.Tensor]]): A list or tensor of the original,
                un-padded lengths of each sequence in the batch. If provided, a mask
                will be created to handle padded data correctly.

        Returns:
            torch.Tensor: The reconstructed spectrogram tensor with the shape
                          (batch_size, sequence_length, mel_channels).
        Raises:
            ValueError: If the input tensor does not have the expected 3 dimensions.
        """
        indices = indices.to(self.device)
        mask = None
        if lengths is not None:
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)
            mask = self._prepare_mask(indices.shape[1], lengths)

        with torch.no_grad():
            try:
                reconstructed_spectrogram = self.model.decode(indices, mask)
                return reconstructed_spectrogram
            except Exception as e:
                raise RuntimeError(f"An error occurred during the decode operation: {e}")
