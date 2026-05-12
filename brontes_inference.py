"""
Brontes inference module for audio refinement/enhancement.
"""
import torch
import torchaudio


class BrontesInference:
    """Inference class for loading and using exported TorchScript Brontes models.
    
    This class provides a simple interface for:
    - Loading CPU or GPU TorchScript models
    - Processing audio with automatic resampling if input sample rate differs
    - Loading/saving audio files
    
    Example:
        >>> inference = BrontesInference('brontes_cuda.pt', device='cuda')
        >>> output = inference.process_file('input.wav', 'output.wav')
        
        # Or with explicit sample rate:
        >>> audio, sr = torchaudio.load('input.wav')
        >>> output = inference.process(audio, input_sr=sr)
    """
    
    def __init__(self, model_path: str, device: str = None, sample_rate: int = 48000):
        """Initialize the inference engine.
        
        Args:
            model_path: Path to the TorchScript model file (.pt)
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            sample_rate: Model's expected sample rate (audio will be resampled to this)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.sample_rate = sample_rate
        
        # Cache for resamplers (to avoid recreating them)
        self._resamplers = {}
        
        # Load model
        print(f"Loading TorchScript model from: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print(f"Model loaded on: {self.device}")
        print(f"Model sample rate: {self.sample_rate} Hz")
    
    def _get_resampler(self, from_sr: int, to_sr: int) -> torchaudio.transforms.Resample:
        """Get or create a cached resampler."""
        key = (from_sr, to_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(from_sr, to_sr).to(self.device)
        return self._resamplers[key]
    
    def process(self, audio: torch.Tensor, input_sr: int = None, 
                normalize: bool = True) -> torch.Tensor:
        """Process audio tensor through the model.
        
        Args:
            audio: Input tensor of shape (batch, channels, samples), (channels, samples), or (samples,)
            input_sr: Sample rate of the input audio. If different from model's sample rate,
                      the audio will be resampled before processing.
                      If None, assumes audio is already at model's sample rate.
            normalize: Whether to normalize input to [-1, 1]
            
        Returns:
            Processed audio tensor at the model's sample rate.
        """
        # Ensure correct shape: (batch, channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device)
        
        # Normalize if requested
        if normalize:
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val
        
        # Resample to model's sample rate if needed
        if input_sr is not None and input_sr != self.sample_rate:
            resampler = self._get_resampler(input_sr, self.sample_rate)
            audio = resampler(audio)
        
        # Process entire tensor
        with torch.no_grad():
            output = self.model(audio)
        
        return output
    
    def load_audio(self, path: str, resample: bool = True, 
                   normalize: bool = True) -> tuple:
        """Load audio file.
        
        Args:
            path: Path to audio file
            resample: If True, resample to model's sample rate. If False, keep original.
            normalize: Whether to normalize to [-1, 1]
            
        Returns:
            Tuple of (audio tensor of shape (1, 1, samples), sample_rate)
        """
        audio, sr = torchaudio.load(path)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if requested
        if resample and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
            sr = self.sample_rate
        
        # Normalize
        if normalize:
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val
        
        return audio.unsqueeze(0), sr
    
    def save_audio(self, audio: torch.Tensor, path: str, sample_rate: int = None):
        """Save audio tensor to file.
        
        Args:
            audio: Audio tensor (any shape, will be properly formatted)
            path: Output file path
            sample_rate: Sample rate for output file (default: model's sample rate)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        
        audio = audio.cpu()
        max_val = audio.abs().max()
        if max_val > 1.0:
            audio = audio / max_val
        
        torchaudio.save(path, audio, sample_rate)
    
    def process_file(self, input_path: str, output_path: str = None,
                     preserve_sample_rate: bool = False) -> torch.Tensor:
        """Load, process, and optionally save an audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save output (optional)
            preserve_sample_rate: If True, resample output back to input sample rate.
                                  If False, output will be at model's sample rate.
            
        Returns:
            Processed audio tensor
        """
        # Load without resampling to get original sample rate
        audio_orig, input_sr = torchaudio.load(input_path)
        
        # Convert to mono
        if audio_orig.shape[0] > 1:
            audio_orig = audio_orig.mean(dim=0, keepdim=True)
        
        # Add batch dimension
        audio = audio_orig.unsqueeze(0)
        
        # Process with automatic resampling to model's sample rate
        output = self.process(audio, input_sr=input_sr)
        output_sr = self.sample_rate
        
        # Resample output back to input sample rate if requested
        if preserve_sample_rate and input_sr != self.sample_rate:
            resampler = self._get_resampler(self.sample_rate, input_sr)
            output = resampler(output)
            output_sr = input_sr
        
        if output_path:
            self.save_audio(output, output_path, output_sr)
            print(f"Saved to: {output_path}")
        
        return output
