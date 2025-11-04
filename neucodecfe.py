import torch
from neucodec import DistillNeuCodec

class NeuCodecFE():
    """
    A front end using a neural audio codec (NeuCodec) for audio encoding/decoding.
    This replaces the traditional spectrogram VQVAE + iSTFTNet approach with a direct neural audio codec.
    """

    def __init__(self, is_cuda=False, model_name="neuphonic/distill-neucodec", offset=0):
        """
        Initializes the neural audio codec front end.

        Args:
          is_cuda (bool): Whether to use CUDA for inference. Defaults to False.
          model_name (str): The NeuCodec model to load. Defaults to 'neuphonic/distill-neucodec'.
          offset: (int): The discrete code offset to compensate for
        """
        self.model = DistillNeuCodec.from_pretrained(model_name)
        self.is_cuda = is_cuda
        self.offset = offset
        self.device = torch.device("cuda" if is_cuda else "cpu")


        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Loaded NeuCodec model: {model_name} (CUDA: {is_cuda})")

    def encode_audio(self, audio_tensor):
        """
        Encodes audio tensor to discrete codes using NeuCodec.

        Args:
          audio_tensor (torch.Tensor): Audio tensor of shape (B, 1, T) at 16kHz sample rate

        Returns:
          torch.Tensor: Discrete codes tensor of shape (B, 1, T_code_len)
        """
        with torch.no_grad():
            fsq_codes = self.model.encode_code(audio_tensor)
        return fsq_codes

    def decode_codes(self, codes):
        """
        Decodes discrete codes back to audio using NeuCodec.

        Args:
          codes (torch.Tensor): Discrete codes tensor of shape (B, 1, T_code_len)

        Returns:
          torch.Tensor: Reconstructed audio tensor of shape (B, 1, T_audio)
        """
        with torch.no_grad():
            codes = codes.to(self.device)
            codes = codes - self.offset
            reconstructed_audio = self.model.decode_code(codes)
        return reconstructed_audio

    def infer(self, audio_tensor):
        """
        Convenience method to encode and decode audio (reconstruct).

        Args:
          audio_tensor (torch.Tensor): Audio tensor of shape (B, 1, T) at 16kHz sample rate

        Returns:
          torch.Tensor: Reconstructed audio tensor of shape (B, 1, T_audio)
        """
        codes = self.encode_audio(audio_tensor)
        reconstructed = self.decode_codes(codes)
        return reconstructed