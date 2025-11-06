import torch
import yaml
from utils.char_tokenizer import CharTokenizer
from model.echolancer import Echolancer


class EcholancerFE:
    """
    A front end for the Echolancer model that simplifies inference operations.
    Provides a clean interface for loading checkpoints and generating audio tokens from text.
    """
    
    def __init__(self, model_config_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the Echolancer front end.

        Args:
            model_config_path (str, optional): Path to the model configuration YAML file.
            device (str): Device to run the model on. Defaults to CUDA if available, else CPU.
        """
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = CharTokenizer()
        self.model_config_path = model_config_path
        
        print(f"EcholancerFE initialized on device: {self.device}")

    def get_vocab_offset(self):
        return self.tokenizer.get_vocab_size()
    
    def load_checkpoint(self, checkpoint_path, model_config_path=None, **model_kwargs):
        """
        Loads a checkpoint into the model.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pt)
            model_config_path (str, optional): Path to model config YAML if not provided in constructor
            **model_kwargs: Additional arguments to override model parameters
        """
        # Load model configuration if provided
        if model_config_path is not None:
            self.model_config_path = model_config_path
            
        if self.model_config_path is None:
            raise ValueError("Model config path must be provided either in constructor or load_checkpoint")
        
        with open(self.model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        
        # Extract parameters from config
        model_params = model_config['model']
        decoder_params = model_params['decoder']
        
        # Apply any overrides from model_kwargs
        decoder_hidden = model_kwargs.get('decoder_hidden', decoder_params['hidden'])
        decoder_layer = model_kwargs.get('decoder_layer', decoder_params['layer'])
        decoder_head = model_kwargs.get('decoder_head', decoder_params['head'])
        decoder_dropout = model_kwargs.get('decoder_dropout', decoder_params['dropout'])
        speaker_channels = model_kwargs.get('speaker_channels', model_params['speaker_channels'])
        multi_speaker = model_kwargs.get('multi_speaker', model_params['multi_speaker'])
        n_speaker = model_kwargs.get('n_speaker', model_params['n_speaker'])
        alibi_alpha = model_kwargs.get('alibi_alpha', model_params['alibi_alpha'])
        use_alibi = model_kwargs.get('use_alibi', model_params['use_alibi'])
        activation = model_kwargs.get('activation', model_params.get('activation', 'relu'))
        vq_vocab_size = model_kwargs.get('vq_vocab_size', model_params['vq_vocab_size'])
        decoder_kv_heads = model_kwargs.get('decoder_kv_heads', model_params['decoder_kv_heads'])
        pretraining_mode = model_kwargs.get('pretraining_mode', model_params.get('pretraining_mode', True))
        zero_shot_mode = model_kwargs.get('zero_shot_mode', model_params.get('zero_shot_mode', False))
        
        # Create the model with parameters
        text_vocab_size = self.get_vocab_offset()

        self.model = Echolancer(
            vocab_size=text_vocab_size,
            decoder_hidden=decoder_hidden,
            decoder_layer=decoder_layer,
            decoder_head=decoder_head,
            decoder_dropout=decoder_dropout,
            emotion_channels=0,
            speaker_channels=speaker_channels,
            multi_speaker=multi_speaker,
            n_speaker=n_speaker,
            alibi_alpha=alibi_alpha,
            use_alibi=use_alibi,
            activation=activation,
            vq_token_mode=False,
            vq_vocab_size=vq_vocab_size,
            decoder_kv_heads=decoder_kv_heads,
            decoder_start_i=0,
            emotion_input_size=0,
            emotion_hidden_sizes=[512, 384],
            emotion_dropout=0.1,
            pretraining_mode=pretraining_mode,
            use_te=model_kwargs.get('use_te', False),
            lora_rank=model_kwargs.get('lora_rank', 0),
            lora_alpha=model_kwargs.get('lora_alpha', 16),
            lora_dropout=model_kwargs.get('lora_dropout', 0.0),
            lora_scale=model_kwargs.get('lora_scale', 1.0),
            zero_shot_mode=zero_shot_mode,
        )
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Handle potential key mismatches (like from torch.compile)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove _orig_mod. prefix if present (from torch.compile)
            if k.startswith("_orig_mod."):
                new_k = k[len("_orig_mod."):]
                cleaned_state_dict[new_k] = v
            # Remove module. prefix if present (from DataParallel)
            elif k.startswith("module."):
                new_k = k[len("module."):]
                cleaned_state_dict[new_k] = v
            else:
                cleaned_state_dict[k] = v
        
        # Load state dict
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Text vocab size: {text_vocab_size}, audio vocab size: {vq_vocab_size}")
        
        return self
    
    def infer(self, text, speaker_id=None, max_length=768, top_p=0.92, temperature=0.8, **kwargs):
        """
        Performs inference on the given text and returns audio tokens.

        Args:
            text (str): Input text to convert to audio
            speaker_id (int, optional): Speaker ID for multi-speaker models
            max_length (int): Maximum length of generated sequence
            top_p (float): Nucleus sampling parameter
            temperature (float): Temperature for sampling
            **kwargs: Additional arguments for generation

        Returns:
            torch.Tensor: Generated audio tokens of shape (1, T) where T is the sequence length
        """
        if self.model is None:
            raise ValueError("Model must be loaded with load_checkpoint() before calling infer()")
        
        # Prepare text tokens (similar to the train.py implementation)
        # 1) Encode text into token IDs
        inp_seq = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # 2) Append decoder BOS token for the audio segment
        inp_seq.append(int(self.model.decoder.bos_token_id))
        
        # 3) Convert to tensor and batch it (1, L)
        text_input = torch.tensor(inp_seq, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Prepare speaker tensor if needed
        if speaker_id is not None:
            if isinstance(speaker_id, torch.Tensor):
                # If speaker_id is already a tensor (e.g., speaker embedding in zero-shot mode)
                speaker_tensor = speaker_id.to(self.device)
            else:
                # If speaker_id is an integer ID
                speaker_tensor = torch.tensor([speaker_id], dtype=torch.long, device=self.device)
        else:
            speaker_tensor = None
        
        # Generate audio tokens
        with torch.no_grad():
            generated_tokens = self.model.infer(
                text_input, 
                speaker_tensor, 
                max_length=max_length, 
                top_p=top_p, 
                temperature=temperature,
                **kwargs
            )
        
        # Return the generated audio tokens (excluding the audio EOS token)
        # Remove the EOS token as the codec expects tokens without it
        generated_tokens = generated_tokens[:, :-1]
        
        return generated_tokens