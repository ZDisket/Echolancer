import argparse
import os
import torch
import numpy as np
import json

# Import our standalone modules
from model import Echolancer
from utils import get_model, get_param_num, load_checkpoint, get_device, load_config

def synthesize_text(model, text_tokens, speaker_id=0, emotion_embedding=None, max_length=1000):
    """
    Synthesize speech from text tokens using the Echolancer model.
    
    Args:
        model: Trained Echolancer model
        text_tokens: List or array of token IDs representing the input text
        speaker_id: ID of the speaker (for multi-speaker models)
        emotion_embedding: Emotion embedding vector (if using emotion conditioning)
        max_length: Maximum length of generated mel-spectrogram
    
    Returns:
        Generated token sequence
    """
    with torch.no_grad():
        # Convert inputs to tensors
        device = next(model.parameters()).device
        
        # Text tokens
        texts = torch.tensor([text_tokens], dtype=torch.long, device=device)
        src_lens = torch.tensor([len(text_tokens)], dtype=torch.long, device=device)
        
        # Speaker ID
        speakers = torch.tensor([speaker_id], dtype=torch.long, device=device)
        
        # Emotion embedding (if provided)
        if emotion_embedding is not None:
            em_hidden = torch.tensor([emotion_embedding], dtype=torch.float32, device=device)
        else:
            em_hidden = None
        
        # Run inference
        model.eval()  # Ensure model is in evaluation mode
        token_outputs = model.infer(
            speakers, texts, src_lens, 
            em_hidden=em_hidden, 
            max_length=max_length
        )
        
        return token_outputs[0].cpu().numpy()

def tokenize_text_simple(text, vocab_size=100):
    """
    Simple tokenizer for demonstration purposes.
    In practice, you would use a proper tokenizer.
    
    Args:
        text: Input text string
        vocab_size: Size of vocabulary
        
    Returns:
        List of token IDs
    """
    # Simple character-level tokenization for demonstration
    tokens = []
    for char in text.lower():
        if char.isalnum():
            # Hash character to token ID
            token_id = hash(char) % (vocab_size - 10)  # Reserve last 10 for special tokens
            tokens.append(token_id)
        elif char == ' ':
            tokens.append(vocab_size - 1)  # Special token for space
    
    return tokens if tokens else [0]  # Return at least one token

def load_emotion_embedding(text, emotion="neutral"):
    """
    Load or generate emotion embedding.
    In practice, you would use a BERT-type model.
    
    Args:
        text: Input text
        emotion: Target emotion (for demonstration)
        
    Returns:
        Emotion embedding vector
    """
    # For demonstration, return random embedding
    # In practice, you would use a BERT-type model to extract emotion
    emotion_embedding = np.random.randn(768).astype(np.float32)
    return emotion_embedding

def main():
    parser = argparse.ArgumentParser(description="Echolancer Inference")
    parser.add_argument("--model_config", type=str, help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default="This is a test sentence.", help="Input text to synthesize")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID")
    parser.add_argument("--emotion", type=str, default="neutral", help="Target emotion")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum output length")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configuration
    if args.model_config:
        model_config = load_config(args.model_config)
        config = model_config['model']
    else:
        # Default configuration (should match training configuration)
        config = {
            'vocab_size': 100,
            'mel_channels': 80,
            'emotion_channels': 256,
            'speaker_channels': 32,
            'multi_speaker': False,
            'n_speaker': 1,
            'encoder': {
                'hidden': 256,
                'head': 4,
                'layer': 4,
                'dropout': 0.1,
                'forward_expansion': 4
            },
            'decoder': {
                'hidden': 256,
                'head': 4,
                'layer': 4,
                'dropout': 0.1
            }
        }
    
    print("Loading model...")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with proper configuration
    if 'encoder' in config:
        # New config format
        model = get_model(
            vocab_size=config['vocab_size'],
            encoder_hidden=config['encoder']['hidden'],
            encoder_head=config['encoder']['head'],
            encoder_layer=config['encoder']['layer'],
            decoder_hidden=config['decoder']['hidden'],
            decoder_layer=config['decoder']['layer'],
            decoder_head=config['decoder']['head'],
            encoder_dropout=config['encoder'].get('dropout', 0.1),
            decoder_dropout=config['decoder'].get('dropout', 0.1),
            mel_channels=config['mel_channels'],
            emotion_channels=config['emotion_channels'],
            speaker_channels=config['speaker_channels'],
            multi_speaker=config.get('multi_speaker', False),
            n_speaker=config.get('n_speaker', 1) if config.get('multi_speaker', False) else 1,
            use_alibi=config.get('use_alibi', False),
            alibi_alpha=config.get('alibi_alpha', 1.0),
            activation=config.get('activation', 'relu'),
            vq_token_mode=config.get('vq_token_mode', False),
            vq_vocab_size=config.get('vq_vocab_size', 1024),
            encoder_kv_heads=config.get('encoder_kv_heads', None),
            decoder_kv_heads=config.get('decoder_kv_heads', None)
        )
    else:
        # Legacy config format
        model = get_model(
            vocab_size=config['vocab_size'],
            encoder_hidden=config['encoder_hidden'],
            encoder_head=config['encoder_head'],
            encoder_layer=config['encoder_layer'],
            decoder_hidden=config['decoder_hidden'],
            decoder_layer=config['decoder_layer'],
            decoder_head=config['decoder_head'],
            encoder_dropout=config['encoder_dropout'],
            decoder_dropout=config['decoder_dropout'],
            mel_channels=config['mel_channels'],
            emotion_channels=config['emotion_channels'],
            speaker_channels=config['speaker_channels'],
            multi_speaker=config['multi_speaker'],
            n_speaker=config['n_speaker'] if config['multi_speaker'] else 1,
            vq_token_mode=config.get('vq_token_mode', False),
            vq_vocab_size=config.get('vq_vocab_size', 1024),
            encoder_kv_heads=config.get('encoder_kv_heads', None),
            decoder_kv_heads=config.get('decoder_kv_heads', None)
        )
    
    # Load checkpoint
    model, _ = load_checkpoint(model, None, args.checkpoint, device)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {get_param_num(model) / 1e6:.2f}M")
    
    # Tokenize input text
    text_tokens = tokenize_text_simple(args.text, config['vocab_size'])
    print(f"Tokenized text: {args.text}")
    print(f"Token IDs: {text_tokens[:20]}{'...' if len(text_tokens) > 20 else ''}")
    
    # Generate emotion embedding
    emotion_embedding = load_emotion_embedding(args.text, args.emotion)
    print(f"Emotion embedding loaded: {args.emotion}")
    
    # Generate tokens
    tokens = synthesize_text(
        model, text_tokens, 
        speaker_id=speaker_id, 
        emotion_embedding=emotion_embedding,
        max_length=max_length
    )
    
    print(f"Generated tokens shape: {tokens.shape}")
    
    # Save tokens
    np.save(f"{output_path}_tokens.npy", tokens)
    print(f"Tokens saved to {output_path}_tokens.npy")
    
    # Save metadata
    metadata = {
        "input_text": args.text,
        "speaker_id": args.speaker_id,
        "emotion": args.emotion,
        "generated_tokens": len(token_outputs),
        "max_length": args.max_length
    }
    
    with open(f"{output_path}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Outputs saved to {output_path}_*.npy")
    print(f"Metadata saved to {output_path}_metadata.json")

if __name__ == "__main__":
    main()