"""
Example usage of the CharTokenizer with the Echolancer model.

This example shows how to use the character-level tokenizer with the Echolancer model
for text-to-speech synthesis.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.char_tokenizer import CharTokenizer
from utils import get_model, get_param_num

def create_sample_tokenizer():
    """Create a tokenizer with fixed vocabulary."""
    # Create tokenizer with fixed vocabulary (no fitting needed)
    tokenizer = CharTokenizer()
    
    # Save the tokenizer for later use
    tokenizer.save("sample_char_tokenizer.json")
    
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print("Sample vocabulary mappings:")
    for i, (char, idx) in enumerate(list(tokenizer.get_vocab().items())[:10]):
        print(f"  '{char}' -> {idx}")
    
    return tokenizer

def demonstrate_tokenization(tokenizer):
    """Demonstrate tokenization and detokenization."""
    # Example text with mixed case
    text = "Hello, World!"
    
    # Encode text to indices
    indices = tokenizer.encode(text, add_bos=True, add_eos=True)
    print(f"\nOriginal text: '{text}'")
    print(f"Encoded indices: {indices}")
    
    # Decode indices back to text
    decoded = tokenizer.decode(indices)
    print(f"Decoded text: '{decoded}'")
    
    # Show text preprocessing features
    print("\n=== Text Preprocessing Examples ===")
    
    # Example with extra whitespace
    whitespace_text = "Hello,    World!\n\t  How are you?"
    whitespace_indices = tokenizer.encode(whitespace_text)
    whitespace_decoded = tokenizer.decode(whitespace_indices)
    print(f"Text with extra whitespace: {repr(whitespace_text)}")
    print(f"Decoded after preprocessing: '{whitespace_decoded}'")
    
    # Example with unicode characters
    unicode_text = "Café résumé"
    unicode_indices = tokenizer.encode(unicode_text)
    unicode_decoded = tokenizer.decode(unicode_indices)
    print(f"Text with unicode chars: {repr(unicode_text)}")
    print(f"Decoded after preprocessing: '{unicode_decoded}'")
    
    # Show what happens with unknown characters
    unknown_text = "Unknown: []{}"  # Characters not in our fixed vocabulary
    unknown_indices = tokenizer.encode(unknown_text)
    unknown_decoded = tokenizer.decode(unknown_indices)
    print(f"\nText with unknown chars: {repr(unknown_text)}")
    print(f"Encoded indices: {unknown_indices}")
    print(f"Decoded (unknowns as <UNK>): '{unknown_decoded}'")

def integrate_with_model(tokenizer):
    """Show how to integrate the tokenizer with the Echolancer model."""
    # Create a small Echolancer model for demonstration
    model = get_model(
        vocab_size=tokenizer.get_vocab_size(),
        encoder_hidden=64,
        encoder_head=2,
        encoder_layer=2,
        decoder_hidden=64,
        decoder_layer=2,
        decoder_head=2,
        mel_channels=80,
        emotion_channels=32,
        speaker_channels=16,
        vq_token_mode=False,
        vq_vocab_size=1024,
        encoder_kv_heads=None,
        decoder_kv_heads=None
    )
    
    print(f"\nCreated Echolancer model with {get_param_num(model) / 1e6:.2f}M parameters")
    
    # Example of converting text to model inputs
    text = "Hello, World!"  # Mixed case text that will be lowercased
    text_indices = tokenizer.encode(text)
    
    print(f"\nText for model input: '{text}'")
    print(f"Token indices: {text_indices}")
    
    # Convert to tensor for model input
    text_tensor = torch.tensor([text_indices], dtype=torch.long)
    print(f"Text tensor shape: {text_tensor.shape}")
    
    # Sequence lengths
    src_lens = torch.tensor([len(text_indices)], dtype=torch.long)
    print(f"Source lengths: {src_lens}")

def main():
    print("=== CharTokenizer Example ===")
    
    # Create and train tokenizer
    tokenizer = create_sample_tokenizer()
    
    # Demonstrate tokenization
    demonstrate_tokenization(tokenizer)
    
    # Show integration with model
    integrate_with_model(tokenizer)
    
    # Clean up
    if os.path.exists("sample_char_tokenizer.json"):
        os.remove("sample_char_tokenizer.json")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()