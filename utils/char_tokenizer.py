"""
Character-level tokenizer for converting text to indices and back.

This tokenizer uses a fixed vocabulary of English lowercase letters and 
speech-relevant punctuation, allowing for consistent conversion between 
text and numerical representations.
"""

import json
import os
import re
import unicodedata
from typing import List, Dict, Union


class CharTokenizer:
    """
    A simple character-level tokenizer that converts text to indices and back.
    
    The tokenizer uses a fixed vocabulary of English lowercase letters and
    speech-relevant punctuation. It handles unknown characters by mapping 
    them to a special unknown token.
    """
    
    def __init__(self, special_tokens: Dict[str, str] = None):
        """
        Initialize the character-level tokenizer with a fixed vocabulary.
        
        Args:
            special_tokens: Dictionary of special tokens (e.g., {'pad': '<PAD>', 'unk': '<UNK>'})
        """
        self.special_tokens = special_tokens or {
            'pad': '<PAD>',
            'unk': '<UNK>',
            'bos': '<BOS>',  # Beginning of sequence
            'eos': '<EOS>'   # End of sequence
        }
        
        # Build fixed vocabulary with English lowercase letters and speech-relevant punctuation
        self._build_fixed_vocab()
        
        # Mark as fitted since we're using a fixed vocabulary
        self.fitted = True
    
    def _build_fixed_vocab(self):
        """Build fixed vocabulary with English letters and speech-relevant punctuation."""
        # Start with special tokens
        self.vocab = {}
        self.inverse_vocab = {}
        idx = 0
        
        # Add special tokens
        for token_name, token_value in self.special_tokens.items():
            self.vocab[token_value] = idx
            self.inverse_vocab[idx] = token_value
            idx += 1
        
        # Add English lowercase letters
        for char in 'abcdefghijklmnopqrstuvwxyz':
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            idx += 1
        
        # Add speech-relevant punctuation
        punctuation = " .,!?;:'\"()-"
        for char in punctuation:
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            idx += 1
        
        # Store the actual vocab size
        self.vocab_size = len(self.vocab)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for tokenization.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace (collapse multiple spaces, tabs, newlines to single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def fit(self, texts: Union[str, List[str]]) -> 'CharTokenizer':
        """
        Dummy fit method for compatibility. Vocabulary is fixed.
        
        Args:
            texts: A string or list of strings (ignored in this implementation)
            
        Returns:
            Self (for method chaining)
        """
        # Vocabulary is fixed, so we don't need to do anything here
        # This method exists for API compatibility
        return self


    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert text to a list of indices.
        
        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of integer indices
        """
        if not self.fitted:
            raise ValueError("Tokenizer must be fitted before encoding. Call fit() first.")
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Convert text to indices
        indices = []
        
        # Add beginning-of-sequence token if requested
        if add_bos:
            indices.append(self.vocab[self.special_tokens['bos']])
        
        # Convert each character to its index
        for char in text:
            if char in self.vocab:
                indices.append(self.vocab[char])
            else:
                # Use unknown token for unseen characters
                indices.append(self.vocab[self.special_tokens['unk']])
        
        # Add end-of-sequence token if requested
        if add_eos:
            indices.append(self.vocab[self.special_tokens['eos']])
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Convert a list of indices back to text.
        
        Args:
            indices: List of integer indices to decode
            skip_special: Whether to skip special tokens in the output
            
        Returns:
            Decoded text string
        """
        if not self.fitted:
            raise ValueError("Tokenizer must be fitted before decoding. Call fit() first.")
        
        # Convert indices to characters
        chars = []
        special_token_indices = {self.vocab[token] for token in self.special_tokens.values() 
                                if token in self.vocab}
        
        for idx in indices:
            if idx in self.inverse_vocab:
                char = self.inverse_vocab[idx]
                # Skip special tokens if requested
                if not skip_special or idx not in special_token_indices:
                    chars.append(char)
            else:
                # Handle unknown indices
                chars.append(self.special_tokens['unk'])
        
        return ''.join(chars)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into individual characters (tokens).
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of character tokens
        """
        return list(text)
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        return self.vocab.copy()
    
    def save(self, filepath: str):
        """
        Save the tokenizer to a file.
        
        Args:
            filepath: Path to save the tokenizer
        """
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab,
            'inverse_vocab': {str(k): v for k, v in self.inverse_vocab.items()},
            'fitted': self.fitted
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CharTokenizer':
        """
        Load a tokenizer from a file.
        
        Args:
            filepath: Path to load the tokenizer from
            
        Returns:
            Loaded CharTokenizer instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Create new tokenizer instance
        tokenizer = cls(
            special_tokens=tokenizer_data['special_tokens']
        )
        
        # Restore vocabulary
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.inverse_vocab = {int(k): v for k, v in tokenizer_data['inverse_vocab'].items()}
        tokenizer.fitted = tokenizer_data['fitted']
        tokenizer.vocab_size = tokenizer_data.get('vocab_size', len(tokenizer.vocab))
        
        return tokenizer
    
    def __len__(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    print("=== Basic Usage ===")
    
    # Create tokenizer (no fitting needed for fixed vocabulary)
    tokenizer = CharTokenizer()
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Sample vocabulary: {dict(list(tokenizer.get_vocab().items())[:10])}")
    
    # Encode and decode
    text = "hello, world!"
    indices = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(indices)
    
    print(f"Original text: {text}")
    print(f"Encoded indices: {indices}")
    print(f"Decoded text: {decoded}")
    
    # Example 2: Saving and loading
    print("\n=== Saving and Loading ===")
    tokenizer.save("char_tokenizer.json")
    loaded_tokenizer = CharTokenizer.load("char_tokenizer.json")
    
    # Test that loaded tokenizer works the same
    test_text = "test loading functionality."
    original_indices = tokenizer.encode(test_text)
    loaded_indices = loaded_tokenizer.encode(test_text)
    
    print(f"Original indices: {original_indices}")
    print(f"Loaded indices: {loaded_indices}")
    print(f"Match: {original_indices == loaded_indices}")
    
    # Clean up
    if os.path.exists("char_tokenizer.json"):
        os.remove("char_tokenizer.json")
    
    print("\n=== Handling Unknown Characters ===")
    # Test with unknown characters
    unknown_text = "Unknown characters: []{}"  # Characters not in our fixed vocabulary
    indices = tokenizer.encode(unknown_text)
    decoded = tokenizer.decode(indices)
    
    print(f"Original with unknown chars: {unknown_text}")
    print(f"Encoded indices: {indices}")
    print(f"Decoded with unknowns: {decoded}")