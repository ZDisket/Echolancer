import torch
from torch.utils.data import Dataset
import os
import glob
from typing import List
import sys

# Add the parent directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the character tokenizer
from utils.char_tokenizer import CharTokenizer
from tqdm import tqdm


class TextCorpusDataset(Dataset):
    """
    Dataset class for loading and tokenizing text corpora.
    Loads a text file, tokenizes it with the character tokenizer,
    and returns chunks of specified length.
    """
    
    def __init__(self, text_file_path: str, chunk_length: int, tokenizer_path: str = None, 
                 add_bos: bool = False, add_eos: bool = False):
        """
        Initialize the text corpus dataset.
        
        Args:
            text_file_path: Path to the text file containing the corpus
            chunk_length: Length of each chunk in tokens
            tokenizer_path: Optional path to a pre-saved tokenizer. If None, creates a new one
            add_bos: Whether to add beginning-of-sequence tokens to each chunk
            add_eos: Whether to add end-of-sequence tokens to each chunk
        """
        self.text_file_path = text_file_path
        self.chunk_length = chunk_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Load or create the tokenizer
        if tokenizer_path is not None and os.path.exists(tokenizer_path):
            self.tokenizer = CharTokenizer.load(tokenizer_path)
        else:
            # Create a new tokenizer
            self.tokenizer = CharTokenizer()
        
        # Load and preprocess the entire text corpus
        self._load_text_corpus()
    
    def _load_text_corpus(self):
        """
        Load the text file and tokenize it, with caching support.
        """
        if not os.path.exists(self.text_file_path):
            raise FileNotFoundError(f"Text file not found: {self.text_file_path}")

        # Determine cache file path
        base_path = os.path.splitext(self.text_file_path)[0]
        cache_file_path = base_path + '_tokenized.pt'

        # Check if cached tokenized data exists
        if os.path.exists(cache_file_path):
            print(f"Loading cached tokenized data from {cache_file_path}")
            cached_data = torch.load(cache_file_path)
            self.encoded_text = cached_data['encoded_text']
            self.total_chunks = cached_data['total_chunks']
            print(f"Loaded cached data with {len(self.encoded_text)} tokens")
            print(f"Total chunks of length {self.chunk_length}: {self.total_chunks}")
        else:
            print(f"Loading text corpus from {self.text_file_path}")
            # Read the text file
            with open(self.text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Preprocess text (this is handled within the tokenizer too, but we do it here as well)
            # Tokenize the entire text corpus
            self.encoded_text = self.tokenizer.encode(text, add_bos=self.add_bos, add_eos=self.add_eos)

            print(f"Loaded text corpus with {len(text)} characters")
            print(f"Tokenized to {len(self.encoded_text)} tokens")

            # Calculate total number of chunks
            self.total_chunks = len(self.encoded_text) // self.chunk_length
            if len(self.encoded_text) % self.chunk_length > 0:
                self.total_chunks += 1
            print(f"Total chunks of length {self.chunk_length}: {self.total_chunks}")

            # Save the tokenized data to cache
            cache_data = {
                'encoded_text': self.encoded_text,
                'total_chunks': self.total_chunks
            }
            print(f"Saving tokenized data to cache: {cache_file_path}")
            torch.save(cache_data, cache_file_path)
    
    def __len__(self):
        """
        Return the total number of chunks.
        """
        return self.total_chunks
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a chunk of tokenized text at the specified index.
        
        Args:
            idx: Index of the chunk to retrieve
            
        Returns:
            Tensor of token IDs with shape (chunk_length,)
        """
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} is out of range for dataset with {self.total_chunks} chunks")

        # Calculate the start and end positions in the encoded text
        start_idx = idx * self.chunk_length
        end_idx = min((idx + 1) * self.chunk_length, len(self.encoded_text))

        # Extract the chunk
        chunk = self.encoded_text[start_idx:end_idx]

        # Pad if necessary to ensure consistent length
        if len(chunk) < self.chunk_length:
            # Pad with the pad token ID
            pad_token_id = self.tokenizer.vocab.get(self.tokenizer.special_tokens['pad'], 0)
            chunk.extend([pad_token_id] * (self.chunk_length - len(chunk)))

        # Convert to tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)

        return chunk_tensor
    
    def decode_chunk(self, chunk_tensor: torch.Tensor) -> str:
        """
        Decode a chunk tensor back to text.
        
        Args:
            chunk_tensor: Tensor of token IDs to decode
            
        Returns:
            Decoded text string
        """
        chunk_list = chunk_tensor.tolist()
        return self.tokenizer.decode(chunk_list)
    
    def get_tokenizer(self):
        """
        Get the tokenizer used by this dataset.
        
        Returns:
            The CharTokenizer instance
        """
        return self.tokenizer
    
    def save_tokenizer(self, tokenizer_path: str):
        """
        Save the tokenizer to a file.
        
        Args:
            tokenizer_path: Path to save the tokenizer
        """
        self.tokenizer.save(tokenizer_path)


class TextCorpusWithMLMDataset(Dataset):
    """
    Dataset class for loading and tokenizing text corpora with support for Masked Language Modeling (MLM).
    Loads a text file, tokenizes it with the character tokenizer, and returns chunks of specified length
    with randomly masked tokens for MLM pretraining.
    """
    
    def __init__(self, text_file_path: str, chunk_length: int, tokenizer_path: str = None,
                 mask_fraction: float = 0.15, mask_token: str = '[MASK]', add_bos: bool = False, 
                 add_eos: bool = False):
        """
        Initialize the text corpus dataset with MLM support.
        
        Args:
            text_file_path: Path to the text file containing the corpus
            chunk_length: Length of each chunk in tokens
            tokenizer_path: Optional path to a pre-saved tokenizer. If None, creates a new one
            mask_fraction: Fraction of tokens to mask for MLM (default: 0.15)
            mask_token: Token to use for masking (will be added to vocabulary if not present)
            add_bos: Whether to add beginning-of-sequence tokens to each chunk
            add_eos: Whether to add end-of-sequence tokens to each chunk
        """
        self.text_file_path = text_file_path
        self.chunk_length = chunk_length
        self.mask_fraction = mask_fraction
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Load or create the tokenizer
        if tokenizer_path is not None and os.path.exists(tokenizer_path):
            self.tokenizer = CharTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = CharTokenizer()
        
        # Load and preprocess the entire text corpus
        self._load_text_corpus()
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def _load_text_corpus(self):
        """
        Load the text file and tokenize it, with caching support.
        """
        if not os.path.exists(self.text_file_path):
            raise FileNotFoundError(f"Text file not found: {self.text_file_path}")

        # Determine cache file path for tokenized data
        base_path = os.path.splitext(self.text_file_path)[0]
        cache_file_path = base_path + '_tokenized.pt'
        # Also determine cache file path for precomputed MLM data
        mlm_cache_file_path = base_path + '_mlm_precomputed.pt'

        # Check if cached precomputed MLM data exists (this is the fastest option)
        if os.path.exists(mlm_cache_file_path):
            print(f"Loading cached precomputed MLM data from {mlm_cache_file_path}")
            cached_data = torch.load(mlm_cache_file_path)
            self.encoded_text = cached_data['encoded_text']
            self.total_chunks = cached_data['total_chunks']
            self.mlm_inputs = cached_data['mlm_inputs']
            self.mlm_targets = cached_data['mlm_targets']
            self.mlm_masks = cached_data['mlm_masks']
            print(f"Loaded cached data with {len(self.encoded_text)} tokens")
            print(f"Total chunks of length {self.chunk_length}: {self.total_chunks}")
        # Check if cached tokenized data exists (then we need to compute MLM data)
        elif os.path.exists(cache_file_path):
            print(f"Loading cached tokenized data from {cache_file_path}")
            cached_data = torch.load(cache_file_path)
            self.encoded_text = cached_data['encoded_text']
            self.total_chunks = cached_data['total_chunks']
            print(f"Loaded cached data with {len(self.encoded_text)} tokens")
            print(f"Total chunks of length {self.chunk_length}: {self.total_chunks}")
            
            # Compute MLM data for all chunks and cache it
            print("Computing MLM data for all chunks...")
            self._compute_mlm_data()
            
            # Cache the precomputed MLM data
            mlm_cache_data = {
                'encoded_text': self.encoded_text,
                'total_chunks': self.total_chunks,
                'mlm_inputs': self.mlm_inputs,
                'mlm_targets': self.mlm_targets, 
                'mlm_masks': self.mlm_masks
            }
            print(f"Saving precomputed MLM data to cache: {mlm_cache_file_path}")
            torch.save(mlm_cache_data, mlm_cache_file_path)
        else:
            print(f"Loading text corpus from {self.text_file_path}")
            # Read the text file
            with open(self.text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Tokenize the entire text corpus
            self.encoded_text = self.tokenizer.encode(text, add_bos=self.add_bos, add_eos=self.add_eos)

            print(f"Loaded text corpus with {len(text)} characters")
            print(f"Tokenized to {len(self.encoded_text)} tokens")

            # Calculate total number of chunks
            self.total_chunks = len(self.encoded_text) // self.chunk_length
            if len(self.encoded_text) % self.chunk_length > 0:
                self.total_chunks += 1
            print(f"Total chunks of length {self.chunk_length}: {self.total_chunks}")

            # Compute MLM data for all chunks
            print("Computing MLM data for all chunks...")
            self._compute_mlm_data()
            
            # Save the tokenized data to cache
            cache_data = {
                'encoded_text': self.encoded_text,
                'total_chunks': self.total_chunks
            }
            print(f"Saving tokenized data to cache: {cache_file_path}")
            torch.save(cache_data, cache_file_path)
            
            # Also save precomputed MLM data to cache
            mlm_cache_data = {
                'encoded_text': self.encoded_text,
                'total_chunks': self.total_chunks,
                'mlm_inputs': self.mlm_inputs,
                'mlm_targets': self.mlm_targets,
                'mlm_masks': self.mlm_masks
            }
            print(f"Saving precomputed MLM data to cache: {mlm_cache_file_path}")
            torch.save(mlm_cache_data, mlm_cache_file_path)

    def _compute_mlm_data(self):
        """
        Compute MLM inputs, targets, and masks for all chunks.
        This replaces the on-the-fly computation in __getitem__ for better performance.
        """
        # Initialize lists to store precomputed MLM data
        self.mlm_inputs = []
        self.mlm_targets = []
        self.mlm_masks = []
        
        # Precompute MLM data for each chunk
        for idx in tqdm(range(self.total_chunks)):
            # Calculate the start and end positions in the encoded text
            start_idx = idx * self.chunk_length
            end_idx = min((idx + 1) * self.chunk_length, len(self.encoded_text))

            # Extract the chunk
            chunk = self.encoded_text[start_idx:end_idx]

            # Pad if necessary to ensure consistent length
            original_length = len(chunk)
            if len(chunk) < self.chunk_length:
                # Pad with the pad token ID
                pad_token_id = self.tokenizer.vocab.get(self.tokenizer.special_tokens['pad'], 0)
                chunk.extend([pad_token_id] * (self.chunk_length - len(chunk)))

            # Convert to tensor
            original_tensor = torch.tensor(chunk, dtype=torch.long)

            # Create a copy for the input that will have masked tokens
            input_tensor = original_tensor.clone()

            # Create mask tensor (True for actual tokens, False for padding)
            mask_tensor = torch.zeros(self.chunk_length, dtype=torch.bool)
            mask_tensor[:original_length] = True  # Mark actual tokens as True

            # Apply MLM masking to non-padding, non-special tokens
            maskable_positions = []
            for i in range(min(original_length, self.chunk_length)):
                token_id = original_tensor[i].item()
                # Find positions that are not special tokens
                if token_id not in [self.tokenizer.vocab.get(token, -1) for token in self.tokenizer.special_tokens.values()]:
                    maskable_positions.append(i)

            # Determine how many tokens to mask
            num_maskable = len(maskable_positions)
            num_to_mask = max(1, int(num_maskable * self.mask_fraction))  # Ensure at least 1 token is masked if possible

            # Select random positions to mask
            if num_maskable > 0:
                # Randomly select positions to mask
                mask_indices = torch.randperm(len(maskable_positions))[:num_to_mask]
                mask_indices = [maskable_positions[i] for i in mask_indices]

                # Apply masking (replace with [MASK] token randomly ~80%, random token ~10%, keep original ~10%)
                mask_token_id = self.tokenizer.vocab.get('[MASK]', 
                                                       self.tokenizer.vocab.get(self.tokenizer.special_tokens['unk'], 0))

                for pos in mask_indices:
                    rand = torch.rand(1).item()
                    if rand < 0.8:  # 80% of the time, replace with [MASK]
                        input_tensor[pos] = mask_token_id
                    elif rand < 0.9:  # 10% of the time, replace with random token
                        random_token_id = torch.randint(0, len(self.tokenizer), (1,)).item()
                        # Make sure it's not a special token
                        if random_token_id in [self.tokenizer.vocab.get(token, -1) for token in self.tokenizer.special_tokens.values()]:
                            random_token_id = self.tokenizer.vocab.get(self.tokenizer.special_tokens['unk'], 0)
                        input_tensor[pos] = random_token_id
                    # 10% of the time, keep the original token (no action needed)

            # Update mask tensor to indicate which tokens were masked
            mlm_mask = torch.zeros(self.chunk_length, dtype=torch.bool)
            for pos in mask_indices if num_maskable > 0 else []:
                mlm_mask[pos] = True

            # Store the precomputed data
            self.mlm_inputs.append(input_tensor)
            self.mlm_targets.append(original_tensor)
            self.mlm_masks.append(mlm_mask)
    
    def __len__(self):
        """
        Return the total number of chunks.
        """
        return self.total_chunks
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a precomputed MLM chunk at the specified index.
        
        Args:
            idx: Index of the chunk to retrieve
            
        Returns:
            A tuple containing:
            - input_tensor: Tensor of token IDs with some masked (shape: chunk_length,)
            - target_tensor: Tensor of original token IDs (shape: chunk_length,)
            - mask_tensor: Boolean tensor indicating masked positions (True = masked, shape: chunk_length,)
        """
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} is out of range for dataset with {self.total_chunks} chunks")

        # Return precomputed MLM data directly
        return self.mlm_inputs[idx], self.mlm_targets[idx], self.mlm_masks[idx]
    
    def decode_chunk(self, chunk_tensor: torch.Tensor) -> str:
        """
        Decode a chunk tensor back to text.
        
        Args:
            chunk_tensor: Tensor of token IDs to decode
            
        Returns:
            Decoded text string
        """
        chunk_list = chunk_tensor.tolist()
        return self.tokenizer.decode(chunk_list)
    
    def get_tokenizer(self):
        """
        Get the tokenizer used by this dataset.
        
        Returns:
            The CharTokenizer instance
        """
        return self.tokenizer
    
    def save_tokenizer(self, tokenizer_path: str):
        """
        Save the tokenizer to a file.
        
        Args:
            tokenizer_path: Path to save the tokenizer
        """
        self.tokenizer.save(tokenizer_path)


class VQAudioDataset(Dataset):
    """
    Dataset class for loading TTS training dataset chunks and returning only VQ tokens
    in fixed-length sequences of N.
    This loads all data into memory and should only be used for smaller datasets.
    For large datasets, use PreprocessedVQAudioDataset with prepared files.
    """
    
    def __init__(self, data_dir: str, chunk_length: int, file_extension: str = ".pt"):
        """
        Initialize the VQ audio dataset.
        
        Args:
            data_dir: Directory containing the .pt chunk files
            chunk_length: Length of each chunk in tokens
            file_extension: Extension of the data files (default: ".pt")
        """
        self.data_dir = data_dir
        self.chunk_length = chunk_length
        self.file_extension = file_extension
        
        # Find all .pt files in the directory
        import glob
        pattern = os.path.join(data_dir, f"*{file_extension}")
        self.file_list = sorted(glob.glob(pattern))
        
        if len(self.file_list) == 0:
            raise ValueError(f"No {file_extension} files found in {data_dir}")
        
        print(f"Found {len(self.file_list)} data files in {data_dir}")
        
        # Load all VQ tokens from all chunk files into a single sequence
        self._load_vq_tokens()
    
    def _load_vq_tokens(self):
        """
        Load VQ tokens from all chunk files.
        """
        import torch
        
        all_tokens_list = []
        
        for file_path in self.file_list:
            # Load the chunk file
            chunk_data = torch.load(file_path, map_location='cpu')
            
            # Process each dictionary in the chunk (each contains 'tokens')
            for item in chunk_data:
                if isinstance(item, dict) and 'tokens' in item:
                    # Extract VQ tokens, squeeze if needed to make it 1D
                    tokens = item['tokens']
                    if tokens.dim() > 1:
                        tokens = tokens.squeeze(0)  # Remove first dimension if needed
                    all_tokens_list.extend(tokens.tolist())
                else:
                    raise ValueError(f"Expected dictionary with 'tokens' key in {file_path}")
        
        self.all_vq_tokens = all_tokens_list
        print(f"Loaded {len(self.all_vq_tokens)} VQ tokens from {len(self.file_list)} files")
        
        # Calculate total number of chunks
        self.total_chunks = len(self.all_vq_tokens) // self.chunk_length
        if len(self.all_vq_tokens) % self.chunk_length > 0:
            self.total_chunks += 1
        print(f"Total chunks of length {self.chunk_length}: {self.total_chunks}")
    
    def __len__(self):
        """
        Return the total number of chunks.
        """
        return self.total_chunks
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a chunk of VQ tokens at the specified index.
        
        Args:
            idx: Index of the chunk to retrieve
            
        Returns:
            Tensor of VQ token IDs with shape (chunk_length,)
        """
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} is out of range for dataset with {self.total_chunks} chunks")
        
        # Calculate the start and end positions in the token sequence
        start_idx = idx * self.chunk_length
        end_idx = min((idx + 1) * self.chunk_length, len(self.all_vq_tokens))
        
        # Extract the chunk
        chunk = self.all_vq_tokens[start_idx:end_idx]
        
        # Pad if necessary to ensure consistent length
        if len(chunk) < self.chunk_length:
            # Pad with zeros (or another appropriate value)
            chunk.extend([0] * (self.chunk_length - len(chunk)))
        
        # Convert to tensor
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        
        return chunk_tensor


class PreprocessedVQAudioDataset(Dataset):
    """
    Dataset class for loading preprocessed VQ token sequences from prepared .pt files.
    Each file contains a list of fixed-length VQ token sequences.
    This dataset loads files sequentially without loading the entire dataset into memory.
    """
    
    def __init__(self, data_dir: str, file_extension: str = ".pt", cache_files: bool = False, offset=0, max_files=None):
        """
        Initialize the preprocessed VQ audio dataset.
        
        Args:
            data_dir: Directory containing the preprocessed .pt files
            file_extension: Extension of the data files (default: ".pt")
            cache_files: Whether to cache loaded files in memory for faster access (default: False)
        """
        self.data_dir = data_dir
        self.file_extension = file_extension
        self.cache_files = cache_files
        self.offset = offset
        
        # Find all preprocessed files in the directory
        import glob
        pattern = os.path.join(data_dir, f"*{file_extension}")
        self.file_list = sorted(glob.glob(pattern))
        if max_files is not None:
            self.file_list = self.file_list[:max_files]
        
        if len(self.file_list) == 0:
            raise ValueError(f"No {file_extension} files found in {data_dir}")
        
        print(f"Found {len(self.file_list)} preprocessed data files in {data_dir}")
        print(f"Data offset: {self.offset}")
        
        # Calculate the total number of sequences by checking file sizes
        self._calculate_total_sequences()
    
    def _calculate_total_sequences(self):
        """
        Calculate total number of sequences across all files without loading them completely.
        """
        if not self.file_list:
            self.total_sequences = 0
            self.sequences_per_file = 0
            self.last_file_sequences = 0
            return
        
        # Load first file to determine sequences per file
        first_sequences = torch.load(self.file_list[0], map_location='cpu')
        self.sequences_per_file = len(first_sequences)
        
        # Load last file to determine if it has fewer sequences
        last_sequences = torch.load(self.file_list[-1], map_location='cpu')
        self.last_file_sequences = len(last_sequences)
        
        # Calculate total sequences accounting for the potential shorter last file
        if len(self.file_list) == 1:
            self.total_sequences = self.last_file_sequences
        else:
            self.total_sequences = (len(self.file_list) - 1) * self.sequences_per_file + self.last_file_sequences
        
        print(f"Total sequences across all files: {self.total_sequences} ({len(self.file_list)-1} full files × {self.sequences_per_file} + 1 partial file × {self.last_file_sequences})")
        
        # If caching files, pre-load them during initialization
        if self.cache_files:
            print("Caching files in memory...")
            self._cache_files()
    
    def _cache_files(self):
        """Cache all files in memory for faster access."""
        self.cached_sequences = []
        for file_path in tqdm(self.file_list, desc="Caching files"):
            sequences = torch.load(file_path, map_location='cpu')
            self.cached_sequences.append(sequences)
        print(f"Cached {len(self.cached_sequences)} files in memory")
    
    def __len__(self):
        """
        Return the total number of sequences.
        """
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a VQ token sequence at the specified index.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Tensor of VQ token IDs with shape matching the sequence length in the files
        """
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(f"Index {idx} is out of range for dataset with {self.total_sequences} sequences")
        
        # Calculate which file and sequence index using arithmetic
        # O(1) arithmetic calculation instead of O(log n) binary search
        file_idx = idx // self.sequences_per_file
        seq_idx_in_file = idx % self.sequences_per_file
        
        # Handle the special case for the last file which might have fewer sequences
        if file_idx == len(self.file_list) - 1:  # Last file
            # In the last file, make sure we don't exceed its actual sequence count
            if seq_idx_in_file >= self.last_file_sequences:
                raise IndexError(f"Sequence index {seq_idx_in_file} exceeds last file's sequence count of {self.last_file_sequences}")
        else:
            # In other files, make sure sequence index is within bounds
            if seq_idx_in_file >= self.sequences_per_file:
                raise IndexError(f"Sequence index {seq_idx_in_file} exceeds file's sequence count of {self.sequences_per_file}")
        
        # Validate file index is within range
        if file_idx >= len(self.file_list):
            raise IndexError(f"Calculated file index {file_idx} is out of range for {len(self.file_list)} files")
        
        # Load the file and get the specific sequence
        if self.cache_files:
            sequences = self.cached_sequences[file_idx]
        else:
            file_path = self.file_list[file_idx]
            sequences = torch.load(file_path, map_location='cpu')

        sequence = sequences[seq_idx_in_file] + self.offset
        return sequence


if __name__ == "__main__":
    # Example usage
    print("Testing TextCorpusDataset...")
    
    # Create a sample text file for testing
    sample_text = """This is a sample text corpus for testing the TextCorpusDataset.
    It contains multiple sentences with various punctuation marks.
    The tokenizer should handle this text properly by converting it to character-level tokens."""
    
    sample_file = "sample_corpus.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    # Test the dataset
    try:
        dataset = TextCorpusDataset(sample_file, chunk_length=50)
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Vocabulary size: {len(dataset.get_tokenizer())}")
        
        # Get a sample chunk
        sample_chunk = dataset[0]
        print(f"Chunk shape: {sample_chunk.shape}")
        print(f"Sample chunk: {sample_chunk.tolist()}")
        print(f"Decoded chunk: '{dataset.decode_chunk(sample_chunk)}'")
        
        # Test MLM dataset
        print("\nTesting TextCorpusWithMLMDataset...")
        mlm_dataset = TextCorpusWithMLMDataset(sample_file, chunk_length=50, mask_fraction=0.2)
        
        print(f"MLM Dataset length: {len(mlm_dataset)}")
        print(f"Vocabulary size: {len(mlm_dataset.get_tokenizer())}")
        
        # Get a sample with MLM
        input_chunk, target_chunk, mlm_mask = mlm_dataset[0]
        print(f"Input chunk shape: {input_chunk.shape}")
        print(f"Target chunk shape: {target_chunk.shape}")
        print(f"Mask shape: {mlm_mask.shape}")
        print(f"Number of masked positions: {mlm_mask.sum().item()}")
        print(f"Input chunk: {input_chunk.tolist()}")
        print(f"Target chunk: {target_chunk.tolist()}")
        print(f"Mask positions: {mlm_mask.tolist()}")
        print(f"Input decoded: '{mlm_dataset.decode_chunk(input_chunk)}'")
        print(f"Target decoded: '{mlm_dataset.decode_chunk(target_chunk)}'")
    
    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)