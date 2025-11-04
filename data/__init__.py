import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import List, Tuple, Dict, Any

def pad_1D(inputs, PAD=0):
    """
    Pad 1D sequences to the same length.
    """
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):
    """
    Pad 2D sequences to the same length.
    """
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not supported")
        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

class EcholancerDataset(Dataset):
    """
    Dataset class for Echolancer model that loads data from .pt files.
    Each .pt file contains a list of dictionaries with audio-text pairs.
    All chunk files are guaranteed to have the same number of entries.
    """
    def __init__(self, data_dir: str, file_extension: str = ".pt", chunk_size: int = None):
        """
        Args:
            data_dir: Directory containing .pt files
            file_extension: Extension of the data files (default: ".pt")
            chunk_size: Number of items per chunk file (if known, to avoid loading files during init)
        """
        self.data_dir = data_dir
        self.file_extension = file_extension
        
        # Find all .pt files in the directory
        pattern = os.path.join(data_dir, f"*{file_extension}")
        self.file_list = sorted(glob.glob(pattern))
        
        if len(self.file_list) == 0:
            raise ValueError(f"No {file_extension} files found in {data_dir}")
        
        print(f"Found {len(self.file_list)} data files in {data_dir}")
        
        if chunk_size is not None:
            # If chunk_size is provided, we can calculate length without loading any files
            self.chunk_size = chunk_size
        else:
            # Only load the first file to determine chunk size
            first_file_data = torch.load(self.file_list[0], map_location='cpu')
            self.chunk_size = len(first_file_data) if isinstance(first_file_data, list) else 1

        # Calculate total length based on file count and chunk size
        self._length = len(self.file_list) * self.chunk_size

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Calculate which file and which index within that file
        file_idx = idx // self.chunk_size
        item_idx = idx % self.chunk_size
        file_path = self.file_list[file_idx]
        
        # Load the .pt file
        file_data = torch.load(file_path, map_location='cpu')
        
        # Get the specific item from the list
        data = file_data[item_idx] if isinstance(file_data, list) else file_data
        
        # Extract data fields from the chunk format:
        # - 'text': text transcription
        # - 'tokens': VQGAN audio tokens (1D tensor)
        # - 'start': start time 
        # - 'end': end time
        # - 'audio_path': path to audio file
        # - 'jsonl_path': path to jsonl file
        
        if isinstance(data, dict):
            # Map the VQGAN tokens to mel (since tokens are the discrete audio representation)
            text = data['text']
            mel = data['tokens'].squeeze(0)  # Remove first dimension to make it 1D
            start_time = data['start']
            end_time = data['end']
            audio_path = data['audio_path']
            jsonl_path = data['jsonl_path']
            
            # For now, set default values for speaker and emotion
            speaker = 0  # Default speaker ID
            emotion = torch.zeros(768)  # Default emotion embedding
        else:
            # Handle legacy format if needed
            raise ValueError(f"Unsupported data format in {file_path}")
        
        # Convert to the required format
        if isinstance(text, str):
            # If text is a string, we might need to tokenize it depending on implementation
            # For now, we'll store it as is but mark the length
            text_len = len(text.split())  # Approximate token length from text
        elif isinstance(text, torch.Tensor):
            text = text.numpy() if text.is_cuda else text.numpy()
            text_len = len(text)
        else:
            text_len = len(text)
            
        if isinstance(mel, torch.Tensor):
            mel = mel.numpy() if mel.is_cuda else mel.numpy()
        if isinstance(emotion, torch.Tensor):
            emotion = emotion.numpy() if emotion.is_cuda else emotion.numpy()
        if isinstance(speaker, torch.Tensor):
            speaker = speaker.item() if speaker.numel() == 1 else speaker.numpy()
        
        return {
            'text': text,
            'mel': mel,
            'speaker': speaker,
            'emotion': emotion,
            'text_len': text_len,
            'mel_len': len(mel),
            'start_time': start_time,
            'end_time': end_time,
            'audio_path': audio_path,
            'jsonl_path': jsonl_path,
            'file_path': file_path  # For debugging
        }

    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        """
        # Sort batch by mel length (descending) for efficient packing
        # Using mel length as it's the audio token sequence length
        batch = sorted(batch, key=lambda x: x['mel_len'], reverse=True)
        
        # Extract data
        texts = [item['text'] for item in batch]
        mels = [item['mel'] for item in batch]
        speakers = np.array([item['speaker'] for item in batch])
        emotions = np.array([item['emotion'] for item in batch])
        text_lens = np.array([item['text_len'] for item in batch])
        mel_lens = np.array([item['mel_len'] for item in batch])
        
        # Extract additional metadata
        start_times = [item['start_time'] for item in batch]
        end_times = [item['end_time'] for item in batch]
        audio_paths = [item['audio_path'] for item in batch]
        jsonl_paths = [item['jsonl_path'] for item in batch]
        
        # Padding
        # Handle string texts separately from arrays
        if isinstance(texts[0], str):
            # If texts are strings, keep them as is for now, the model needs to tokenize them
            processed_texts = texts
        else:
            # If texts are already tokenized arrays
            processed_texts = pad_1D(texts)
        
        padded_mels = pad_1D(mels)
        
        # Convert to tensors
        if isinstance(processed_texts[0], str):
            # If texts are strings, return as string list
            texts_tensor = processed_texts
        else:
            # If texts are arrays, convert to tensor
            texts_tensor = torch.from_numpy(processed_texts).long()
        
        mels_tensor = torch.from_numpy(padded_mels).long()
        speakers_tensor = torch.from_numpy(speakers).long()
        emotions_tensor = torch.from_numpy(emotions).float()
        text_lens_tensor = torch.from_numpy(text_lens).long()
        mel_lens_tensor = torch.from_numpy(mel_lens).long()  # This remains as 1D tensor of lengths
        
        return (
            [item['file_path'] for item in batch],  # ids/file paths
            texts_tensor,  # raw_texts (can be strings or tokenized)
            speakers_tensor,
            mels_tensor,  # VQGAN tokens
            text_lens_tensor,
            mels_tensor,  # Using same VQGAN tokens as target
            mel_lens_tensor,  # 1D tensor of actual lengths
            emotions_tensor
        )

def get_data_loader(data_dir: str, batch_size: int = 16, shuffle: bool = True, 
                   num_workers: int = 0, file_extension: str = ".pt"):
    """
    Create a DataLoader for the EcholancerDataset.
    
    Args:
        data_dir: Directory containing .pt files
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        file_extension: Extension of the data files (default: ".pt")
        
    Returns:
        DataLoader instance
    """
    dataset = EcholancerDataset(data_dir, file_extension)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

def create_dummy_data_files(data_dir: str, num_samples: int = 100, 
                           max_text_len: int = 20, max_mel_len: int = 100,
                           vocab_size: int = 100, emotion_dim: int = 768):
    """
    Create dummy .pt data files for testing and demonstration.
    
    Args:
        data_dir: Directory to save the .pt files
        num_samples: Number of sample files to create
        max_text_len: Maximum text length
        max_mel_len: Maximum mel length
        vocab_size: Size of the text vocabulary
        emotion_dim: Dimension of emotion embeddings
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Random text tokens
        text_len = np.random.randint(10, max_text_len)
        text = torch.randint(0, vocab_size, (text_len,))

        # Random mel tokens
        mel_len = np.random.randint(50, max_mel_len)
        mel = torch.randint(0, 1010, (mel_len,))

        # Speaker ID
        speaker = torch.randint(0, 10, (1,)).item()

        # Emotion embedding
        emotion = torch.randn(emotion_dim)

        # Create data dictionary
        data = {
            'text': text,
            'mel': mel,
            'speaker': speaker,
            'emotion': emotion
        }

        # Save to .pt file
        file_path = os.path.join(data_dir, f"sample_{i:05d}.pt")
        torch.save(data, file_path)
    
    print(f"Created {num_samples} dummy .pt files in {data_dir}")

# For backward compatibility
def create_dummy_data(num_samples=100, max_text_len=20, max_mel_len=100, 
                     vocab_size=100, emotion_dim=768):
    """
    Create dummy data for testing and demonstration (backward compatibility).
    """
    data_list = []
    for i in range(num_samples):
        # Random text tokens
        text_len = np.random.randint(10, max_text_len)
        text = np.random.randint(0, vocab_size, (text_len,)).tolist()

        # Random mel tokens
        mel_len = np.random.randint(50, max_mel_len)
        mel = np.random.randint(0, 1010, (mel_len,)).tolist()

        # Speaker ID
        speaker = np.random.randint(0, 10)

        # Emotion embedding
        emotion = np.random.randn(emotion_dim).tolist()

        data_list.append({
            'text': text,
            'mel': mel,
            'speaker': speaker,
            'emotion': emotion
        })
    
    return data_list

def get_data_loader_from_list(data_list, batch_size=16, shuffle=True, num_workers=0):
    """
    Create a DataLoader from a list of data (backward compatibility).
    """
    # This is for backward compatibility with the old API
    from torch.utils.data import Dataset as TorchDataset
    
    class ListDataset(TorchDataset):
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            data = self.data_list[idx]
            
            # Text processing
            text = np.array(data['text'])
            
            # Mel processing
            mel = np.array(data['mel'])
            
            # Speaker ID
            speaker = data['speaker']
            
            # Emotion embedding
            emotion = np.array(data['emotion'])
            
            return {
                'text': text,
                'mel': mel,
                'speaker': speaker,
                'emotion': emotion,
                'text_len': len(text),
                'mel_len': len(mel)
            }

        def collate_fn(self, batch):
            # Sort batch by text length (descending)
            batch = sorted(batch, key=lambda x: x['text_len'], reverse=True)
            
            # Extract data
            texts = [item['text'] for item in batch]
            mels = [item['mel'] for item in batch]
            speakers = np.array([item['speaker'] for item in batch])
            emotions = np.array([item['emotion'] for item in batch])
            text_lens = np.array([item['text_len'] for item in batch])
            mel_lens = np.array([item['mel_len'] for item in batch])
            
            # Padding
            texts = pad_1D(texts)
            mels = pad_1D(mels)
            
            # Convert to tensors
            texts = torch.from_numpy(texts).long()
            mels = torch.from_numpy(mels).long()
            speakers = torch.from_numpy(speakers).long()
            emotions = torch.from_numpy(emotions).float()
            text_lens = torch.from_numpy(text_lens).long()
            mel_lens = torch.from_numpy(mel_lens).long()
            
            return (
                [f"sample_{i}" for i in range(len(batch))],  # ids
                ["" for _ in range(len(batch))],  # raw_texts
                speakers,
                texts,
                text_lens,
                mels,
                mel_lens,
                emotions
            )
    
    dataset = ListDataset(data_list)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers
    )

from .text_dataset import TextCorpusDataset, TextCorpusWithMLMDataset, VQAudioDataset, PreprocessedVQAudioDataset

__all__ = [
    "EcholancerDataset",
    "TextCorpusDataset", 
    "TextCorpusWithMLMDataset",
    "VQAudioDataset",
    "PreprocessedVQAudioDataset",
    "get_data_loader",
    "create_dummy_data_files",
    "create_dummy_data",
    "get_data_loader_from_list"
]