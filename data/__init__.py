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
    """
    def __init__(self, data_dir: str, file_extension: str = ".pt"):
        """
        Args:
            data_dir: Directory containing .pt files
            file_extension: Extension of the data files (default: ".pt")
        """
        self.data_dir = data_dir
        self.file_extension = file_extension
        
        # Find all .pt files in the directory
        pattern = os.path.join(data_dir, f"*{file_extension}")
        self.file_list = sorted(glob.glob(pattern))
        
        if len(self.file_list) == 0:
            raise ValueError(f"No {file_extension} files found in {data_dir}")
        
        print(f"Found {len(self.file_list)} data files in {data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load the .pt file
        file_path = self.file_list[idx]
        data = torch.load(file_path, map_location='cpu')
        
        # Extract data fields
        # Assuming the .pt file contains a dictionary with the following keys:
        # - 'text': text token IDs (1D tensor)
        # - 'mel': mel-spectrogram tokens (1D tensor)
        # - 'speaker': speaker ID (int or tensor)
        # - 'emotion': emotion embedding (1D tensor)
        
        # Handle different data formats
        if isinstance(data, dict):
            text = data['text']
            mel = data['mel']
            speaker = data['speaker']
            emotion = data.get('emotion', torch.zeros(768))  # Default emotion embedding
        else:
            # Assume it's a tuple/list in order: text, mel, speaker, emotion
            text, mel, speaker, emotion = data[:4]
            if len(data) < 4:
                emotion = torch.zeros(768)  # Default emotion embedding
        
        # Convert to the required format
        if isinstance(text, torch.Tensor):
            text = text.numpy() if text.is_cuda else text.numpy()
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
            'text_len': len(text),
            'mel_len': len(mel),
            'file_path': file_path  # For debugging
        }

    def collate_fn(self, batch):
        """
        Collate function for the DataLoader.
        """
        # Sort batch by text length (descending) for efficient packing
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
            [item['file_path'] for item in batch],  # ids/file paths
            ["" for _ in range(len(batch))],  # raw_texts (empty for now)
            speakers,
            texts,
            text_lens,
            mels,
            mel_lens,
            emotions
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