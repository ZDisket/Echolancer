"""
Build Audio Enhancement Dataset

Creates paired original/codec audio files for training an audio enhancement model
that repairs codec artifacts.

Outputs:
- output_dir/original/*.wav  - Original audio segments at 48kHz mono
- output_dir/codec/*.wav     - NeuCodec-decoded audio at 48kHz mono

Usage:
    python build_enhancement_dataset.py --shard_dir ./shards --audio_root /path/to/audio --output_dir ./enhancement_dataset
"""

import argparse
import glob
import os

# TF32 on ROCm is disabled unless we do this.
os.environ["HIPBLASLT_ALLOW_TF32"] = "1"
# CUDAgraphs prevent us from using grad acc and torch.compile
os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

from pathlib import Path
from tqdm import tqdm

import torch
torch.backends.cuda.enable_cudagraph_trees = False
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


import torchaudio
from torchaudio.functional import resample

from neucodecfe import NeuCodecFE


def load_audio_file(audio_path: str) -> tuple[torch.Tensor, int]:
    """
    Load audio file and convert to mono.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        tuple: (waveform tensor of shape (1, T), sample_rate)
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform, sr


def extract_segment_from_loaded(waveform: torch.Tensor, sr: int, start: float, end: float, target_sr: int = 48000) -> torch.Tensor:
    """
    Extract segment from pre-loaded audio, resample to target_sr.
    
    Args:
        waveform: Pre-loaded audio tensor of shape (1, T)
        sr: Sample rate of the waveform
        start: Start time in seconds
        end: End time in seconds
        target_sr: Target sample rate (default: 48000)
    
    Returns:
        torch.Tensor: Audio tensor of shape (1, T) at target_sr
    """
    # Extract segment
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = waveform[:, start_sample:end_sample]
    
    # Resample to target sample rate
    if sr != target_sr:
        segment = resample(segment, sr, target_sr)
    
    return segment


def decode_tokens_to_audio(codec: NeuCodecFE, tokens: torch.Tensor, target_sr: int = 48000) -> torch.Tensor:
    """
    Decode NeuCodec tokens to audio and resample to target sample rate.
    
    Args:
        codec: NeuCodecFE instance
        tokens: Token tensor of shape (1, T_tokens) or (B, 1, T_tokens)
        target_sr: Target sample rate (default: 48000)
    
    Returns:
        torch.Tensor: Audio tensor of shape (1, T) at target_sr
    """
    CODEC_OUTPUT_SR = 24000
    
    # Decode tokens
    # tokens shape from dataset: (1, T_tokens), dtype int32
    decoded = codec.decode_codes(tokens)  # Output: (B, 1, T_audio) at 24kHz
    
    # Remove batch dim if present, ensure shape is (1, T)
    if decoded.dim() == 3:
        decoded = decoded.squeeze(0)  # (1, T_audio)
    
    # Resample to target sample rate
    if CODEC_OUTPUT_SR != target_sr:
        decoded = resample(decoded, CODEC_OUTPUT_SR, target_sr)
    
    return decoded.cpu()


def generate_filename(sample_idx: int, sample: dict) -> str:
    """
    Generate a unique filename for the sample.
    Uses the audio file stem + timestamp to ensure uniqueness.
    """
    audio_stem = Path(sample['audio_path']).stem
    start_ms = int(sample['start'] * 1000)
    end_ms = int(sample['end'] * 1000)
    return f"{audio_stem}_{start_ms}_{end_ms}.wav"


def main():
    parser = argparse.ArgumentParser(description="Build audio enhancement dataset from NeuCodec tokenized data")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing shard .pt files")
    parser.add_argument("--audio_root", type=str, required=True, help="Root directory containing source audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for decoding")
    args = parser.parse_args()
    
    # Create output directories
    original_dir = Path(args.output_dir) / "original"
    codec_dir = Path(args.output_dir) / "codec"
    original_dir.mkdir(parents=True, exist_ok=True)
    codec_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all shard files
    shard_pattern = str(Path(args.shard_dir) / "*.pt")
    shard_files = sorted(glob.glob(shard_pattern))
    
    if not shard_files:
        print(f"No .pt files found in {args.shard_dir}")
        return
    
    print(f"Found {len(shard_files)} shard files")
    
    # Preload all shards into memory
    print("\nPreloading all shards into memory...")
    all_samples = []
    failed_shards = []
    
    for shard_idx, shard_path in enumerate(shard_files):
        shard_name = Path(shard_path).name
        print(f"  [{shard_idx + 1}/{len(shard_files)}] Loading: {shard_name}", end=" ")
        
        try:
            data = torch.load(shard_path, weights_only=False)
            # Add shard name to each sample for error reporting
            for sample in data:
                sample['_shard'] = shard_name
            all_samples.extend(data)
            print(f"({len(data)} samples)")
        except Exception as e:
            print(f"FAILED: {e}")
            failed_shards.append(f"{shard_name}: {e}")
    
    print(f"\nTotal samples loaded: {len(all_samples):,}")
    if failed_shards:
        print(f"Failed shards: {len(failed_shards)}")
    
    # Sort globally by audio_path for maximum I/O efficiency
    print("Sorting samples by audio_path...")
    all_samples.sort(key=lambda x: x['audio_path'])
    print("Done sorting.")
    
    # Load the codec
    print("\nLoading NeuCodec...")
    is_cuda = args.device == "cuda"
    codec = NeuCodecFE(is_cuda=is_cuda, offset=0)
    
    # Stats
    success_count = 0
    error_count = 0
    all_errors = list(failed_shards)  # Start with shard load errors
    
    # Audio cache
    cached_audio_path = None
    cached_waveform = None
    cached_sr = None
    
    print(f"\nProcessing {len(all_samples):,} samples...")
    
    for idx, sample in enumerate(tqdm(all_samples, desc="Processing")):
        try:
            # Generate filename
            filename = generate_filename(idx, sample)
            original_path = original_dir / filename
            codec_path = codec_dir / filename
            
            # Skip if already processed
            if original_path.exists() and codec_path.exists():
                success_count += 1
                continue
            
            # Build full audio path
            audio_full_path = str(Path(args.audio_root) / sample['audio_path'])
            
            # Load audio only if different from cached
            if audio_full_path != cached_audio_path:
                cached_waveform, cached_sr = load_audio_file(audio_full_path)
                cached_audio_path = audio_full_path
            
            # Extract segment from cached audio
            original_audio = extract_segment_from_loaded(
                cached_waveform,
                cached_sr,
                sample['start'],
                sample['end'],
                target_sr=48000
            )
            
            # Decode tokens to get codec audio
            tokens = sample['tokens']
            codec_audio = decode_tokens_to_audio(codec, tokens, target_sr=48000)
            
            # Save both files
            torchaudio.save(str(original_path), original_audio, 48000)
            torchaudio.save(str(codec_path), codec_audio, 48000)
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            shard_name = sample.get('_shard', 'unknown')
            error_msg = f"{shard_name} sample {idx}: {str(e)}"
            all_errors.append(error_msg)
            if error_count <= 10:  # Only print first 10 errors
                tqdm.write(f"Error: {error_msg}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Total Success: {success_count:,}")
    print(f"  Total Errors:  {error_count:,}")
    print(f"  Output:        {args.output_dir}")
    print(f"{'='*50}")
    
    # Save error log if there were errors
    if all_errors:
        error_log_path = Path(args.output_dir) / "errors.log"
        with open(error_log_path, "w") as f:
            f.write("\n".join(all_errors))
        print(f"Error log saved to: {error_log_path}")


if __name__ == "__main__":
    main()
