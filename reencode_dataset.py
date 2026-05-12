#!/usr/bin/env python3
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
from collections import Counter

import torch
import torchaudio
from torchaudio import transforms as T
from tqdm import tqdm

# NeuCodec (distilled) for audio -> VQ codes -> audio
from neucodec import DistillNeuCodec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-encode dataset: load audio, encode with neural codec, decode back, and save as .wav with identical folder structure."
    )
    parser.add_argument("--in_dir", required=True, help="Folder containing filelist.txt and audio files.")
    parser.add_argument("--out_dir", required=True, help="Output folder for re-encoded .wav files (preserves folder structure).")
    parser.add_argument("--filelist", default="filelist.txt", help="Filename of the filelist inside --in_dir.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for codec model.")
    parser.add_argument("--output_sr", type=int, default=48000, help="Output sample rate for saved .wav files (default: 48000).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if present.")
    parser.add_argument("--max_items", type=int, default=0, help="For quick tests: cap total processed items (0 = all).")
    parser.add_argument("--log_every", type=int, default=1000, help="Print progress stats every N processed items.")
    return parser.parse_args()


def ensure_out_dir(path, overwrite):
    if os.path.isdir(path):
        if not overwrite:
            print("Output directory already exists. Use --overwrite to continue.")
            pass
    else:
        os.makedirs(path, exist_ok=True)


def load_codec_model(device):
    model = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec")
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model


def load_and_resample(audio_path, target_sr, device):
    wav, sr = torchaudio.load(audio_path)  # (C, T)
    if wav.dim() != 2:
        raise RuntimeError("Unexpected audio tensor shape.")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr
    wav = wav.unsqueeze(0).to(device)  # (B=1, 1, T)
    return wav, sr


def encode_to_codes(codec_model, wav):
    with torch.no_grad():
        codes = codec_model.encode_code(wav)  # Expect (B, Q, T_code) or similar
    return codes


def decode_from_codes(codec_model, codes):
    with torch.no_grad():
        wav = codec_model.decode_code(codes)  # Expect (B, 1, T) or similar
    return wav


def save_wav(wav_tensor, output_path, current_sr, target_sr):
    """Save a waveform tensor to a .wav file, resampling if needed."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Handle tensor shape: expect (B, C, T) or (C, T) or (T,)
    if wav_tensor.dim() == 3:
        wav_tensor = wav_tensor.squeeze(0)  # Remove batch dim -> (C, T)
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)  # Add channel dim -> (1, T)
    
    # Move to CPU if needed
    wav_tensor = wav_tensor.cpu()
    
    # Resample if needed
    if current_sr != target_sr:
        resampler = T.Resample(current_sr, target_sr)
        wav_tensor = resampler(wav_tensor)
    
    torchaudio.save(output_path, wav_tensor, target_sr)


def compute_relative_path(audio_path, in_dir):
    """Compute the relative path of audio_path with respect to in_dir."""
    # Normalize paths
    audio_path = os.path.normpath(os.path.abspath(audio_path))
    in_dir = os.path.normpath(os.path.abspath(in_dir))
    
    # Try to get relative path
    try:
        rel_path = os.path.relpath(audio_path, in_dir)
    except ValueError:
        # On Windows, paths on different drives can't have relative paths
        # In this case, use just the filename
        rel_path = os.path.basename(audio_path)
    
    return rel_path


def main():
    args = parse_args()
    ensure_out_dir(args.out_dir, args.overwrite)

    filelist_path = os.path.join(args.in_dir, args.filelist)
    if not os.path.isfile(filelist_path):
        raise RuntimeError(f"filelist not found: {filelist_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    codec_model = load_codec_model(device)
    codec_input_sr = 16000   # NeuCodec expects 16KHz input
    codec_output_sr = 24000  # NeuCodec outputs 24KHz after decoding
    output_sr = args.output_sr

    # Read filelist and collect audio paths
    # Detect multi-speaker format: path|speaker_id|text vs single-speaker: path|text
    lines = []
    is_multi_speaker = None
    
    with open(filelist_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if "|" not in s:
                continue
            
            parts = s.split("|", 2)
            if len(parts) == 2:
                # Single-speaker format: path|text
                if is_multi_speaker is None:
                    is_multi_speaker = False
                    print("Detected single-speaker format (path|text)")
                elif is_multi_speaker:
                    raise RuntimeError(f"Mixed formats detected: expected multi-speaker but got single-speaker line: {s}")
                
                rel_path, text = parts
            elif len(parts) == 3:
                # Multi-speaker format: path|speaker_id|text
                if is_multi_speaker is None:
                    is_multi_speaker = True
                    print("Detected multi-speaker format (path|speaker_id|text)")
                elif not is_multi_speaker:
                    raise RuntimeError(f"Mixed formats detected: expected single-speaker but got multi-speaker line: {s}")
                
                rel_path, _, text = parts
            else:
                continue
            
            rel_path = rel_path.strip()
            text = text.strip()
            if not rel_path or text == "":
                continue
            full_path = rel_path if os.path.isabs(rel_path) else os.path.join(args.in_dir, rel_path)
            lines.append((full_path, rel_path))

    # Process and re-encode all audio files
    total_items = 0
    total_duration_sec = 0.0
    failures = 0
    skipped = 0

    print(f"Re-encoding {len(lines)} audio file(s)...")
    for full_path, original_rel_path in tqdm(lines, desc="Re-encoding", dynamic_ncols=True):
        if args.max_items and total_items >= args.max_items:
            break

        if full_path is None:
            skipped += 1
            continue

        # Compute output path preserving folder structure
        rel_path = compute_relative_path(full_path, args.in_dir)
        
        # Change extension to .wav
        base, _ = os.path.splitext(rel_path)
        output_rel_path = base + ".wav"
        output_path = os.path.join(args.out_dir, output_rel_path)
        
        # Skip if output exists and not overwriting
        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            # Load and resample audio to codec input sample rate (16KHz)
            wav, sr = load_and_resample(full_path, codec_input_sr, device)
            
            # Encode to codes
            codes = encode_to_codes(codec_model, wav)
            
            # Decode back to audio (output is 24KHz)
            decoded_wav = decode_from_codes(codec_model, codes)
            
            # Save the decoded audio, resampling from 24KHz to output_sr
            save_wav(decoded_wav, output_path, codec_output_sr, output_sr)
            
            # Track duration (based on input audio at 16KHz)
            duration = wav.size(-1) / float(codec_input_sr)
            total_duration_sec += float(duration)
            
        except Exception as e:
            failures += 1
            if args.log_every and failures <= 10:
                tqdm.write(f"Failed to process {full_path}: {e}")
            continue

        total_items += 1

        if args.log_every and (total_items % args.log_every == 0):
            hrs = total_duration_sec / 3600.0
            tqdm.write(f"[{total_items} items] ~{hrs:.2f} h accumulated | failures={failures} skipped={skipped}")

    total_hours = total_duration_sec / 3600.0
    print("\n=== Re-encoding Complete ===")
    print(f"Total items processed: {total_items}")
    print(f"Total duration (h):    {total_hours:.2f}")
    print(f"Failures:              {failures}")
    print(f"Skipped:               {skipped}")
    print(f"\nOutput directory:      {args.out_dir}")


if __name__ == "__main__":
    main()
