#!/usr/bin/env python3
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import argparse
from collections import Counter

import torch
import torchaudio
from torchaudio import transforms as T
from tqdm import tqdm

# NeuCodec (distilled) for audio -> VQ codes
from neucodec import DistillNeuCodec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert filelist (path|text or path|speaker_id|text) to VQ-token shards (.pt), sorted by duration asc."
    )
    parser.add_argument("--in_dir", required=True, help="Folder containing filelist.txt and audio files.")
    parser.add_argument("--out_dir", required=True, help="Output folder for .pt shards.")
    parser.add_argument("--filelist", default="filelist.txt", help="Filename of the filelist inside --in_dir.")
    parser.add_argument("--shard_size", type=int, default=8192, help="Items per shard (default: 8192).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for codec model.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory if present.")
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
    if codes.dim() < 3:
        codes = codes.squeeze(0).contiguous().cpu().long()
    else:
        codes = codes[0].contiguous().cpu().long()
    return codes


def save_shard(items, out_dir, shard_idx):
    shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
    torch.save(items, shard_path)
    return shard_path


def safe_duration_seconds(audio_path):
    # Format-agnostic duration via torchaudio.info; fallback to light load if needed
    try:
        info = torchaudio.info(audio_path)
        if info.num_frames is not None and info.sample_rate and info.sample_rate > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    try:
        wav, sr = torchaudio.load(audio_path)  # fallback (may decode)
        if wav.dim() == 2:
            return float(wav.size(-1)) / float(sr)
    except Exception:
        pass
    return None


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
    target_sr = 16000

    # First pass: read and measure durations (format-agnostic), collect items
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
                speaker_id = 0
            elif len(parts) == 3:
                # Multi-speaker format: path|speaker_id|text
                if is_multi_speaker is None:
                    is_multi_speaker = True
                    print("Detected multi-speaker format (path|speaker_id|text)")
                elif not is_multi_speaker:
                    raise RuntimeError(f"Mixed formats detected: expected single-speaker but got multi-speaker line: {s}")
                
                rel_path, speaker_id_str, text = parts
                try:
                    speaker_id = int(speaker_id_str.strip())
                except ValueError:
                    print(f"Warning: Invalid speaker_id '{speaker_id_str}' in line: {s}. Skipping.")
                    continue
            else:
                continue
            
            rel_path = rel_path.strip()
            text = text.strip()
            if not rel_path or text == "":
                continue
            full_path = rel_path if os.path.isabs(rel_path) else os.path.join(args.in_dir, rel_path)
            lines.append((full_path, speaker_id, text))

    durations = []
    print(f"Scanning durations for {len(lines)} item(s)...")
    for full_path, speaker_id, text in tqdm(lines, desc="Scanning", dynamic_ncols=True):
        d = safe_duration_seconds(full_path)
        if d is None:
            # keep but mark as large so it ends up at the end; also will still be attempted
            d = float("inf")
        durations.append((d, full_path, speaker_id, text))

    # Sort ascending by duration
    durations.sort(key=lambda x: x[0])

    # Second pass: encode sorted items, shard
    shard_items = []
    shard_idx = 0

    total_items = 0
    total_duration_sec = 0.0
    processed_per_speaker = Counter()
    failures = 0
    skipped = 0

    token_len_hist = Counter()
    token_len_sum = 0
    token_len_count = 0

    print(f"Encoding {len(durations)} item(s) (ascending by duration). Shard size = {args.shard_size}.")
    for d, audio_path, speaker_id, text in tqdm(durations, desc="Encoding", dynamic_ncols=True):
        if args.max_items and total_items >= args.max_items:
            break

        if audio_path is None or text is None or text == "":
            skipped += 1
            continue

        try:
            wav, sr = load_and_resample(audio_path, target_sr, device)
            codes = encode_to_codes(codec_model, wav)
        except Exception:
            failures += 1
            continue

        duration = wav.size(-1) / float(target_sr)
        total_duration_sec += float(duration)

        if codes.dim() == 2:
            tlen = int(codes.size(-1))
            token_len_hist[tlen] += 1
            token_len_sum += tlen
            token_len_count += 1
        elif codes.dim() == 1:
            tlen = int(codes.numel())
            token_len_hist[tlen] += 1
            token_len_sum += tlen
            token_len_count += 1

        item = {
            "text": text,
            "codes": codes,       # torch.LongTensor
            "speaker_id": speaker_id,
        }
        shard_items.append(item)
        processed_per_speaker[speaker_id] += 1
        total_items += 1

        if len(shard_items) >= args.shard_size:
            save_shard(shard_items, args.out_dir, shard_idx)
            shard_idx += 1
            shard_items = []

        if args.log_every and (total_items % args.log_every == 0):
            hrs = total_duration_sec / 3600.0
            tqdm.write(f"[{total_items} items] ~{hrs:.2f} h accumulated | failures={failures} skipped={skipped}")

    if shard_items:
        save_shard(shard_items, args.out_dir, shard_idx)
        shard_idx += 1
        shard_items = []

    total_hours = total_duration_sec / 3600.0
    print("\n=== Build Complete (sorted asc) ===")
    print(f"Total items processed: {total_items}")
    print(f"Total shards written:  {shard_idx}")
    print(f"Total duration (h):    {total_hours:.2f}")
    print(f"Failures:              {failures}")
    print(f"Skipped (bad lines):   {skipped}")
    if token_len_count > 0:
        avg_tokens = token_len_sum / float(token_len_count)
        print(f"Avg token length:      {avg_tokens:.2f} (on {token_len_count} items)")
        most_common = token_len_hist.most_common(10)
        mc_str = ", ".join([f"{k}:{v}" for k, v in most_common])
        print(f"Common token lengths:  {mc_str}")
    print("\nPer-speaker item counts (speaker_id: count):")
    for spk_id in sorted(processed_per_speaker.keys()):
        print(f"  {spk_id}: {processed_per_speaker[spk_id]}")

    print("\nShard pattern:         shard_00000.pt, shard_00001.pt, ...")


if __name__ == "__main__":
    main()
