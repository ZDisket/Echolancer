#!/usr/bin/env python3
"""
Script to load all shards from a directory, tokenize the text using a text tokenizer, 
and save them to a new directory with text replaced by tokenized tensors.
"""

import os
import argparse
from tqdm import tqdm
import torch
from utils.char_tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize text in shards")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Input directory containing .pt shard files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory to save tokenized shards")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to saved tokenizer (optional, will use default CharTokenizer if not provided)")
    parser.add_argument("--add_bos", action="store_true", default=True,
                        help="Whether to add beginning-of-sequence token (default: True)")
    parser.add_argument("--add_eos", action="store_true", default=True,
                        help="Whether to add end-of-sequence token (default: True)")
    parser.add_argument("--no_add_bos", dest="add_bos", action="store_false",
                        help="Do not add beginning-of-sequence token")
    parser.add_argument("--no_add_eos", dest="add_eos", action="store_false",
                        help="Do not add end-of-sequence token")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        text_tokenizer = CharTokenizer.load(args.tokenizer_path)
        print(f"Loaded tokenizer from {args.tokenizer_path}")
    else:
        text_tokenizer = CharTokenizer()
        print("Initialized default CharTokenizer")

    # Find all .pt files in input directory
    shard_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pt')]
    if not shard_files:
        raise ValueError(f"No .pt files found in {args.input_dir}")

    print(f"Found {len(shard_files)} shard files to process")
    print(f"ADD BOS/EOS: {args.add_bos}/{args.add_eos}")

    # Process each shard with progress bar
    for filename in tqdm(shard_files, desc="Processing shards"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        # Load shard
        shard = torch.load(input_path, map_location="cpu")

        # Process each entry in the shard
        processed_shard = []
        for entry in tqdm(shard, desc=f"Processing {filename}", leave=False):
            # Tokenize text using the tokenizer
            text = entry["text"]
            tokenized_text = text_tokenizer.encode(text, add_bos=args.add_bos, add_eos=args.add_eos)
            tokenized_tensor = torch.tensor(tokenized_text, dtype=torch.long)

            # Preserve audio data and create new entry
            new_entry = {
                "text": tokenized_tensor,  # Replace text with tokenized tensor
                "codes": entry["codes"]    # Preserve audio codes as is
            }
            processed_shard.append(new_entry)

        # Save the processed shard to output directory
        torch.save(processed_shard, output_path)

    print(f"Successfully processed all shards. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()