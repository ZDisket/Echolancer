import os
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import OrderedDict
from utils.char_tokenizer import CharTokenizer
from tqdm import tqdm
import random
def round_up(x, m):
    if m <= 1:
        return x
    r = x % m
    return x if r == 0 else x + (m - r)

def unroll_multicodebook(x):
    return x.t().reshape(-1)


class TTSPairedVQConcatDataset(Dataset):
    """
    Dataset for paired text -> VQ audio tokens with concatenated sequences.
    Expects a folder of shard files (e.g., .pt), where each shard is a list of dicts.
    Required keys in each dict:
      - "embedding": float speaker embedding, shape (192,)
      - "text":      utterance string
      - "tokens":    int tensor of shape (1, L)
    All shards are assumed to have the same number of items (N).
    """

    def __init__(
            self,
            data_dir,
            file_extension=".pt",
            preload=False,
            text_tokenizer=None,
            pad_id=0,
            max_cached_shards=2,
            max_shards=None,
            shard_type=1, # Shard type 1: / shard type 2: text, codes, speaker_id
            audio_bos_id=None,
            audio_eos_id=None,
    ):
        self.data_dir = data_dir
        self.file_extension = file_extension
        self.pad_id = pad_id
        self.text_tokenizer = text_tokenizer if text_tokenizer is not None else CharTokenizer()
        self.vocab_offset = self.text_tokenizer.get_vocab_size()
        self.audio_bos_id = audio_bos_id
        self.audio_eos_id = audio_eos_id
        print(f"Dataset vocab offset for VQ tokens: {self.vocab_offset}")


        pattern = os.path.join(data_dir, f"*{file_extension}")
        self.shard_paths = sorted(glob.glob(pattern))
        if max_shards is not None:
            self.shard_paths = self.shard_paths[:max_shards]

        if len(self.shard_paths) == 0:
            raise FileNotFoundError("No shard files found in '{}' with extension '{}'".format(data_dir, file_extension))

        # Peek first shard to get items per shard
        first = torch.load(self.shard_paths[0], map_location="cpu")
        if not isinstance(first, (list, tuple)):
            raise ValueError("Shard should be a list/tuple of dict entries, got: {}".format(type(first)))
        self.items_per_shard = len(first)
        if self.items_per_shard == 0:
            raise ValueError("Shard '{}' is empty.".format(self.shard_paths[0]))

        self.preload = preload
        self._entries = None
        self._shard_cache = OrderedDict()
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.shard_type = shard_type

        if preload:
            # Load everything to CPU RAM and pre-tokenize text
            all_entries = []
            text_too_short = 0
            tokens_too_short = 0

            for p in tqdm(self.shard_paths):
                shard = torch.load(p, map_location="cpu")
                if shard_type == 1:
                    for e in shard:
                        # Validate keys lightly
                        if "embedding" not in e or "text" not in e or "tokens" not in e:
                            continue
                        emb = torch.as_tensor(e["embedding"], dtype=torch.float32)  # (192,)
                        txt_ids = self.text_tokenizer.encode(e["text"])
                        txt_ids = torch.as_tensor(txt_ids, dtype=torch.long)  # (Lt,)
                        toks = e["tokens"]
                        if not torch.is_tensor(toks):
                            toks = torch.as_tensor(toks)
                        toks = toks.to(dtype=torch.long)  # (1, L)

                        if toks.size(-1) < 10:
                            tokens_too_short += 1
                            continue

                        if txt_ids.size(-1) < 5:
                            text_too_short += 1
                            continue

                        all_entries.append({
                            "embedding": emb,
                            "text_ids": txt_ids,
                            "tokens": toks,
                            "text": e["text"]
                        })

                elif shard_type == 2:
                     for e in shard:
                        # Validate keys
                        if "text" not in e or "codes" not in e or "speaker_id" not in e:
                            continue

                        # Text -> ids
                        txt_ids = self.text_tokenizer.encode(e["text"])
                        txt_ids = torch.as_tensor(txt_ids, dtype=torch.long)  # (Lt,)

                        # Audio codes -> long tensor, accept saved torch tensor or list/np
                        toks = e["codes"]
                        if not torch.is_tensor(toks):
                            toks = torch.as_tensor(toks)
                        toks = toks.to(dtype=torch.long)  # (Q, T) or (T,) or (1, T)

                        if toks.size(0) > 1: # unroll multi-codebook.
                            toks = unroll_multicodebook(toks) # (Q*T,)

                        # Normalize shape so last dim is time; if 1D, make (1, T)
                        if toks.dim() == 1:
                            toks = toks.unsqueeze(0)  # (1, T)

                        # Filters (mirror shard_type==1 logic)
                        if toks.size(-1) < 10:
                            tokens_too_short += 1
                            continue
                        if txt_ids.size(-1) < 5:
                            text_too_short += 1
                            continue

                        # Speaker id as tensor for batching consistency
                        speaker_id = torch.tensor(e["speaker_id"], dtype=torch.long)

                        all_entries.append({
                            "embedding": speaker_id,
                            "text_ids": txt_ids,
                            "tokens": toks,
                            "text": e["text"]
                        })

            self._entries = all_entries
            print(f"Finished preloading. Skipped because too short, text: {text_too_short} / audio: {tokens_too_short}")
            self.shuffle_order()

    def __len__(self):
        if self.preload:
            return len(self._entries)
        return len(self.shard_paths) * self.items_per_shard

    def shuffle_order(self):
        if self.preload:
            random.shuffle(self._entries)


    def _load_shard(self, shard_idx):
        key = self.shard_paths[shard_idx]
        if key in self._shard_cache:
            # move to end (recently used)
            self._shard_cache.move_to_end(key)
            return self._shard_cache[key]
        shard = torch.load(key, map_location="cpu")
        # Maintain tiny LRU cache
        self._shard_cache[key] = shard
        if len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return shard

    def _encode_audio(self, codes_tensor):
        if not torch.is_tensor(codes_tensor):
            codes = torch.as_tensor(codes_tensor, dtype=torch.long)
        else:
            codes = codes_tensor.to(dtype=torch.long)
        codes = codes.squeeze()
        codes = codes + self.vocab_offset
        if self.audio_bos_id is not None and self.audio_eos_id is not None:
            bos = torch.tensor([self.audio_bos_id], dtype=torch.long)
            eos = torch.tensor([self.audio_eos_id], dtype=torch.long)
            return torch.cat([bos, codes, eos], dim=0)
        else:
            return codes

    def __getitem__(self, idx):
        if self.preload:
            item = self._entries[idx]
            # Prepare concatenated sequence
            txt_ids = item["text_ids"]
            aud_ids = self._encode_audio(item["tokens"])
            seq = torch.cat([txt_ids, aud_ids], dim=0)  # Concatenate text and audio
            
            # Return the concatenated sequence
            x_in = seq[:-1].contiguous()  # (seq_len-1,)
            x_tg = seq[1:].contiguous()  # (seq_len-1,)
            
            return {
                "input_ids": x_in,
                "target_ids": x_tg,
                "text": item["text"],
                "embedding": item["embedding"],
            }

        # Lazy path: compute shard + within-shard indices
        shard_idx = idx // self.items_per_shard
        within = idx % self.items_per_shard
        shard = self._load_shard(shard_idx)
        e = shard[within]

        if "embedding" not in e or "text" not in e or "tokens" not in e:
            raise KeyError("Entry missing required keys at idx {} (shard {}, offset {})".format(idx, shard_idx, within))

        emb = torch.as_tensor(e["embedding"], dtype=torch.float32)  # (192,)
        txt_ids = self.text_tokenizer.encode(e["text"])
        txt_ids = torch.as_tensor(txt_ids, dtype=torch.long)  # (Lt,)
        aud_ids = self._encode_audio(e["tokens"])

        # Concatenate text and audio tokens
        seq = torch.cat([txt_ids, aud_ids], dim=0)  # (Lt + La,)
        
        # Create input and target sequences
        x_in = seq[:-1].contiguous()  # (seq_len-1,)
        x_tg = seq[1:].contiguous()  # (seq_len-1,)
        
        return {
            "input_ids": x_in,
            "target_ids": x_tg,
            "text": e["text"]
        }


def tts_paired_vq_concat_collate(batch, pad_id=0, shard_type=1):
    """
    Pads concatenated text-audio sequences to the max length in the batch.
    Returns a dict of:
      - input_ids:     (B, max_seq_len) long - padded input sequences
      - target_ids:    (B, max_seq_len) long - padded target sequences  
      - seq_lens:      (B,) long - actual sequence lengths before padding
    """
    B = len(batch)

    if shard_type == 1:
        # Speaker embeddings
        spk = torch.stack([b["embedding"] for b in batch], dim=0).to(dtype=torch.float32)  # (B,192)
    elif shard_type == 2:
        # Speaker IDs (scalars) -> (B,)
        spk = torch.stack([b["embedding"] for b in batch], dim=0).to(dtype=torch.long)
    else:
        raise ValueError(f"Invalid shard type {shard_type}, cannot fetch speaker embedding.")


    # Get the sequences and their lengths
    input_seqs = [b["input_ids"] for b in batch]
    target_seqs = [b["target_ids"] for b in batch]
    seq_lens = torch.as_tensor([seq.numel() for seq in input_seqs], dtype=torch.long)  # (B,)
    
    # Find max length in the batch
    max_len = int(seq_lens.max().item()) if B > 0 else 0
    if max_len == 0:
        max_len = 1

    # Pad input sequences
    input_padded = torch.full((B, max_len), fill_value=pad_id, dtype=torch.long)
    for i, inp in enumerate(input_seqs):
        inp_len = inp.numel()
        input_padded[i, :inp_len] = inp

    # Pad target sequences
    target_padded = torch.full((B, max_len), fill_value=pad_id, dtype=torch.long)
    for i, tgt in enumerate(target_seqs):
        tgt_len = tgt.numel()
        target_padded[i, :tgt_len] = tgt

    texts_raw = [b.get("text", "") for b in batch]

    return {
        "input_ids": input_padded,  # (B, max_seq_len) long
        "target_ids": target_padded,  # (B, max_seq_len) long
        "seq_lens": seq_lens,  # (B,) long
        "speakers": spk,  # (B,192) float32 (shard_type=1) or (B) long (shard_type=2)
    }