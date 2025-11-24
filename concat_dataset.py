import os
import re
import math
import random
import torch
from tqdm import tqdm
# ===== Dataset =====

class ConcatTextAudioSharded(torch.utils.data.Dataset):
    def __init__(
        self,
        shards_dir,
        text_tokenizer,
        seq_len,
        audio_bos_id,
        audio_eos_id,
        max_shard_cache=2,
        max_shards=None,
    ):
        self.shards_dir = os.path.abspath(shards_dir)
        self.text_tokenizer = text_tokenizer
        self.seq_len = int(seq_len) + 1
        self.audio_bos_id = int(audio_bos_id)
        self.audio_eos_id = int(audio_eos_id)
        self.max_shard_cache = int(max_shard_cache)
        self.audio_offset = len(self.text_tokenizer)

        if not os.path.isdir(self.shards_dir):
            raise ValueError("shards_dir not found: %s" % self.shards_dir)

        # Discover shards
        self.shard_files = sorted(
            [os.path.join(self.shards_dir, f) for f in os.listdir(self.shards_dir) if f.endswith(".pt")]
        )
        if max_shards is not None:
            self.shard_files = self.shard_files[:max_shards]

        if not self.shard_files:
            raise ValueError("No .pt shards found in %s" % self.shards_dir)

        # Parse sizes from filenames when available (e.g., shard_00012_n8192_tokXXXX.pt)
        self.shard_sizes = []
        n_re = re.compile(r"_n(\d+)(?:_|\.|$)")

        for p in tqdm(self.shard_files):
            m = n_re.search(os.path.basename(p))
            if m:
                self.shard_sizes.append(int(m.group(1)))
            else:
                shard = torch.load(p, map_location="cpu")
                self.shard_sizes.append(len(shard))
                del shard

        # Build cumulative starts for global indexing
        self.cum = [0]
        for n in tqdm(self.shard_sizes):
            self.cum.append(self.cum[-1] + n)
        self.total_utts = self.cum[-1]

        # Tiny per-worker LRU cache
        self._cache = {}
        self._order = []

    def __len__(self):
        # Expose total utterance count; the sampler will encode (shard_id, local_idx) into an int
        return self.total_utts

    def _encode_index(self, shard_id, local_idx):
        return (shard_id << 32) | int(local_idx)

    def _decode_index(self, encoded):
        shard_id = int(encoded) >> 32
        local_idx = int(encoded) & 0xFFFFFFFF
        return shard_id, local_idx

    def _load_shard(self, shard_id):
        if shard_id in self._cache:
            if shard_id in self._order:
                self._order.remove(shard_id)
            self._order.append(shard_id)
            return self._cache[shard_id]
        path = self.shard_files[shard_id]
        shard = torch.load(path, map_location="cpu")
        self._cache[shard_id] = shard
        self._order.append(shard_id)
        while len(self._order) > self.max_shard_cache:
            evict = self._order.pop(0)
            if evict in self._cache:
                del self._cache[evict]
        return shard

    def _encode_text(self, text):
        if torch.is_tensor(text):
            # If text is already a tensor, return it as-is (assuming it's already properly formatted with BOS/EOS)
            if not text.dtype == torch.long:
                text = text.to(dtype=torch.long)
            return text
        else:
            # If text is a string, tokenize it
            ids = self.text_tokenizer.encode(text, add_bos=True, add_eos=True)
            if not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, dtype=torch.long)
            else:
                ids = ids.to(dtype=torch.long)
            return ids

    def _encode_audio(self, codes_tensor):
        if not torch.is_tensor(codes_tensor):
            codes = torch.as_tensor(codes_tensor, dtype=torch.long)
        else:
            codes = codes_tensor.to(dtype=torch.long)
        codes = codes + self.audio_offset
        bos = torch.tensor([self.audio_bos_id], dtype=torch.long)
        eos = torch.tensor([self.audio_eos_id], dtype=torch.long)
        return torch.cat([bos, codes, eos], dim=0)

    def __getitem__(self, index):
        # index is a composite from the shard-aware sampler: encodes (shard_id, local_idx)
        shard_id, local_idx = self._decode_index(index)

        shard = self._load_shard(shard_id)
        shard_len = self.shard_sizes[shard_id]

        out = []
        total = 0
        cur = local_idx

        # Build a fixed-length seq by concatenating text[BOS/EOS] + audio[BOS ... EOS]
        # Stay within the same shard; wrap inside the shard if needed to preserve IO locality
        while total < self.seq_len:
            e = shard[cur]
            txt = self._encode_text(e["text"])
            aud = self._encode_audio(e["codes"])
            pair = torch.cat([txt, aud], dim=0)

            remaining = self.seq_len - total
            if pair.numel() <= remaining:
                out.append(pair)
                total += pair.numel()
                cur += 1
                if cur >= shard_len:
                    cur = 0  # wrap within shard
            else:
                out.append(pair[:remaining])
                total += remaining
                break

        seq = torch.cat(out, dim=0).to(dtype=torch.long)  # (seq_len,)
        x_in = seq[:-1].contiguous()  # (seq_len-1)
        x_tg = seq[1:].contiguous()  # (seq_len-1,)
        return x_in, x_tg

def collate_stack(batch):
    # batch is list of tuples (x_in, x_tg)
    inputs = torch.stack([b[0] for b in batch], dim=0)        # (B, seq_len)
    targets = torch.stack([b[1] for b in batch], dim=0)       # (B, seq_len-1)
    return inputs, targets

# ===== Shard-aware random sampler =====

class ShardAwareRandomSampler(torch.utils.data.Sampler):
    """
    Yields indices grouped by shard blocks to preserve IO locality.
    Each block sticks to one shard for many samples, but the order of blocks is randomized.
    Within a shard, utterance starts are randomly permuted.

    block_utterances: how many samples to draw from a shard before moving to another shard
                      (set high enough to amortize a 250MB load; e.g., thousands)
    """

    def __init__(self, dataset, block_utterances=4096, seed=1234, drop_last=False):
        if not isinstance(dataset, ConcatTextAudioSharded):
            raise ValueError("ShardAwareRandomSampler expects ConcatTextAudioSharded dataset")
        self.ds = dataset
        self.block_utterances = int(block_utterances)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        # Precompute per-shard local index permutations
        g = random.Random(self.seed)
        self.per_shard_perm = []
        for shard_id, size in enumerate(self.ds.shard_sizes):
            order = list(range(size))
            g.shuffle(order)
            self.per_shard_perm.append(order)

        # Split each shard's perm into blocks
        self.blocks = []  # list of (shard_id, start_offset, block_len)
        for shard_id, order in enumerate(self.per_shard_perm):
            n = len(order)
            for i in range(0, n, self.block_utterances):
                blen = min(self.block_utterances, n - i)
                if self.drop_last and blen < self.block_utterances:
                    break
                self.blocks.append((shard_id, i, blen))

        # Shuffle block order globally
        g.shuffle(self.blocks)

    def __iter__(self):
        # Emit composite encoded indices block-by-block
        for shard_id, start, blen in self.blocks:
            order = self.per_shard_perm[shard_id]
            for j in range(start, start + blen):
                local_idx = order[j]
                yield self.ds._encode_index(shard_id, local_idx)

    def __len__(self):
        if self.drop_last:
            # only full blocks count
            full = 0
            for shard_id, order in enumerate(self.per_shard_perm):
                full += len(order) // self.block_utterances * self.block_utterances
            return full
        return self.ds.total_utts


class DistributedShardAwareSampler(torch.utils.data.Sampler):
    """
    Distributed version of ShardAwareRandomSampler.
    Ensures that each rank gets a subset of the blocks, preserving IO locality
    while distributing the workload.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 seed=0, drop_last=False, block_utterances=4096):
        if not isinstance(dataset, ConcatTextAudioSharded):
            raise ValueError("DistributedShardAwareSampler expects ConcatTextAudioSharded dataset")
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.block_utterances = int(block_utterances)

        if self.num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.num_replicas = torch.distributed.get_world_size()
        
        if self.rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.rank = torch.distributed.get_rank()
            
        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(self.rank, self.num_replicas - 1))
                
        # Precompute per-shard local index permutations (same for all ranks initially)
        # We use a fixed seed here so all ranks agree on the inner-shard permutation
        # The randomness across epochs comes from shuffling the BLOCKS, not the inner items
        g = random.Random(self.seed) 
        self.per_shard_perm = []
        for shard_id, size in enumerate(self.dataset.shard_sizes):
            order = list(range(size))
            g.shuffle(order)
            self.per_shard_perm.append(order)

        # Split each shard's perm into blocks
        self.blocks = []  # list of (shard_id, start_offset, block_len)
        for shard_id, order in enumerate(self.per_shard_perm):
            n = len(order)
            for i in range(0, n, self.block_utterances):
                blen = min(self.block_utterances, n - i)
                if self.drop_last and blen < self.block_utterances:
                    break
                self.blocks.append((shard_id, i, blen))
                
        # Calculate total size for this rank
        # We need to know how many blocks this rank will get
        self.num_samples = 0
        # This is an approximation since we don't know exactly which blocks we'll get until iter
        # But for __len__ we usually need a constant. 
        # Standard DistributedSampler strategy:
        # If drop_last is True, we drop the tail blocks that don't fit evenly
        # If drop_last is False, we might pad (or just have uneven sizes)
        
        # Total blocks available
        total_blocks = len(self.blocks)
        
        if self.drop_last:
            self.num_blocks_per_rank = total_blocks // self.num_replicas
        else:
            self.num_blocks_per_rank = math.ceil(total_blocks / self.num_replicas)
            
        self.total_size = self.num_blocks_per_rank * self.num_replicas
        
        # Note: The actual number of samples yielded might vary slightly because blocks 
        # can have different sizes (the last block of a shard). 
        # However, PyTorch samplers usually expect __len__ to return number of indices.
        # Since we yield indices one by one, we can't easily predict exact sample count 
        # without simulating the shuffle.
        # For progress bars, an estimate is usually fine.

    def __iter__(self):
        # Deterministically shuffle blocks based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = torch.randperm(len(self.blocks), generator=g).tolist()
        
        # Add extra blocks to make it evenly divisible across ranks
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]
            
        assert len(indices) == self.total_size
        
        # Subsample blocks for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_blocks_per_rank
        
        # Collect all sample indices from assigned blocks
        all_samples = []
        for block_idx in indices:
            shard_id, start, blen = self.blocks[block_idx]
            order = self.per_shard_perm[shard_id]
            
            for j in range(start, start + blen):
                local_idx = order[j]
                all_samples.append(self.dataset._encode_index(shard_id, local_idx))
        
        # CRITICAL: Ensure all ranks yield the same number of samples
        # Compute the target number of samples per rank
        if not self.drop_last:
            # Pad to make all ranks have the same length
            num_samples = math.ceil(self.dataset.total_utts / self.num_replicas)
        else:
            # Drop to make all ranks have the same length
            num_samples = self.dataset.total_utts // self.num_replicas
        
        # Pad or truncate to exactly num_samples
        if len(all_samples) < num_samples:
            # Pad by repeating samples
            padding_size = num_samples - len(all_samples)
            if padding_size <= len(all_samples):
                all_samples += all_samples[:padding_size]
            else:
                all_samples += (all_samples * math.ceil(padding_size / len(all_samples)))[:padding_size]
        elif len(all_samples) > num_samples:
            # Truncate
            all_samples = all_samples[:num_samples]
        
        assert len(all_samples) == num_samples
        
        # Yield samples
        for sample_idx in all_samples:
            yield sample_idx

    def __len__(self):
        # Return the exact number of samples this rank will yield
        if not self.drop_last:
            return math.ceil(self.dataset.total_utts / self.num_replicas)
        else:
            return self.dataset.total_utts // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

# ===== Example usage =====
# from torch.utils.data import DataLoader
# ds = ConcatTextAudioSharded(
#     shards_dir="/path/to/pt_shards",
#     text_tokenizer=CharTokenizer(),
#     seq_len=2048,
#     audio_bos_id=65000,
#     audio_eos_id=65001,
#     max_shard_cache=2,
# )
# sampler = ShardAwareRandomSampler(ds, block_utterances=8192, seed=42)
# loader = DataLoader(
#     ds,
#     batch_size=8,
#     sampler=sampler,           # use sampler instead of shuffle=True
#     num_workers=4,
#     pin_memory=True,
#     persistent_workers=True,
#     prefetch_factor=4,
#     collate_fn=collate_stack,
# )
# for x in loader:
#     # x: (B, seq_len) long
#     pass
