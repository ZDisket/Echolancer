import torch
import torch.nn as nn
import torch.nn.functional as F
from model.echolancer import sequence_mask


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid: bid + 1],
                target_lengths=key_lens[bid: bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss



class MaskedCrossEntropy(nn.Module):
    """
    Cross-entropy over ONLY the audio slice of a shared (text+audio) vocab.
    Pass the text_vocab_size (V_text). Audio targets are assumed to be offset by V_text.
    """
    def __init__(self, text_vocab_size):
        super(MaskedCrossEntropy, self).__init__()
        self.V_text = text_vocab_size
        self.ignore_index = -100

    def forward(self, logits, targets, mask=None):
        """
        logits:  (B, T, V_total)
        targets: (B, T)  with audio IDs in [V_text, V_total), or -100 for ignore
        mask:    (B, T)  1.0 valid, 0.0 pad (optional)
        """
        B, T, V_total = logits.size()
        V_text = self.V_text
        V_audio = V_total - V_text

        # Slice logits to the audio range and shift targets down by V_text
        logits_audio = logits[:, :, V_text:]                      # (B, T, V_audio)
        targets_audio = torch.where(
            targets == self.ignore_index,
            targets,
            targets - V_text
        )                                                          # now in [0, V_audio)

        # Flatten
        logits_flat = logits_audio.reshape(-1, V_audio)
        targets_flat = targets_audio.reshape(-1)

        if mask is not None:
            m = mask.reshape(-1).bool()
            targets_flat = torch.where(
                m, targets_flat, torch.full_like(targets_flat, self.ignore_index)
            )
            loss = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=self.ignore_index,
                reduction='sum'
            )
            valid = m.sum().clamp(min=1)
            return loss / valid
        else:
            return F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=self.ignore_index,
                reduction='mean'
            )


class EcholancerLoss(nn.Module):
    """
    Loss function for the TTS model (Echolancer) - Discrete Token Prediction
    """

    def __init__(self, text_vocab_size):
        super(EcholancerLoss, self).__init__()
        self.token_loss_fn = MaskedCrossEntropy(text_vocab_size)
        self.forward_sum = ForwardSumLoss(blank_logprob=-8)
        self.attn_loss_start_step = 5000
        self.attn_loss_weight = 10.0
        self.ce_loss_weight = 1.5

    def forward(self, batch, model_out, current_step=0):
        """
        Compute loss for Echolancer model with discrete token prediction.
        
        Args:
            batch: Tuple containing (ids, raw_texts, speakers, texts, src_lens, mels, mel_lens, em_hidden)
            model_out: Tuple containing (text_mask, mel_mask, attn_logprob, x_mask_in, logits, indices_gt)
            current_step: Current step number (for scheduling)
            
        Returns:
            List of losses: [total_loss, dummy_attn_loss, token_loss]
        """
        # Extract model outputs
        text_mask, mel_mask, attn_logprob, x_mask_in, logits, indices_gt = model_out

        # Extract batch
        src_lens = batch['text_lens']  # (B,)
        out_lens = batch['audio_input_lens']  # (B,)

        # Extract targets
        token_targets = batch['audio_targets']
        audio_target_lens = batch['audio_target_lens']
        targets_mask = sequence_mask(token_targets.size(1), audio_target_lens) # True=padded
        
        # Create sequence mask for token loss (True for valid positions)
        seq_mask = ~targets_mask  # Invert: True for valid positions, False for padded

        # Compute token loss
        token_loss = self.token_loss_fn(logits, token_targets, seq_mask)

        attn_loss = torch.tensor(0.0, device=token_loss.device)


        if current_step > self.attn_loss_start_step and attn_logprob is not None:
            output_lengths = torch.clamp_max(out_lens, attn_logprob.size(2))
            input_lengths = torch.clamp_max(src_lens, attn_logprob.size(3))

            # attn_logprob = (B, N_layers, x, y)
            # need to feed (B, 1, x, y) to forward sum
            for d in range(0, attn_logprob.size(1)):
                curr_logprob = attn_logprob[:, d].unsqueeze(1)
                # (attn_logprob: (B, 1, audio_len, text_len))
                attn_loss += self.forward_sum(curr_logprob, in_lens=input_lengths, out_lens=output_lengths)

            attn_loss /= attn_logprob.size(1)

        # For compatibility with existing code that expects a list of losses
        total_loss = token_loss * self.ce_loss_weight + attn_loss * self.attn_loss_weight
        
        return [total_loss, attn_loss, token_loss]