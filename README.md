# Echolancer 

Echolancer is a multi-speaker, transformer decoder-only English TTS model. We use [NeuCodec](https://github.com/neuphonic/neucodec/tree/main) as the audio tokenizer.

We (me and my cat) release pretrained checkpoint and a demo notebook.

## 📦 Checkpoints

| Name | Params | Training Data | Speaker Control | Download | Demo |
|------|---------|---------------|-----------------|----------|-------|
| **Echolancer-ZS v0.1** | ~177M | Base+7k hours multi-speaker | ✔️ Zero-shot (ECAPA-TDNN) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-v0.1-zs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CNFDZwwvao9AoFdEKd_rwc45X8oLMkkw?usp=sharing) |
| Echolancer-Base v0.1 | ~177M | 30K+ hours multi-speaker | ❌ None (random) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-v0.1-base) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/118n_iDb3WZz3WPCKCDWnbdho1dp-mh65?usp=sharing) |


### 🔊 Inference

For inference code, please see the notebook for Echolancer-ZS v0.1

## Features

Marked with ❌ means not currently available but is on high priority.

- ✔️ Base model without speaker conditioning
- ✔️ Inference notebook
- ✔️ Zero-shot
- 🟡 LoRA finetuning (already capable - still need to write guide)
- ❌ Inference with KV cache
- ❌ ONNX export


## Fine-tuning
The base model can be finetuned to adapt it to a new voice (or multiple). You can either do full finetuning or LoRA. For LoRA, we recommend at least 1.5 hrs of audio; for full tuning, much more.

TODO: expand this

## License
This codebase and model weights are released under the MIT license; basically, do what you want.


## Contact
For any business/other formal inquiries, please e-mail nika109021@gmail.com
