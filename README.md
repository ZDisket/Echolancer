# Echolancer 
*(Speaker cloning not yet implemented — planned for v0.2)*

Echolancer is a multi-speaker, transformer decoder-only English TTS model. We use [NeuCodec](https://github.com/neuphonic/neucodec/tree/main) as the audio tokenizer.

We (me and my cat) release pretrained checkpoint and a demo notebook.

## 📦 Checkpoints

| Name | Params | Training Data | Speaker Control | Download | Notes |
|------|---------|---------------|-----------------|----------|-------|
| **Echolancer-Base v0.1** | ~177M | 30K+ hours multi-speaker | ❌ Not yet | [HuggingFace](https://huggingface.co/ZDisket/echolancer-v0.1-base) | Current release |


### 🔊 Colab Notebook

Run inference in your browser:          [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/118n_iDb3WZz3WPCKCDWnbdho1dp-mh65?usp=sharing)

## Features

Marked with ❌ means not currently available but is on high priority.

- ✔️ Base model without speaker conditioning
- ✔️ Inference notebook
- 🟡 LoRA finetuning (already capable - still need to write guide)
- ❌ Inference with KV cache
- ❌ ONNX export
- ❌ Zero-shot (soon)

## Fine-tuning
The base model can be finetuned to adapt it to a new voice (or multiple). You can either do full finetuning or LoRA. For LoRA, we recommend at least 1.5 hrs of audio; for full tuning, much more.

TODO: expand this

## License
This codebase and model weights are released under the MIT license; basically, do what you want.
