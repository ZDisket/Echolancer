
# Echolancer 

Echolancer is a multi-speaker, transformer decoder-only English TTS model. We use [NeuCodec](https://github.com/neuphonic/neucodec/tree/main) as the audio tokenizer.

We (me and my cat) release pretrained checkpoints, notebooks, **and a [technical report](docs/EcholancerTE-Repo.pdf)**


## 📦 Checkpoints

| Name | Params | Training Data | Speaker Control | Download | Demo |
|------|---------|---------------|-----------------|----------|-------|
| **Echolancer Stage 3 ZS** | ~1.3B| Base+7k hours multi-speaker | ✔️ Zero-shot (ECAPA-TDNN) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-stage3-zs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DzclkPa9mA73aGYbi4ELdweerYsG8w4l?usp=sharing) |
| Echolancer Stage 3 Base | ~1.3B | 30K+ hours multi-speaker | ❌ None (random) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-stage3-base) | N/A |
| Echolancer Stage 2 ZS | ~550M| Base+7k hours multi-speaker | ✔️ Zero-shot (ECAPA-TDNN) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-stage2-zs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_5zXoJIclFusVrs1j3jbNnUcSCRNJgNp?usp=sharing) |
| Echolancer Stage 2 Base | ~550M | 30K+ hours multi-speaker | ❌ None (random) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-stage2-base) | N/A |
| Echolancer Stage 1 ZS | ~177M | Base+7k hours multi-speaker | ✔️ Zero-shot (ECAPA-TDNN) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-v0.1-zs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CNFDZwwvao9AoFdEKd_rwc45X8oLMkkw?usp=sharing) |
| Echolancer Stage 1 Base | ~177M | 30K+ hours multi-speaker | ❌ None (random) | [HuggingFace](https://huggingface.co/ZDisket/echolancer-v0.1-base) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/118n_iDb3WZz3WPCKCDWnbdho1dp-mh65?usp=sharing) |


### 🔊 Inference

For inference code, please see the Colab demos

## Features

Marked with ❌ means not currently available but is on high priority.

- ✔️ Base model without speaker conditioning
- ✔️ Inference notebook
- ✔️ Zero-shot
- ✔️ Multi-GPU training
- 🟡 LoRA finetuning (already capable - still need to write guide)
- ❌ Inference with KV cache
- ❌ ONNX export


## Training & Fine-tuning
The base model can be finetuned to adapt it to a new voice (or multiple). You can either do full finetuning or LoRA. For LoRA, we recommend at least 10 minutes of audio; for full tuning, much more.

### Single GPU
```bash
python train.py --train_config config/train_config.yaml --model_config config/model_config.yaml --shards_dir /path/to/shards --out_dir output
```

### Multi-GPU (Single Node)
```bash
torchrun --nproc_per_node=NUM_GPUS train.py --train_config config/train_config.yaml --model_config config/model_config.yaml --shards_dir /path/to/shards --out_dir output
```

TODO: expand this

## License
This codebase and model weights are released under the MIT license; basically, do what you want.


## Contact
For any business/other formal inquiries, please e-mail nika109021@gmail.com
