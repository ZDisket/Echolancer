# Echolancer Standalone Package

This is a standalone implementation of the Echolancer model that can be trained and used for inference independently of the larger codebase.

## Overview

Echolancer is a Transformer-based text-to-speech model with:
- Non-autoregressive text encoder
- Autoregressive spectrogram decoder
- Emotion conditioning support
- Multi-speaker support
- ALiBi (Attention with Linear Biases) support
- Dedicated FeedForward classes with sequence masking
- Optimized MultiHeadAttention with efficient linear projections
- Full audio generation pipeline testing

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
standalone/
├── model/
│   ├── __init__.py
│   ├── echolancer.py     # Main model definition
│   └── loss.py           # Loss functions
├── utils/
│   ├── __init__.py       # Utility functions
│   ├── config.py         # Configuration utilities
│   ├── logger.py         # Weights & Biases logger
│   ├── scheduler.py       # Learning rate schedulers
│   ├── early_stopping.py # Early stopping utilities
│   └── char_tokenizer.py # Character-level tokenizer
├── data/
│   └── __init__.py       # Data loading utilities
├── config/
│   ├── model.yaml        # Model configuration
│   └── train.yaml        # Training configuration
├── train.py # Fully featured training script
├── infer.py              # Inference script
├── bertfe.py             # BERT frontend for emotion encoding
├── istftnetfe.py         # ISTFTNet frontend for waveform generation
├── scripted_preencoder.py # Scripted pre-encoder for VQGAN decoding
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Usage

### Training

To train the model with the fully featured training script, run:
```bash
# Create dummy data (optional)
python train.py --model_config config/model.yaml --train_config config/train.yaml --train_data_dir ./data/train --val_data_dir ./data/val --create_dummy

# Train with existing data
python train.py --model_config config/model.yaml --train_config config/train.yaml --train_data_dir ./data/train --val_data_dir ./data/val
```

The fully featured training script includes:
1. Configuration management with YAML files
2. Weights & Biases integration for experiment tracking
3. Robust checkpointing with model and optimizer state
4. Validation loop during training
5. Early stopping based on validation loss
6. Learning rate scheduling
7. Gradient clipping and accumulation
8. Comprehensive logging of metrics and hyperparameters
9. ALiBi support for improved attention mechanisms
10. Configurable activation functions
11. Optimized attention mechanisms
12. Separate test interval for novel sentence evaluation

### Inference

To run inference with a trained model:
```bash
python infer.py --checkpoint path/to/checkpoint.pth
```

The inference script will:
1. Load a trained model from a checkpoint
2. Generate tokens from input text
3. Save the generated tokens and gate outputs

## Configuration

### Model Configuration (config/model.yaml)
Contains model architecture parameters:
- Vocabulary size
- Encoder/decoder dimensions
- Attention heads
- Layers
- Dropout rates
- ALiBi parameters (use_alibi, alibi_alpha)
- Activation function (activation)

### Training Configuration (config/train.yaml)
Contains training hyperparameters:
- Optimizer settings
- Learning rate scheduler
- Loss function parameters
- Batch size and training steps
- Weights & Biases settings
- Early stopping parameters
- Test interval parameters

## Data Format

The data loader expects .pt files in the specified data directories. Each .pt file should contain a dictionary with the following keys:

- `text`: Text token IDs (1D tensor)
- `mel`: Mel-spectrogram tokens (1D tensor)
- `speaker`: Speaker ID (int or tensor)
- `emotion`: Emotion embedding (1D tensor, optional)

Example of creating a .pt file:
```python
import torch

data = {
    'text': torch.randint(0, 100, (20,)),      # 20 text tokens
    'mel': torch.randint(0, 1010, (100,)),     # 100 mel tokens
    'speaker': 0,                              # Speaker ID
    'emotion': torch.randn(768)                # Emotion embedding
}

torch.save(data, 'sample_00001.pt')
```

## ALiBi (Attention with Linear Biases) Support

Echolancer now supports ALiBi (Attention with Linear Biases), which can improve training stability and performance. To enable ALiBi:

1. Set `use_alibi: True` in the model configuration
2. Optionally adjust `alibi_alpha` parameter (default: 1.0)

ALiBi parameters in `config/model.yaml`:
```yaml
model:
  use_alibi: True      # Enable ALiBi
  alibi_alpha: 1.0     # ALiBi alpha parameter
  # ... other parameters
```

## FeedForward Class with Sequence Masking

The feed forward components have been refactored into dedicated `FeedForward` classes with the following features:

1. **Dedicated Class**: Separate `FeedForward` class for better modularity
2. **Sequence Masking**: Support for masking padded positions with `.masked_fill`
3. **Configurable Activation**: Support for different activation functions (ReLU, GELU)
4. **Proper Integration**: Integrated into both encoder and decoder layers

FeedForward parameters in `config/model.yaml`:
```yaml
model:
  activation: "relu"   # or "gelu"
  # ... other parameters
```

## Optimized MultiHeadAttention

The MultiHeadAttention implementation has been optimized with efficient linear projections:

1. **Self-Attention**: Uses a single combined linear layer for Q, K, V projections
2. **Cross-Attention**: Uses separate linear layers for Q, K, V projections
3. **Memory Efficiency**: Reduces the number of linear operations where appropriate
4. **Performance**: Better memory locality and reduced kernel launch overhead

This optimization improves performance while maintaining mathematical equivalence to the standard implementation.

## Full Audio Generation Pipeline Testing

The training script includes a separate test interval that evaluates the model on completely novel sentences using the full audio generation pipeline:

1. **Echolancer Model**: Generates spectrogram tokens from text
2. **BERT Frontend**: Extracts emotion encoding from input text
3. **Scripted Pre-Encoder**: Decodes VQGAN indices to mel spectrograms
4. **ISTFTNet**: Converts mel spectrograms to waveforms

The test functionality:
- Runs at configurable intervals during training
- Tests on user-provided novel sentences
- Supports multiple speaker IDs
- Generates and logs audio samples to Weights & Biases
- Provides end-to-end evaluation of the complete TTS pipeline

To use the testing functionality, provide the required model components:
```bash
python train.py \
  --model_config config/model.yaml \
  --train_config config/train.yaml \
  --train_data_dir ./data/train \
  --val_data_dir ./data/val \
  --bert_model answerdotai/ModernBERT-base \
  --pre_encoder_dir ./pre_encoder \
  --istftnet_dir ./istftnet \
  --test_sentences "This is a test." "Another example." \
  --test_speakers 0 1
```

## Customization

### Model Configuration

You can modify the model configuration in `config/model.yaml`:
```yaml
model:
  vocab_size: 100
  mel_channels: 80
  emotion_channels: 256
  speaker_channels: 32
  use_alibi: True
  alibi_alpha: 1.0
  activation: "gelu"
  # ... other parameters
```

### Training Configuration

You can modify the training configuration in `config/train.yaml`:
```yaml
optimizer:
  type: "adamw"
  learning_rate: 0.0001
  # ... other parameters

training:
  batch_size: 16
  epochs: 100
  test_step: 2000  # Test every 2000 steps
  # ... other parameters
```

### Data

The data loader works with .pt files containing tensor data. To use your own data:
1. Convert your data to the expected format
2. Save each sample as a separate .pt file
3. Place all files in a directory
4. Point the training script to that directory

### Model Architecture

The model architecture can be customized by modifying the parameters in:
- `model/echolancer.py` - Main model definition
- `config/model.yaml` - Model configuration

## Key Components

### Echolancer Model
The main model class that combines:
- Text encoder (Transformer-based)
- Spectrogram decoder (Autoregressive Transformer)
- Emotion conditioning
- Speaker embedding (for multi-speaker models)
- ALiBi attention support
- Dedicated FeedForward classes with sequence masking
- Optimized MultiHeadAttention with efficient linear projections

### Loss Functions
- `EcholancerLoss`: Main loss function for the model
- `MaskedMAE`: Masked mean absolute error
- `MaskedBCE`: Masked binary cross-entropy

### Data Handling
- `EcholancerDataset`: Dataset class for loading .pt files
- `get_data_loader`: Function to create a DataLoader

### Pipeline Components
- `BERTFrontEnd`: Extracts emotion encoding from text
- `ScriptedPreEncoder`: Decodes VQGAN indices to spectrograms
- `ISTFTNetFE`: Converts spectrograms to waveforms

## Weights & Biases Integration

The fully featured training script integrates with Weights & Biases for:
- Experiment tracking
- Hyperparameter logging
- Metric visualization
- Model checkpointing as artifacts
- Audio sample logging during testing

To use Weights & Biases, you need to:
1. Install wandb: `pip install wandb`
2. Log in to your Weights & Biases account: `wandb login`
3. Configure your project in `config/train.yaml`

## License

This is a standalone version of the Echolancer model. The original model is part of a larger codebase with its own licensing terms.