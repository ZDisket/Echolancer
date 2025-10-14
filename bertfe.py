import torch

import io
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

class BERTFrontEnd():
  """
  A class for performing inference with a BERT model.

  This class handles loading a pre-trained BERT model and tokenizer,
  moving the model to a CUDA device if available, and performing
  inference on input text.
  """
  def __init__(self,is_cuda = False,model_name = "answerdotai/ModernBERT-base"):
    """
    Initializes the BERTFrontEnd.

    Args:
      is_cuda (bool): Whether to use CUDA for inference. Defaults to False.
      model_name (str): The name of the pre-trained BERT model to load.
        Defaults to "answerdotai/ModernBERT-base".
    """
    # Load the pre-trained BERT model and tokenizer
    self.model = AutoModel.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Store whether to use CUDA
    self.is_cuda = is_cuda
    
    # Move the model to the CUDA device if requested
    if is_cuda:
      self.model = self.model.cuda()

    print("Loaded BERT")
    

  def infer(self,in_txt):
    """
    Performs inference on a single text input.

    Args:
      in_txt (str): The input text string.

    Returns:
      tuple: A tuple containing:
        - encoded_layers (torch.Tensor): The hidden states, with shape [1, n_tokens, bert_size].
        - pooled (torch.Tensor): The pooled output, with shape [1, bert_size].
    """
    # Tokenize the input text
    inputs = self.tokenizer(in_txt, return_tensors="pt")
    
    # Move the input tensors to the CUDA device if requested
    if self.is_cuda:
      inputs["input_ids"] = inputs["input_ids"].cuda()
      inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
      inputs["attention_mask"] = inputs["attention_mask"].cuda()

    # Perform inference with no gradient calculation
    with torch.no_grad():
      # Get the encoded layers and pooled output from the model
      encoded_layers, pooled = self.model(**inputs, return_dict=False)
    
    return encoded_layers, pooled
