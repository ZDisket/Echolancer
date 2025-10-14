import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model=None):
        """
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Model to save weights from
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """
        Save model weights if it's the best so far.
        """
        if self.restore_best_weights and model is not None:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best_weights(self, model):
        """
        Load the best model weights.
        """
        if self.best_weights is not None and model is not None:
            model.load_state_dict(self.best_weights)