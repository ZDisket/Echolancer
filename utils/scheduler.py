import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import math

class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup and then cosine decay.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))

class WarmupCosineLRSchedule:
    """
    Warmup + Cosine decay scheduler
    """
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_scheduler = LambdaLR(
            optimizer, 
            lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 1.0
        )
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps
        )
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.optimizer = optimizer

    def step(self):
        if self.step_count < self.warmup_steps:
            self.warmup_scheduler.step(self.step_count)
        else:
            self.cosine_scheduler.step(self.step_count - self.warmup_steps)
        self.step_count += 1

    def get_last_lr(self):
        if self.step_count < self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.cosine_scheduler.get_last_lr()

def get_scheduler(optimizer, scheduler_config, total_steps):
    """
    Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration
        total_steps: Total training steps
        
    Returns:
        Scheduler object
    """
    scheduler_type = scheduler_config.get('type', 'warmup_cosine')
    
    if scheduler_type == 'warmup_cosine':
        warmup_steps = scheduler_config.get('warmup_steps', 4000)
        return WarmupCosineLRSchedule(optimizer, warmup_steps, total_steps)
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    elif scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")