import torch.optim as optim
from typing import Dict, Any

def get_warmup_exp_decay(step: int, warmup_steps: int, total_steps: int):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 0.5 ** ((step - warmup_steps) / total_steps * 10)

def get_constant_lr(step: int, **kwargs):
    return 1.0

def get_scheduler(optimizer: optim.Optimizer, cfg: Dict[str, Any]):
    """Returns a scheduler based on config name."""
    name = cfg.get("scheduler_name", "warmup_exp")
    
    if name == "warmup_exp":
        lr_lambda = lambda step: get_warmup_exp_decay(
            step, 
            cfg.get("warmup_steps", 100), 
            cfg.get("total_steps", 1000)
        )
    elif name == "constant":
        lr_lambda = get_constant_lr
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
