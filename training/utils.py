import os
import random
import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:

    def __init__(
        self,
        patience: int = config.PATIENCE,
        mode: str = "max",
        checkpoint_path: str = None,
        verbose: bool = True,
    ):
        self.patience = patience
        self.mode     = mode
        self.path     = checkpoint_path
        self.verbose  = verbose
        self.counter  = 0
        self.best_score = None
        self.early_stop = False
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) if checkpoint_path else None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        if self._is_better(score):
            self.best_score = score
            self.counter = 0
            if self.path:
                torch.save(model.state_dict(), self.path)
                if self.verbose:
                    print(f"  checkpoint saved (score={score:.4f})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
