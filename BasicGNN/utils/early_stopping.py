import math
from typing import Any


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.min_val_loss = math.inf
        self.counter = 0
        self.is_stop = False

    def __call__(self, val_loss) -> Any:
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            if self.counter >= self.patience:
                self.is_stop = True
            else:
                self.counter += 1

        return self.is_stop
