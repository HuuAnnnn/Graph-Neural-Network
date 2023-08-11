import torch
from torch import nn


class GAT(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(GAT, self).__init__(*args, **kwargs)
