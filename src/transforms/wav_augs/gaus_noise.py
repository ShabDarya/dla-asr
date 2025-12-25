import torch
from torch import Tensor, nn


class Gaus(nn.Module):
    def __init__(self, scale=0.5, p=0.5, *args, **kwargs):
        super().__init__()
        self.scale = scale
        self.p = p

    def __call__(self, data: Tensor):
        if torch.rand(1, device=data.device) > self.p:
            return data

        return data + torch.randn(data.size(), device=data.device) * self.scale
