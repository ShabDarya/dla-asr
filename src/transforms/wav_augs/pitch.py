import torch
from torch import Tensor, nn
from torch_audiomentations import PitchShift


class Pitch(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
