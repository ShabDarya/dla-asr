import torchaudio
from torch import Tensor, nn


class Volume(nn.Module):
    def __init__(self, gain, *args, **kwargs):
        super().__init__()
        self.vol = torchaudio.transforms.Vol(gain=gain, gain_type="amplitude")
        self.sr = 16000

    def __call__(self, data: Tensor):
        return self.vol(data)
