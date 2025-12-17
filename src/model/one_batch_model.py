import torch
from torch import nn
from torch.nn import Sequential

from src.model.baseline_model import BaselineModel


class OneBatch(BaselineModel):
    """
    Simple MLP
    """

    def __init__(self, n_feats, n_tokens, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__(n_feats, n_tokens, fc_hidden)

    def forward(self, text_encoded, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        batch_size, time = text_encoded.shape
        log_probs = torch.full(
            (batch_size, time, self.n_tokens), -float("inf"), device=text_encoded.device
        )
        for b in range(batch_size):
            for t in range(time):
                token = int(text_encoded[b, t].item())
                log_probs[b, t, token] = 0.0
        log_probs_length = torch.sum(text_encoded != 0, dim=1)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}
