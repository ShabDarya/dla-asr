import torch
from torch import nn
from torch.nn import Sequential

# comments with struct of model from article
# specAug
# ConvSubsamp
# Linear
# Dropout
# Conformer
# FFM
# MHSA
# LayerNorm
# MHA with RPE
# Dropout
# Conv
# FFM
# LayerNorm


class FFM(nn.Module):
    """
    Feed Forward Module for Conformer
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        """
        Args:
            dim (int): number of input features.
            hidden_dim (int): number of hidden features.
            dropout (float): probability of dropout.
        """
        super().__init__()

        self.net = Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Convolution Module for Conformer
    """

    def __init__(self, dim, kernel_size=31, padding=15, dropout=0.1):
        """
        Args:
            dim (int): number of input features.
            kernel_size (int): number of kernel size for depthwise convolution.
            padding (int): number of padding.
            dropout (float): probability of dropout.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)

        self.net = Sequential(
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x).transpose(1, 2)
        return self.net(x).transpose(1, 2)


class ConformerBlock(nn.Module):
    """
    Conformer Block for model
    """

    def __init__(self, dim, ff_hidden, num_heads=4, dropout=0.1):
        """
        Args:
            dim (int): number of input features.
            ff_hidden (int): number of hidden features.
            num_heads (int): number of head of conformer.
            dropout (float): probability of dropout.
        """
        super().__init__()

        self.ffm1 = FFM(dim, ff_hidden, dropout)

        self.layer_norm = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.conv = ConformerConvModule(dim, dropout=dropout)

        self.ffm2 = FFM(dim, ff_hidden, dropout)

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        x = x + self.ffm1(x) * 0.5

        res = x
        x = self.layer_norm(x)
        x, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + res

        x = x + self.conv(x)
        return self.final_norm(x + self.ffm2(x) * 0.5)


class ConformerModel(nn.Module):
    """
    Conformer Model
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        dim=128,
        fc_hidden=512,
        num_heads=4,
        num_layers=8,
        dropout=0.1,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            dim (int): number of hidden features in linear layear.
            fc_hidden (int): number of hidden features on conformer.
            num_heads (int): number of head of conformer.
            num_layers (int): number of layers of conformer blocks.
            dropout (float): probability of dropout.
        """

        super().__init__()

        self.input_linear = nn.Sequential(nn.Linear(n_feats, dim), nn.Dropout(dropout))

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=dim, ff_hidden=fc_hidden, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.output = nn.Linear(dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = self.input_linear(spectrogram)
        spectrogram_length = spectrogram_length.to(x.device)

        max_len = x.size(1)
        key_padding_mask = (
            torch.arange(max_len, device=x.device)[None, :]
            >= spectrogram_length[:, None]
        )

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        logits = self.output(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
