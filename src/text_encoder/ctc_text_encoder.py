import re
from string import ascii_lowercase

import torch

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.use_beam = False

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decoding with CTC.

        Args:
            inds (list): list of tokens.
        Returns:
            text (str): text without empty tokens and repetitions.
        """

        text = self.merge_inds(inds)

        return self.decode(text)

    @staticmethod
    def merge_inds(inds: list) -> list:
        text = [inds[0]]
        current_char = inds[0]
        is_repeat = False
        for i in range(1, len(inds)):
            if inds[i] != current_char:
                if is_repeat:
                    is_repeat = False
                    current_char = inds[i]
                    text.append(current_char)
                else:
                    current_char = inds[i]
                    text.append(current_char)
            else:
                if not is_repeat:
                    is_repeat = True
        if current_char != inds[-1]:
            text.append(inds[-1])

        return text

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
