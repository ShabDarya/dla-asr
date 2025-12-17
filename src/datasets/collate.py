import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch = {}

    audios = [x["audio"].squeeze(0) for x in dataset_items]
    audio_lens = torch.tensor([a.shape[0] for a in audios])
    batch["audio"] = pad_sequence(audios, batch_first=True)
    batch["audio_len"] = audio_lens

    specs = [x["spectrogram"].squeeze(0) for x in dataset_items]
    specs_T = [s.transpose(0, 1) for s in specs]
    spec_lens = torch.tensor([s.shape[0] for s in specs_T])
    specs_padded = pad_sequence(specs_T, batch_first=True)
    batch["spectrogram"] = specs_padded.transpose(1, 2)
    batch["spectrogram_length"] = spec_lens

    batch["text"] = [x["text"] for x in dataset_items]
    texts_encoded = [x["text_encoded"].squeeze(0) for x in dataset_items]
    text_lens = torch.tensor([t.shape[0] for t in texts_encoded])
    batch["text_encoded"] = pad_sequence(texts_encoded, batch_first=True)
    batch["text_encoded_length"] = text_lens

    batch["audio_path"] = torch.tensor([x["audio_path"] for x in dataset_items])

    return batch
