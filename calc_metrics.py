from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.metrics.utils import calc_cer, calc_wer


@hydra.main(
    version_base=None, config_path="src/configs/metrics", config_name="calc_metrics"
)
def main(cfg: DictConfig):
    """
    script for calculate metrics with using paths.

    Args:
        config (DictConfig): hydra experiment config.
    """
    preds_path = Path(cfg.pred_path)
    trans_path = Path(cfg.trans_path)
    one_file_one_text = cfg.one_file_one_text

    pred_files_all = [str(p) for p in preds_path.rglob("*.txt") if p.is_file()]
    trans_files = [str(p) for p in trans_path.rglob("*.txt") if p.is_file()]

    cer = []
    wer = []

    for trans in trans_files:
        if one_file_one_text:
            with open(trans, encoding="utf-8") as f:
                trans_text = [f.readline().strip()]
                trans_name = [Path(trans).stem]

        else:
            with open(trans, encoding="utf-8") as f:
                names = []
                texts = []
                for line in f:
                    trans_name, trans_text = line.split(" ", 1)
                    trans_name = Path(trans_name).stem
                    texts.append(trans_text.strip())
                    names.append(trans_name)
                trans_text = texts
                trans_name = names
        trans_dict = dict(zip(trans_name, trans_text))

        pred_files = [
            (p, name)
            for p in pred_files_all
            for name in trans_name
            if Path(p).stem == name
        ]

        if pred_files is None:
            print(f"Нет файла с предсказаниями для - {trans_name}")
            continue

        for pred_file, tr_name in pred_files:
            with open(pred_file) as p_f:
                pred_text = p_f.read()

            cer.append(calc_cer(trans_dict[tr_name].lower(), pred_text.lower()))
            wer.append(calc_wer(trans_dict[tr_name].lower(), pred_text.lower()))

    res = {"cer": sum(cer) / len(cer), "wer": sum(wer) / len(wer)}

    print(res)

    return res


if __name__ == "__main__":
    main()
