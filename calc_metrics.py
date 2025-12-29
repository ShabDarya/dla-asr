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

    pred_files = [str(p) for p in preds_path.rglob("*.txt") if p.is_file()]
    trans_files = [str(p) for p in trans_path.rglob("*.txt") if p.is_file()]

    cer = []
    wer = []

    for trans in trans_files:
        if one_file_one_text:
            with open(trans, encoding="utf-8") as f:
                trans_text = f.readline()
                trans_name = trans.split("\\")[-1].split(".")[0]

        else:
            with open(trans, encoding="utf-8") as f:
                for line in f:
                    trans_name, trans_text = line.split(" ", 1)

        pred_file = next(
            p
            for p in pred_files
            if p.endswith(f"/{trans_name}.txt") or p.endswith(f"\\{trans_name}.txt")
        )

        with open(pred_file) as p_f:
            pred_text = p_f.read()

        cer.append(calc_cer(trans_text.lower(), pred_text.lower()))
        wer.append(calc_wer(trans_text.lower(), pred_text.lower()))

    res = {"cer": sum(cer) / len(cer), "wer": sum(wer) / len(wer)}

    print(res)

    return res


if __name__ == "__main__":
    main()
