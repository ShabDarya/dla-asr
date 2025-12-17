import numpy as np


def calc_cer(target_text, predicted_text) -> float:
    m = len(target_text)  # row
    n = len(predicted_text)  # column

    if m == 0:
        return n

    d_arr = np.zeros((m + 1, n + 1), dtype=float)

    d_arr[0, :] = np.arange(n + 1, dtype=float)
    d_arr[:, 0] = np.arange(m + 1, dtype=float)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if target_text[i - 1] == predicted_text[j - 1]:
                d_arr[i, j] = d_arr[i - 1, j - 1]
            else:
                d_arr[i, j] = 1 + min(
                    d_arr[i - 1, j], d_arr[i, j - 1], d_arr[i - 1, j - 1]
                )

    return d_arr[m, n] / m


def calc_wer(target_text, predicted_text) -> float:
    target_text = target_text.split()
    predicted_text = predicted_text.split()

    return calc_cer(target_text, predicted_text)
