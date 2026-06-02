"""Тесты формата сабмита: колонки, разделитель, округление, детерминизм байтов.

`make_submission` пишет с фиксированным CRLF и округлением до 6 знаков, чтобы байты
не зависели от ОС (это важно для побайтного воспроизведения рекорда в режиме
reproduce). `verify_submission` сверяет набор ключей со схемой commit.csv.
"""

import numpy as np
import pandas as pd
import pytest

from alfa_cred.inference import make_submission, verify_submission


def _keys() -> pd.DataFrame:
    return pd.DataFrame({"request_id": ["0", "0", "1"], "variant_no": [10, 2, 5]})


def test_format_columns_separator_rounding(tmp_path):
    out = tmp_path / "sub.csv"
    make_submission(_keys(), np.array([0.1234567, 0.7, 0.5]), out)
    df = pd.read_csv(out, sep=";")
    assert list(df.columns) == ["request_id", "variant_no", "score"]
    assert df["score"].tolist() == [0.123457, 0.7, 0.5]  # округление до 6 знаков


def test_crlf_line_endings(tmp_path):
    out = tmp_path / "sub.csv"
    make_submission(_keys(), np.array([0.1, 0.2, 0.3]), out)
    raw = out.read_bytes()
    assert raw.endswith(b"\r\n")     # фиксированный CRLF
    assert b"\r\r\n" not in raw      # без задвоения переводов строки


def test_byte_determinism(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    scores = np.array([0.111111, 0.222222, 0.333333])
    make_submission(_keys(), scores, a)
    make_submission(_keys(), scores, b)
    assert a.read_bytes() == b.read_bytes()


def test_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        make_submission(_keys(), np.array([0.1, np.nan, 0.3]), tmp_path / "x.csv")


def test_verify_accepts_matching_keys(tmp_path):
    out, sample = tmp_path / "sub.csv", tmp_path / "commit.csv"
    make_submission(_keys(), np.array([0.1, 0.2, 0.3]), out)
    make_submission(_keys(), np.array([0.0, 0.0, 0.0]), sample)
    verify_submission(out, sample)  # одинаковый набор ключей -> без исключения


def test_verify_detects_key_mismatch(tmp_path):
    out, sample = tmp_path / "sub.csv", tmp_path / "commit.csv"
    make_submission(_keys(), np.array([0.1, 0.2, 0.3]), out)
    other = pd.DataFrame({"request_id": ["0", "0"], "variant_no": [10, 2]})
    make_submission(other, np.array([0.0, 0.0]), sample)
    with pytest.raises(ValueError):
        verify_submission(out, sample)
