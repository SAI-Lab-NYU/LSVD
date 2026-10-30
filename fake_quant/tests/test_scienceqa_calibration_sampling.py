from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

FAKE_QUANT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAKE_QUANT_ROOT))

from utils import data_utils
from utils import utils as runtime_utils


def test_default_sampling_preserves_historical_with_replacement_sequence() -> None:
    reference_rng = random.Random(19)
    expected = [reference_rng.randint(0, 99) for _ in range(40)]
    expected_next = reference_rng.random()

    actual = data_utils._scienceqa_calibration_indices(100, 40, 19)

    assert actual == expected
    assert random.random() == expected_next


def test_sampling_rejects_empty_dataset_or_negative_nsamples() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        data_utils._scienceqa_calibration_indices(0, 4, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        data_utils._scienceqa_calibration_indices(32, -1, 0)


def test_evaluator_override_uses_existing_boolean_optional_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["test-parser", "--hybrid_ffn_act_aware", "--no-hybrid_ffn_act_aware"],
    )
    monkeypatch.setattr(runtime_utils.os, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime_utils, "config_logging", lambda *_args, **_kwargs: None)
    args = runtime_utils.parser_gen()
    assert args.hybrid_ffn_act_aware is False
