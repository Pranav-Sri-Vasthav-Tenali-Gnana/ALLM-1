import pytest

from core.utils.config import apply_overrides, load_config, parse_set_args

SMOKE = "configs/classification/smoke.yaml"


def test_load_smoke_config():
    cfg = load_config(SMOKE)
    assert cfg.task == "classification"
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert cfg.run_dir.as_posix() == "artifacts/smoke-cls"


def test_cli_overrides_and_float_coercion():
    cfg = load_config(SMOKE, parse_set_args(["train.epochs=1", "train.learning_rate=2e-5"]))
    assert cfg.train.epochs == 1
    assert isinstance(cfg.train.learning_rate, float)
    assert cfg.train.learning_rate == pytest.approx(2e-5)


def test_apply_overrides_returns_new_config():
    cfg = load_config(SMOKE)
    new = apply_overrides(cfg, {"train.batch_size": 32})
    assert new.train.batch_size == 32 and cfg.train.batch_size == 8


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(SMOKE, {"train.learning_rat": 1.0})
