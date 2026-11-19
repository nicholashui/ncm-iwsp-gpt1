from pathlib import Path

import pytest
import yaml

from ncm_iwsp_gpt1.config import load_config


def _write(tmp_path: Path, updates: dict) -> Path:
    cfg = yaml.safe_load(Path("configs/gpt1_data.yaml").read_text(encoding="utf-8"))
    cfg.update(updates)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_rejects_the_gradient_model_name(tmp_path):
    for name in ("NCM-GPT1", "gpt1", "NCM-GPT1-smoke"):
        path = _write(tmp_path, {"name": name})
        with pytest.raises(ValueError, match="NCM-GPT1"):
            load_config(path)


def test_rejects_a_train_section(tmp_path):
    path = _write(tmp_path, {"train": {"optimizer": "adamw"}})
    with pytest.raises(ValueError, match="train"):
        load_config(path)
