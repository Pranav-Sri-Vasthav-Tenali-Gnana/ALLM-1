"""Offline round trip: tiny random model -> INT8 -> save -> load -> predict."""
import numpy as np
import pytest
import torch
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

from core.models.factory import write_meta
from core.models.predictor import load_predictor
from core.quant.quantize import save_int8
from core.utils.common import dir_size_mb


@pytest.fixture
def fp32_dir(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(95)] + ["hello", "world"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))

    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(vocab), dim=256, hidden_dim=512, n_layers=2, n_heads=4,
        num_labels=2, id2label={0: "neg", 1: "pos"}, label2id={"neg": 0, "pos": 1},
    )
    out = tmp_path / "fp32"
    DistilBertForSequenceClassification(config).save_pretrained(out)
    tok.save_pretrained(out)
    write_meta(out, {
        "name": "t", "task": "classification", "variant": "fp32", "base_model": "tiny",
        "max_length": 32, "max_target_length": 16, "prefix": "", "label_names": ["neg", "pos"],
    })
    return out


def test_int8_roundtrip(fp32_dir):
    int8_dir = save_int8(fp32_dir, fp32_dir.parent / "int8")
    fp32, int8 = load_predictor(fp32_dir), load_predictor(int8_dir)
    assert int8.variant == "int8"

    # Linear weights are stored as int8 -> materially smaller on disk.
    assert dir_size_mb(int8_dir) < 0.6 * dir_size_mb(fp32_dir)

    texts = ["hello world", "w1 w2 w3 hello"]
    a, b = fp32.logits(texts), int8.logits(texts)
    assert np.allclose(a, b, atol=0.1), (a, b)
    assert [p["label"] for p in int8.predict(texts)] == [p["label"] for p in fp32.predict(texts)]
