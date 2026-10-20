"""Post-training quantization.

Two backends:

1. **PyTorch dynamic INT8** (classification + summarization).
   Every ``nn.Linear`` gets int8 weights; activations are quantized on the fly
   per batch. No calibration data needed, and for transformers the Linear
   layers hold the vast majority of weights/FLOPs, so this is where the
   size and CPU-latency wins come from.

2. **ONNX Runtime dynamic uint8** (classification only).
   Export FP32 -> ONNX graph, then quantize weights to uint8 with ORT. ORT's
   fused CPU kernels are usually faster than eager PyTorch.

Usage::

    python -m core.quant.quantize --config configs/classification/distilbert_agnews.yaml
"""
from __future__ import annotations

import argparse
import shutil
from collections import OrderedDict
from pathlib import Path

import torch

from core.models.factory import (
    auto_model_cls,
    build_empty_model,
    load_tokenizer,
    read_meta,
    write_meta,
)
from core.utils.common import get_logger
from core.utils.config import ExperimentConfig, add_config_args, config_from_args

log = get_logger(__name__)
INT8_WEIGHTS = "model_int8.pt"
ONNX_FP32 = "model.onnx"
ONNX_UINT8 = "model.uint8.onnx"


# ---------------------------------------------------------------- PyTorch INT8
def quantize_dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


QLinear = torch.ao.nn.quantized.dynamic.Linear


def export_int8_state(qmodel: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Flatten a dynamic-quantized model into plain tensors.

    Each quantized Linear is stored as int8 values + scale + zero point (+ bias).
    Plain tensors keep the file portable across torch versions and loadable
    with ``torch.load(weights_only=True)`` (no arbitrary pickle execution).
    """
    out = {k: v for k, v in qmodel.state_dict().items() if isinstance(v, torch.Tensor) and not v.is_quantized}
    for name, mod in qmodel.named_modules():
        if not isinstance(mod, QLinear):
            continue
        w, b = mod._weight_bias()
        p = f"{name}.qweight."
        out[p + "int8"] = w.int_repr()
        if w.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            out[p + "scale"] = torch.tensor(w.q_scale(), dtype=torch.float64)
            out[p + "zero_point"] = torch.tensor(w.q_zero_point())
        else:
            out[p + "scales"] = w.q_per_channel_scales()
            out[p + "zero_points"] = w.q_per_channel_zero_points()
            out[p + "axis"] = torch.tensor(w.q_per_channel_axis())
        if b is not None:
            out[p + "bias"] = b.detach()
    return out


def import_int8_state(qmodel: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Inverse of ``export_int8_state`` onto an already-quantized model shell."""
    state = dict(state)
    for name, mod in qmodel.named_modules():
        if not isinstance(mod, QLinear):
            continue
        p = f"{name}.qweight."
        q = state.pop(p + "int8")
        if p + "scale" in state:
            w = torch._make_per_tensor_quantized_tensor(
                q, float(state.pop(p + "scale")), int(state.pop(p + "zero_point"))
            )
        else:
            w = torch._make_per_channel_quantized_tensor(
                q, state.pop(p + "scales"), state.pop(p + "zero_points"), int(state.pop(p + "axis"))
            )
        # Rebuild the native packed-params entries the quantized Linear expects.
        state[f"{name}._packed_params.dtype"] = torch.qint8
        state[f"{name}._packed_params._packed_params"] = (w, state.pop(p + "bias", None))
    # Quantized modules branch on per-module version metadata; reuse the shell's.
    full = OrderedDict(state)
    full._metadata = qmodel.state_dict()._metadata
    qmodel.load_state_dict(full, strict=True)


def save_int8(fp32_dir: str | Path, out_dir: str | Path) -> Path:
    """Quantize the model in ``fp32_dir`` and save it as an int8 variant."""
    fp32_dir, out_dir = Path(fp32_dir), Path(out_dir)
    meta = read_meta(fp32_dir)
    model = auto_model_cls(meta["task"]).from_pretrained(str(fp32_dir))
    qmodel = quantize_dynamic_int8(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    # save_pretrained can't serialise quantized modules, so we store the int8
    # weights ourselves and keep config/tokenizer to rebuild the model shell.
    torch.save(export_int8_state(qmodel), out_dir / INT8_WEIGHTS)
    model.config.save_pretrained(out_dir)
    load_tokenizer(fp32_dir).save_pretrained(out_dir)
    if (fp32_dir / "generation_config.json").exists():
        shutil.copy(fp32_dir / "generation_config.json", out_dir)
    write_meta(out_dir, {**meta, "variant": "int8"})
    log.info("Saved INT8 model -> %s", out_dir)
    return out_dir


def load_int8(variant_dir: str | Path, task: str) -> torch.nn.Module:
    """Rebuild architecture -> apply same quantization -> load int8 weights."""
    variant_dir = Path(variant_dir)
    model = quantize_dynamic_int8(build_empty_model(task, variant_dir))
    state = torch.load(variant_dir / INT8_WEIGHTS, map_location="cpu", weights_only=True)
    import_int8_state(model, state)
    return model.eval()


# ------------------------------------------------------------ ONNX uint8 (cls)
def export_onnx_uint8(fp32_dir: str | Path, out_dir: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification

    fp32_dir, out_dir = Path(fp32_dir), Path(out_dir)
    meta = read_meta(fp32_dir)
    if meta["task"] != "classification":
        raise ValueError("ONNX export is only wired up for classification")

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(fp32_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(fp32_dir)).eval()
    model.config.return_dict = False

    dummy = tokenizer(["hello world"], return_tensors="pt")
    input_names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in dummy]
    dynamic = {k: {0: "batch", 1: "seq"} for k in input_names}
    dynamic["logits"] = {0: "batch"}

    class ByName(torch.nn.Module):
        """Map positional ONNX inputs to keyword args (forward() order varies by architecture)."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *tensors):
            return self.inner(**dict(zip(input_names, tensors)))[0]

    fp32_onnx = out_dir / ONNX_FP32
    torch.onnx.export(
        ByName(model),
        tuple(dummy[k] for k in input_names),
        str(fp32_onnx),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic,
        opset_version=17,
        dynamo=False,  # legacy TorchScript exporter: stable for HF encoders
    )
    quantize_dynamic(str(fp32_onnx), str(out_dir / ONNX_UINT8), weight_type=QuantType.QUInt8)
    fp32_onnx.unlink()  # keep only the deployable artifact

    tokenizer.save_pretrained(out_dir)
    write_meta(out_dir, {**meta, "variant": "onnx-uint8"})
    log.info("Saved ONNX uint8 model -> %s", out_dir)
    return out_dir


def quantize_run(cfg: ExperimentConfig) -> list[Path]:
    fp32_dir = cfg.variant_dir("fp32")
    outputs = [save_int8(fp32_dir, cfg.variant_dir("int8"))]
    if cfg.quant.onnx and cfg.task == "classification":
        outputs.append(export_onnx_uint8(fp32_dir, cfg.variant_dir("onnx-uint8")))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    quantize_run(config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
