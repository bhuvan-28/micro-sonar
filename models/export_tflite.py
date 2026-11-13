"""
models/export_tflite.py
========================
Export trained PyTorch model → ONNX → TFLite (quantised, Android-ready).

Usage:
    python models/export_tflite.py \
        --checkpoint checkpoints/best_sonar_model.pt \
        --backbone custom_cnn \
        --output models/sonar_model.tflite

Requirements:
    pip install onnx onnxruntime tensorflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def export_onnx(model, onnx_path: str, input_shape=(1, 1, 128, 128)) -> None:
    import torch
    dummy = torch.randn(*input_shape)
    # Wrap to return a single concatenated tensor (ONNX friendly)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            dims, mat = self.inner(x)
            return dims, mat

    wrapper = ExportWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["mel_spectrogram"],
        output_names=["room_dims", "material_logits"],
        dynamic_axes={"mel_spectrogram": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX model exported → {onnx_path}")


def verify_onnx(onnx_path: str) -> None:
    import onnx
    import onnxruntime as ort
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 1, 128, 128).astype(np.float32)
    dims, mat = sess.run(None, {"mel_spectrogram": dummy})
    print(f"ONNX verified | dims: {dims.shape}, mat: {mat.shape}")


def export_tflite(onnx_path: str, tflite_path: str, quantize: bool = True) -> None:
    """Convert ONNX → TFLite via TensorFlow SavedModel intermediate."""
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare  # pip install onnx-tf
    except ImportError:
        print("[WARNING] onnx-tf not found. Trying direct tf2onnx path...")
        _export_tflite_via_tf2onnx(onnx_path, tflite_path, quantize)
        return

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = str(Path(tflite_path).with_suffix("")) + "_saved_model"
    tf_rep.export_graph(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    Path(tflite_path).write_bytes(tflite_model)
    size_kb = Path(tflite_path).stat().st_size / 1024
    print(f"TFLite model exported → {tflite_path} ({size_kb:.1f} KB)")


def _export_tflite_via_tf2onnx(onnx_path: str, tflite_path: str, quantize: bool) -> None:
    """Alternative: use tf2onnx (reverse direction) is not standard.
    For now, produce a note about the manual step needed."""
    print("[INFO] For TFLite conversion, use one of:")
    print("  Option A: pip install onnx-tf  (then rerun this script)")
    print("  Option B: Use ONNX Runtime for Android directly (no TFLite needed)")
    print("  Option C: Use ai.onnxruntime.mobile on Android")
    print(f"\n  Your ONNX model is at: {onnx_path}")
    print("  On Android, ONNX Runtime Mobile is simpler than TFLite for PyTorch models.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--backbone", default="custom_cnn", choices=["custom_cnn", "mobilenet_v3_small"])
    parser.add_argument("--output", default="models/sonar_model.tflite")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from models.sonar_net import build_model
    except ImportError:
        print("[ERROR] PyTorch required.")
        return

    device = torch.device("cpu")
    model = build_model(backbone=args.backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    onnx_path = str(Path(args.output).with_suffix(".onnx"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    export_onnx(model, onnx_path)
    verify_onnx(onnx_path)
    export_tflite(onnx_path, args.output, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()

