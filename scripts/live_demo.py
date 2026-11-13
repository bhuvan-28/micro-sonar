"""
scripts/live_demo.py
=====================
Live Interactive Micro-Sonar Demo (Laptop)

Press ENTER to trigger a sonar sweep. The laptop emits a near-ultrasonic
chirp, records the echo, runs matched filtering, and optionally runs
CNN inference to predict room dimensions + material.

Usage:
    # Without trained model (shows just DSP analysis):
    python scripts/live_demo.py

    # With trained model (shows room predictions):
    python scripts/live_demo.py --model checkpoints/best_sonar_model.pt --backbone custom_cnn
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dsp.chirp import log_chirp
from dsp.feature_extractor import compute_mel_spectrogram
from dsp.matched_filter import compute_rt60, extract_rir_peaks, matched_filter

MATERIAL_NAMES = ["Concrete", "Drywall", "Wood", "Carpet", "Glass"]
SAMPLE_RATE = 48_000
CHIRP_F_START = 18_000
CHIRP_F_END = 22_000
CHIRP_DURATION = 0.1
RECORD_DURATION = 0.6
PRE_SILENCE = 0.05
AMPLITUDE = 0.8


def _build_emit_buffer(chirp: np.ndarray) -> np.ndarray:
    n_pre = int(PRE_SILENCE * SAMPLE_RATE)
    n_post = int((RECORD_DURATION - CHIRP_DURATION - PRE_SILENCE) * SAMPLE_RATE)
    return np.concatenate([
        np.zeros(n_pre, dtype=np.float32),
        chirp,
        np.zeros(n_post, dtype=np.float32),
    ])


def single_sweep(chirp: np.ndarray, emit_buffer: np.ndarray) -> np.ndarray:
    """Emit chirp and record echo. Returns the echo portion."""
    n_pre = int(PRE_SILENCE * SAMPLE_RATE)
    recorded = sd.playrec(
        emit_buffer[:, np.newaxis],
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    echo = recorded[n_pre:, 0]
    return echo


def run_inference(model, feature: np.ndarray, device) -> dict:
    """Run model inference on a single Mel-spectrogram feature."""
    with torch.no_grad():
        x = torch.from_numpy(feature).unsqueeze(0).to(device)
        pred_dims, pred_mat = model(x)
        dims = pred_dims[0].cpu().numpy()
        mat_idx = pred_mat[0].argmax().item()
        mat_conf = torch.softmax(pred_mat[0], dim=0)[mat_idx].item()
    return {
        "width": float(dims[0]),
        "length": float(dims[1]),
        "height": float(dims[2]),
        "material": MATERIAL_NAMES[mat_idx],
        "material_confidence": mat_conf,
    }


def run_live_demo(model_path: str | None = None, backbone: str = "custom_cnn") -> None:
    if not SD_AVAILABLE:
        print("[ERROR] sounddevice is not installed: pip install sounddevice")
        return

    # Load model if provided
    model = None
    device = None
    if model_path and TORCH_AVAILABLE:
        from models.sonar_net import build_model
        device = torch.device("cpu")
        model = build_model(backbone=backbone, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"[OK] Loaded model: {model_path}")
    elif model_path:
        print("[WARNING] PyTorch not available. Running DSP-only mode.")

    chirp = log_chirp(
        f_start=CHIRP_F_START, f_end=CHIRP_F_END,
        duration=CHIRP_DURATION, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE,
    )
    emit_buffer = _build_emit_buffer(chirp)

    print("\n╔══════════════════════════════════════════╗")
    print("║     MICRO-SONAR LIVE DEMO (Laptop)       ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Press ENTER to fire a sonar sweep       ║")
    print("║  Press Ctrl+C to exit                    ║")
    print("╚══════════════════════════════════════════╝\n")

    sweep_count = 0
    try:
        while True:
            input(">>> Press ENTER to trigger sweep...")
            sweep_count += 1
            t_start = time.perf_counter()

            echo = single_sweep(chirp, emit_buffer)
            rir = matched_filter(echo, chirp)
            peaks = extract_rir_peaks(rir, sample_rate=SAMPLE_RATE)
            rt60 = compute_rt60(rir, sample_rate=SAMPLE_RATE)
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            print(f"\n━━━ Sweep #{sweep_count} Results ━━━━━━━━━━━━━━━━━━")
            print(f"  Processing time : {elapsed_ms:.1f} ms")
            if not np.isnan(rt60):
                print(f"  RT60 (reverb)  : {rt60:.3f} s")
            print(f"  Echo peaks      : {len(peaks['times_ms'])} detected")
            for i, (t_ms, d_m) in enumerate(zip(peaks['times_ms'][:4], peaks['distances_m'][:4])):
                print(f"    Peak {i+1}: {t_ms:.1f} ms → ~{d_m:.2f} m away")

            if model is not None:
                try:
                    feature = compute_mel_spectrogram(rir, sample_rate=SAMPLE_RATE)
                    pred = run_inference(model, feature, device)
                    print(f"\n  ─── Room Prediction ───")
                    print(f"  Width    : {pred['width']:.2f} m")
                    print(f"  Length   : {pred['length']:.2f} m")
                    print(f"  Height   : {pred['height']:.2f} m")
                    print(f"  Material : {pred['material']} ({pred['material_confidence']*100:.1f}% conf)")
                    vol = pred['width'] * pred['length'] * pred['height']
                    print(f"  Volume   : {vol:.1f} m³")
                except Exception as e:
                    print(f"  [Inference error]: {e}")
            print()

    except KeyboardInterrupt:
        print("\n\nExiting Micro-Sonar demo. Goodbye!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--backbone", type=str, default="custom_cnn")
    args = parser.parse_args()
    run_live_demo(model_path=args.model, backbone=args.backbone)

