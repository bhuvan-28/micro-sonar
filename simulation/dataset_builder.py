"""
simulation/dataset_builder.py
==============================
Batch synthetic RIR dataset generator.

Generates N synthetic rooms, computes their RIRs, convolves with the
sonar chirp, extracts Mel-spectrograms, and saves everything to a
compressed .npz file ready for PyTorch training.

Usage:
    python -m simulation.dataset_builder --n_rooms 500 --output data/dataset.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsp.chirp import log_chirp
from dsp.feature_extractor import compute_mel_spectrogram
from dsp.matched_filter import matched_filter
from simulation.room_sim import (
    MATERIAL_LIBRARY,
    RoomConfig,
    generate_random_room,
    simulate_room,
)


def build_dataset(
    n_rooms: int = 1000,
    sample_rate: int = 48_000,
    max_order: int = 3,
    snr_db_range: tuple = (15, 40),
    seed: int = 42,
    output_path: str = "data/dataset.npz",
) -> None:
    """
    Generate a synthetic dataset of (Mel-spectrogram, [W,L,H], material_label) tuples.

    Parameters
    ----------
    n_rooms : int
        Number of synthetic rooms to generate.
    sample_rate : int
        Audio sample rate.
    max_order : int
        Image-source reflection order.
    snr_db_range : (float, float)
        Random SNR range in dB for added white noise.
    seed : int
        Random seed for reproducibility.
    output_path : str
        Output .npz file path.
    """
    rng = np.random.default_rng(seed)
    chirp = log_chirp(sample_rate=sample_rate)

    features = []
    dimensions = []
    material_labels = []
    rt60s = []

    failures = 0
    pbar = tqdm(total=n_rooms, desc="Generating rooms", unit="room")

    generated = 0
    while generated < n_rooms:
        config = generate_random_room(rng=rng)
        try:
            rir = simulate_room(config, sample_rate=sample_rate, max_order=max_order)
        except Exception as e:
            failures += 1
            if failures > n_rooms // 5:
                print(f"\n[ERROR] Too many failures ({failures}). Aborting.")
                break
            continue

        # Convolve RIR with the near-ultrasonic chirp
        convolved_len = len(chirp) + len(rir) - 1
        convolved = np.convolve(chirp, rir).astype(np.float32)

        # Add white noise at random SNR
        snr_db = rng.uniform(*snr_db_range)
        signal_power = np.mean(convolved**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.standard_normal(len(convolved)).astype(np.float32) * np.sqrt(noise_power)
        noisy_signal = convolved + noise

        # Extract matched-filter RIR from noisy observation
        mf_rir = matched_filter(noisy_signal, chirp)

        # Extract Mel-spectrogram feature (1, 128, 128)
        try:
            mel = compute_mel_spectrogram(
                mf_rir,
                sample_rate=sample_rate,
                fmin=17_000,
                fmax=23_000,
            )
        except Exception:
            failures += 1
            continue

        features.append(mel)
        dimensions.append([config.width, config.length, config.height])
        material_labels.append(config.material_label)
        rt60s.append(0.0)  # placeholder; can compute via matched_filter.compute_rt60()
        generated += 1
        pbar.update(1)

    pbar.close()

    if not features:
        print("[ERROR] No data generated. Check pyroomacoustics installation.")
        return

    X = np.array(features, dtype=np.float32)             # (N, 1, 128, 128)
    Y_dims = np.array(dimensions, dtype=np.float32)      # (N, 3)
    Y_mat = np.array(material_labels, dtype=np.int64)    # (N,)
    Y_rt60 = np.array(rt60s, dtype=np.float32)           # (N,)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=X,
        dimensions=Y_dims,
        material_labels=Y_mat,
        rt60s=Y_rt60,
    )

    print(f"\nDataset saved → {output_path}")
    print(f"  Samples: {len(X)} (failures: {failures})")
    print(f"  Feature shape: {X.shape}")
    print(f"  Dimension range W: [{Y_dims[:,0].min():.2f}, {Y_dims[:,0].max():.2f}] m")
    print(f"  Dimension range L: [{Y_dims[:,1].min():.2f}, {Y_dims[:,1].max():.2f}] m")
    print(f"  Dimension range H: [{Y_dims[:,2].min():.2f}, {Y_dims[:,2].max():.2f}] m")
    print(f"  Material class counts: {np.bincount(Y_mat)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic RIR dataset builder")
    parser.add_argument("--n_rooms", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/dataset.npz")
    parser.add_argument("--max_order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        n_rooms=args.n_rooms,
        output_path=args.output,
        max_order=args.max_order,
        seed=args.seed,
    )

