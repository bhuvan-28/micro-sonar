"""
dsp/chirp.py
============
Chirp waveform generators for the Micro-Sonar engine.

Supports:
  - Linear frequency sweep (constant rate)
  - Logarithmic (exponential) frequency sweep (equal octave spacing)
  - Golay complementary sequence pair (best noise immunity)
  
All chirps are generated at a configurable sample rate (default 48 kHz)
and are Hanning-windowed to suppress spectral leakage at the edges.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Tuple


def _time_axis(duration: float, sample_rate: int) -> np.ndarray:
    """Return a time vector from 0 to duration (exclusive of endpoint)."""
    n_samples = int(duration * sample_rate)
    return np.linspace(0, duration, n_samples, endpoint=False)


def linear_chirp(
    f_start: float = 18_000,
    f_end: float = 22_000,
    duration: float = 0.1,
    sample_rate: int = 48_000,
    amplitude: float = 0.85,
    apply_window: bool = True,
) -> np.ndarray:
    """
    Generate a linear (constant sweep rate) sine chirp.

    Parameters
    ----------
    f_start : float
        Start frequency in Hz.
    f_end : float
        End frequency in Hz.
    duration : float
        Chirp duration in seconds.
    sample_rate : int
        Audio sample rate in Hz.
    amplitude : float
        Peak amplitude (0.0–1.0).
    apply_window : bool
        If True, apply a Hanning envelope to suppress edge clicks.

    Returns
    -------
    np.ndarray
        Float32 chirp waveform, shape (n_samples,).
    """
    t = _time_axis(duration, sample_rate)
    # Instantaneous phase: φ(t) = 2π [f_start·t + (f_end-f_start)/(2T)·t²]
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    chirp = amplitude * np.sin(phase)
    if apply_window:
        chirp *= np.hanning(len(t))
    return chirp.astype(np.float32)


def log_chirp(
    f_start: float = 18_000,
    f_end: float = 22_000,
    duration: float = 0.1,
    sample_rate: int = 48_000,
    amplitude: float = 0.85,
    apply_window: bool = True,
) -> np.ndarray:
    """
    Generate a logarithmic (exponential) sine sweep chirp.
    
    Log chirps have constant energy per octave band, which is better suited
    for acoustic room IR measurements.

    Parameters
    ----------
    (same as linear_chirp)

    Returns
    -------
    np.ndarray
        Float32 log chirp waveform, shape (n_samples,).
    """
    t = _time_axis(duration, sample_rate)
    # Instantaneous frequency: f(t) = f_start * (f_end/f_start)^(t/T)
    log_ratio = np.log(f_end / f_start)
    phase = 2 * np.pi * f_start * duration / log_ratio * (np.exp(t / duration * log_ratio) - 1)
    chirp = amplitude * np.sin(phase)
    if apply_window:
        chirp *= np.hanning(len(t))
    return chirp.astype(np.float32)


def golay_pair(
    n: int = 10,
    sample_rate: int = 48_000,
    f_center: float = 20_000,
    pulse_duration: float = 0.002,
    amplitude: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Golay complementary pair (a, b) modulated onto a carrier.

    Golay sequences a and b have the property that:
        autocorr(a) + autocorr(b) = 2n · δ[0]
    This eliminates time-sidelobe artefacts, achieving the best possible
    processing gain for a given sequence length without sidelobes.

    Parameters
    ----------
    n : int
        Sequence length is 2^n chips.
    sample_rate : int
        Audio sample rate in Hz.
    f_center : float
        Carrier frequency for BPSK modulation (Hz).
    pulse_duration : float
        Duration of each chip pulse in seconds.
    amplitude : float
        Peak amplitude.

    Returns
    -------
    (a, b) : Tuple[np.ndarray, np.ndarray]
        Two complementary float32 sequences.
    """
    length = 2 ** n
    # Recursive Golay pair construction
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    for _ in range(n):
        a_new = np.concatenate([a, b])
        b_new = np.concatenate([a, -b])
        a, b = a_new, b_new

    # Modulate chips onto carrier via BPSK
    chip_samples = int(pulse_duration * sample_rate)
    t_chip = np.linspace(0, pulse_duration, chip_samples, endpoint=False)
    carrier = np.cos(2 * np.pi * f_center * t_chip).astype(np.float32)

    def modulate(seq: np.ndarray) -> np.ndarray:
        return amplitude * np.concatenate([s * carrier for s in seq])

    return modulate(a), modulate(b)


def get_chirp(
    mode: Literal["linear", "logarithmic", "golay"] = "logarithmic",
    **kwargs,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Factory function: returns a chirp based on mode string from config.

    Parameters
    ----------
    mode : str
        'linear', 'logarithmic', or 'golay'.
    **kwargs :
        Passed to the underlying generator.

    Returns
    -------
    np.ndarray or tuple
        Single chirp array, or (a, b) pair for Golay.
    """
    if mode == "linear":
        return linear_chirp(**kwargs)
    elif mode == "logarithmic":
        return log_chirp(**kwargs)
    elif mode == "golay":
        return golay_pair(**kwargs)
    else:
        raise ValueError(f"Unknown chirp mode: '{mode}'. Choose linear/logarithmic/golay.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sr = 48_000
    chirp = log_chirp(sample_rate=sr)
    t = np.linspace(0, len(chirp) / sr, len(chirp))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(t * 1000, chirp)
    axes[0].set(title="Log Chirp Waveform (18–22 kHz)", xlabel="Time (ms)", ylabel="Amplitude")

    freqs = np.fft.rfftfreq(len(chirp), 1 / sr)
    mag = np.abs(np.fft.rfft(chirp))
    axes[1].plot(freqs / 1000, 20 * np.log10(mag + 1e-9))
    axes[1].set(title="Chirp Spectrum", xlabel="Frequency (kHz)", ylabel="Magnitude (dB)", xlim=[0, 25])
    axes[1].axvspan(18, 22, alpha=0.15, color="green", label="18–22 kHz band")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("chirp_analysis.png", dpi=150)
    print("Saved chirp_analysis.png")
    plt.show()

