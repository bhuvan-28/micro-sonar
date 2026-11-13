"""
dsp/feature_extractor.py
========================
Converts raw RIR signals into ML-ready feature tensors.

Feature pipeline:
  RIR (time domain) → Mel-Spectrogram (log-compressed) → (C, H, W) tensor

Also provides an energy-envelope feature for fast inference on device.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Construct a triangular Mel filterbank matrix.

    Parameters
    ----------
    n_fft : int
        Number of FFT bins (rfft bins = n_fft // 2 + 1).
    n_mels : int
        Number of Mel filter bands.
    sample_rate : int
        Audio sample rate.
    fmin : float
        Lowest frequency for the filterbank.
    fmax : float
        Highest frequency for the filterbank.

    Returns
    -------
    np.ndarray
        Filterbank matrix, shape (n_mels, n_fft // 2 + 1). Float32.
    """
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_freqs)

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return filterbank


def compute_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 48_000,
    n_fft: int = 4096,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 17_000,
    fmax: float = 23_000,
    target_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Compute a log-Mel spectrogram from an RIR or echo signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D audio/RIR signal. Float32.
    sample_rate : int
        Sample rate in Hz.
    n_fft : int
        FFT frame size.
    hop_length : int
        Hop between frames.
    n_mels : int
        Number of Mel bands.
    fmin, fmax : float
        Frequency range for Mel filterbank.
    target_size : (int, int)
        (H, W) to resize the final spectrogram to (bilinear).

    Returns
    -------
    np.ndarray
        Log-Mel spectrogram tensor, shape (1, H, W). Float32.
        Channel dimension is prepended for PyTorch compatibility.
    """
    signal = signal.astype(np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    fb = mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax)

    # Pad signal to fit at least one frame
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))

    frames = _frame_signal(signal, n_fft, hop_length)
    spectrogram = []
    for frame in frames:
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2  # power spectrum
        mel_frame = fb @ spectrum  # shape: (n_mels,)
        spectrogram.append(mel_frame)

    spec = np.array(spectrogram, dtype=np.float32).T  # (n_mels, n_frames)
    # Log compression
    spec = 10.0 * np.log10(spec + 1e-9)
    # Resize to target_size
    spec = _resize_2d(spec, target_size)
    # Normalise to [-1, 1]
    s_min, s_max = spec.min(), spec.max()
    if s_max > s_min:
        spec = 2 * (spec - s_min) / (s_max - s_min) - 1.0
    return spec[np.newaxis, :, :].astype(np.float32)  # (1, H, W)


def compute_energy_envelope(
    rir: np.ndarray,
    n_bands: int = 32,
    sample_rate: int = 48_000,
    fmin: float = 17_000,
    fmax: float = 23_000,
) -> np.ndarray:
    """
    Lightweight energy-envelope feature for fast mobile inference.
    
    Splits the signal into n_bands time segments and computes log-RMS energy
    in the [fmin, fmax] passband for each segment.

    Returns
    -------
    np.ndarray
        Shape (n_bands,). Float32.
    """
    # Bandpass: keep only frequency bins in [fmin, fmax]
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mask = (freqs >= fmin) & (freqs <= fmax)

    segments = np.array_split(rir.astype(np.float32), n_bands)
    envelope = []
    for seg in segments:
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spec = np.abs(np.fft.rfft(seg, n=n_fft))
        energy = np.mean(spec[mask] ** 2)
        envelope.append(10.0 * np.log10(energy + 1e-9))
    return np.array(envelope, dtype=np.float32)


def _frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Split signal into overlapping frames. Returns array (n_frames, frame_length)."""
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.stack(
        [signal[i * hop_length: i * hop_length + frame_length] for i in range(n_frames)]
    )
    return frames


def _resize_2d(arr: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear resize of a 2-D array to target (H, W) using numpy.
    Uses scipy if available for quality; falls back to nearest-neighbour.
    """
    try:
        from scipy.ndimage import zoom
        factors = (target[0] / arr.shape[0], target[1] / arr.shape[1])
        return zoom(arr, factors, order=1).astype(np.float32)
    except ImportError:
        # Nearest neighbour fallback
        rows = np.linspace(0, arr.shape[0] - 1, target[0]).astype(int)
        cols = np.linspace(0, arr.shape[1] - 1, target[1]).astype(int)
        return arr[np.ix_(rows, cols)]

