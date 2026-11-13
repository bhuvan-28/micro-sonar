"""
dsp/matched_filter.py
=====================
Matched filtering and Room Impulse Response (RIR) extraction.

The matched filter is simply cross-correlation of the recorded echo signal
with the known transmitted reference chirp. In the frequency domain this is:

    H[f] = conj(CHIRP[f]) * ECHO[f]  (then IFFT)

This maximises the SNR for the echo detection (Wiener filter interpretation).
The result h[t] is the estimated Room Impulse Response (RIR).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def matched_filter(
    echo: np.ndarray,
    reference_chirp: np.ndarray,
) -> np.ndarray:
    """
    Compute the Room Impulse Response via frequency-domain matched filter.

    Parameters
    ----------
    echo : np.ndarray
        Recorded signal (may be longer than the chirp). Float32 or float64.
    reference_chirp : np.ndarray
        The transmitted chirp waveform (same dtype as echo).

    Returns
    -------
    np.ndarray
        Estimated RIR (same length as echo). Float32.

    Notes
    -----
    The FFT length is padded to the next power-of-2 beyond
    len(echo) + len(reference_chirp) - 1 to prevent circular aliasing.
    """
    n_fft = _next_pow2(len(echo) + len(reference_chirp) - 1)
    ECHO = np.fft.rfft(echo.astype(np.float64), n=n_fft)
    REF = np.fft.rfft(reference_chirp.astype(np.float64), n=n_fft)
    # Matched filter: multiply by complex conjugate of reference
    H = np.conj(REF) * ECHO
    rir = np.fft.irfft(H, n=n_fft)
    # Trim to echo length and normalise
    rir = rir[: len(echo)]
    peak = np.max(np.abs(rir))
    if peak > 1e-10:
        rir /= peak
    return rir.astype(np.float32)


def deconvolve_rir(
    echo: np.ndarray,
    reference_chirp: np.ndarray,
    regularisation: float = 1e-4,
) -> np.ndarray:
    """
    Compute the RIR via inverse filtering (spectral division with regularisation).

    More accurate than matched filter when the chirp bandwidth is not flat,
    but sensitive to noise if regularisation is too low.

    Parameters
    ----------
    echo : np.ndarray
        Recorded signal.
    reference_chirp : np.ndarray
        Transmitted chirp.
    regularisation : float
        Tikhonov regularisation constant (epsilon). Prevents division-by-zero
        in frequency bands where the chirp has near-zero energy.

    Returns
    -------
    np.ndarray
        Estimated RIR (same length as echo). Float32.
    """
    n_fft = _next_pow2(len(echo) + len(reference_chirp) - 1)
    ECHO = np.fft.rfft(echo.astype(np.float64), n=n_fft)
    REF = np.fft.rfft(reference_chirp.astype(np.float64), n=n_fft)
    ref_power = np.abs(REF) ** 2
    # Regularised inverse: H = conj(R) / (|R|² + ε)
    H = np.conj(REF) * ECHO / (ref_power + regularisation * ref_power.max())
    rir = np.fft.irfft(H, n=n_fft)
    rir = rir[: len(echo)]
    peak = np.max(np.abs(rir))
    if peak > 1e-10:
        rir /= peak
    return rir.astype(np.float32)


def extract_rir_peaks(
    rir: np.ndarray,
    sample_rate: int = 48_000,
    min_distance_ms: float = 2.0,
    prominence: float = 0.05,
    n_peaks: int = 10,
) -> dict:
    """
    Detect primary echo peaks in the RIR, corresponding to first-order reflections.

    Parameters
    ----------
    rir : np.ndarray
        Normalised Room Impulse Response.
    sample_rate : int
        Audio sample rate in Hz.
    min_distance_ms : float
        Minimum time separation between peaks (ms).
    prominence : float
        Minimum peak amplitude prominence (fraction of max).
    n_peaks : int
        Maximum number of peaks to return.

    Returns
    -------
    dict with keys:
        'times_ms'    : np.ndarray - peak times in milliseconds
        'distances_m' : np.ndarray - estimated reflection distances in metres
        'amplitudes'  : np.ndarray - peak amplitudes
        'indices'     : np.ndarray - sample indices
    """
    min_dist_samples = int(min_distance_ms * sample_rate / 1000)
    peaks, props = find_peaks(
        np.abs(rir),
        distance=min_dist_samples,
        prominence=prominence,
    )
    # Sort by amplitude descending, keep top n_peaks
    if len(peaks) > 0:
        order = np.argsort(np.abs(rir[peaks]))[::-1][:n_peaks]
        peaks = peaks[order]
        # Re-sort by time
        peaks = peaks[np.argsort(peaks)]

    times_ms = peaks / sample_rate * 1000.0
    # d = c·t/2 (round-trip), speed of sound ≈ 343 m/s
    distances_m = (peaks / sample_rate) * 343.0 / 2.0

    return {
        "times_ms": times_ms,
        "distances_m": distances_m,
        "amplitudes": np.abs(rir[peaks]) if len(peaks) > 0 else np.array([]),
        "indices": peaks,
    }


def compute_rt60(rir: np.ndarray, sample_rate: int = 48_000) -> float:
    """
    Estimate RT60 (reverberation time) from the RIR using the Schroeder backwards
    integration method.

    RT60 is the time it takes the sound energy to decay by 60 dB.
    We estimate it from the -5 to -35 dB slope (EDT extrapolated ×2).

    Parameters
    ----------
    rir : np.ndarray
        Normalised RIR.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    float
        Estimated RT60 in seconds.
    """
    # Schroeder energy decay curve
    energy = rir.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-20) + 1e-20)

    # Fit line between -5 dB and -35 dB (30 dB dynamic range → scale to 60 dB)
    try:
        i_5 = np.where(edc_db <= -5)[0][0]
        i_35 = np.where(edc_db <= -35)[0][0]
        t = np.arange(i_5, i_35) / sample_rate
        slope, _ = np.polyfit(t, edc_db[i_5:i_35], 1)
        rt60 = -60.0 / slope
    except (IndexError, np.linalg.LinAlgError):
        rt60 = float("nan")
    return rt60


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 ≥ n."""
    return 1 << (n - 1).bit_length()

