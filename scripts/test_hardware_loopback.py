"""
scripts/test_hardware_loopback.py
==================================
Step 1: Laptop Hardware Validation Test

Emits a near-ultrasonic log chirp (18–22 kHz) through the laptop speakers
and simultaneously records via the laptop microphone. Then:

  1. Plots the emitted vs recorded waveforms
  2. Plots FFT spectrum of recording (checks hardware frequency response)
  3. Runs matched filter → shows extracted RIR peaks
  4. Estimates RT60 and reports SNR

Run this FIRST to verify your laptop hardware can handle 18-22 kHz.
Many consumer laptop speakers start rolling off above 16 kHz — this test
will show you the actual frequency cutoff and what SNR you're working with.

Usage:
    python scripts/test_hardware_loopback.py
    python scripts/test_hardware_loopback.py --list-devices
    python scripts/test_hardware_loopback.py --input-device 2 --output-device 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    print("[ERROR] sounddevice not installed. Run: pip install sounddevice")

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[WARNING] matplotlib not installed. Plots will be skipped.")

from dsp.chirp import log_chirp
from dsp.matched_filter import compute_rt60, deconvolve_rir, extract_rir_peaks, matched_filter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48_000
CHIRP_F_START = 18_000
CHIRP_F_END = 22_000
CHIRP_DURATION = 0.1          # seconds
RECORD_DURATION = 0.6         # seconds (chirp + echo tail)
PRE_SILENCE = 0.05            # seconds of silence before chirp
AMPLITUDE = 0.8


def list_audio_devices() -> None:
    """Print all available audio devices."""
    if not SD_AVAILABLE:
        return
    print("\n=== Available Audio Devices ===")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        in_ch = dev["max_input_channels"]
        out_ch = dev["max_output_channels"]
        print(f"  [{i:2d}] {dev['name']}")
        print(f"       IN: {in_ch} ch | OUT: {out_ch} ch | SR: {int(dev['default_samplerate'])} Hz")
    default = sd.query_devices(kind="input")
    print(f"\n  Default input  → {default['name']}")
    default = sd.query_devices(kind="output")
    print(f"  Default output → {default['name']}\n")


def run_loopback_test(
    input_device: int | None = None,
    output_device: int | None = None,
    n_sweeps: int = 3,
    save_recording: bool = True,
) -> dict:
    """
    Perform the full loopback chirp emission + echo recording test.

    Parameters
    ----------
    input_device : int | None
        sounddevice input device index (None = system default).
    output_device : int | None
        sounddevice output device index (None = system default).
    n_sweeps : int
        Number of chirp sweeps to average (reduces noise).
    save_recording : bool
        Save the raw recording to a WAV file.

    Returns
    -------
    dict with keys: chirp, recording, rir, peaks, rt60, snr_db
    """
    if not SD_AVAILABLE:
        print("[ERROR] sounddevice is required.")
        return {}

    # Build the emission buffer: pre-silence + chirp + post-silence (for echo capture)
    chirp = log_chirp(
        f_start=CHIRP_F_START,
        f_end=CHIRP_F_END,
        duration=CHIRP_DURATION,
        sample_rate=SAMPLE_RATE,
        amplitude=AMPLITUDE,
    )
    n_pre = int(PRE_SILENCE * SAMPLE_RATE)
    n_post = int((RECORD_DURATION - CHIRP_DURATION - PRE_SILENCE) * SAMPLE_RATE)
    emit_buffer = np.concatenate([
        np.zeros(n_pre, dtype=np.float32),
        chirp,
        np.zeros(n_post, dtype=np.float32),
    ])

    total_samples = len(emit_buffer)
    recordings = []

    print(f"\n=== Micro-Sonar Hardware Loopback Test ===")
    print(f"Sample Rate   : {SAMPLE_RATE} Hz")
    print(f"Chirp Range   : {CHIRP_F_START}–{CHIRP_F_END} Hz")
    print(f"Chirp Duration: {CHIRP_DURATION*1000:.0f} ms")
    print(f"Record Buffer : {RECORD_DURATION*1000:.0f} ms")
    print(f"Sweeps        : {n_sweeps}\n")

    device_pair = None
    if input_device is not None or output_device is not None:
        device_pair = (input_device, output_device)

    for sweep in range(n_sweeps):
        print(f"  Sweep {sweep+1}/{n_sweeps} — emitting chirp...", end=" ", flush=True)
        # Full-duplex: play and record simultaneously
        recorded = sd.playrec(
            emit_buffer[:, np.newaxis],   # (N, 1) mono
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device_pair,
            dtype="float32",
        )
        sd.wait()
        recordings.append(recorded[:, 0])
        print("done")
        time.sleep(0.3)  # brief pause between sweeps

    # Average across sweeps (coherent averaging suppresses incoherent noise)
    recording = np.mean(np.stack(recordings, axis=0), axis=0)

    # Trim recording to match emit buffer (may differ by 1 sample)
    recording = recording[:total_samples]

    # The echo is the portion AFTER the pre-silence
    echo = recording[n_pre:]

    # Matched filter → RIR
    rir = matched_filter(echo, chirp)
    deconv_rir = deconvolve_rir(echo, chirp)

    # Peak analysis
    peaks = extract_rir_peaks(rir, sample_rate=SAMPLE_RATE)

    # RT60 estimation
    rt60 = compute_rt60(rir, sample_rate=SAMPLE_RATE)

    # SNR estimate (signal power in chirp band vs noise power outside)
    # Measure in-band energy (18-22 kHz) vs adjacent ultrasonic noise floor (15-17 kHz & 23-24 kHz)
    freqs = np.fft.rfftfreq(len(recording), 1 / SAMPLE_RATE)
    spectrum = np.abs(np.fft.rfft(recording))
    band_mask = (freqs >= CHIRP_F_START) & (freqs <= CHIRP_F_END)
    signal_power = np.mean(spectrum[band_mask] ** 2)
    noise_mask = ((freqs >= 15_000) & (freqs < CHIRP_F_START)) | ((freqs > CHIRP_F_END) & (freqs <= 24_000))
    noise_power = np.mean(spectrum[noise_mask] ** 2) if np.any(noise_mask) else 1e-20
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-20))

    print(f"\n--- Results ---")
    print(f"RT60 (estimated) : {rt60:.3f} s" if not np.isnan(rt60) else "RT60             : N/A (decay too fast)")
    print(f"Band SNR         : {snr_db:.1f} dB")
    print(f"Echo Peaks found : {len(peaks['times_ms'])}")
    for i, (t_ms, d_m, amp) in enumerate(zip(
        peaks['times_ms'][:5],
        peaks['distances_m'][:5],
        peaks['amplitudes'][:5],
    )):
        print(f"  Peak {i+1}: t={t_ms:.2f} ms | d≈{d_m:.2f} m | amp={amp:.4f}")

    if snr_db < 5:
        print("\n[WARNING] Low SNR! Your laptop speaker may not reach 18-22 kHz.")
        print("  Try: Lower the chirp start to 15 kHz (edit config/sonar_config.yaml)")
    elif snr_db < 15:
        print(f"\n[NOTE] Moderate SNR ({snr_db:.1f} dB). Using Golay sequences will improve it.")
    else:
        print(f"\n[OK] Good SNR ({snr_db:.1f} dB). Hardware is suitable for micro-sonar.")

    result = {
        "chirp": chirp,
        "emit_buffer": emit_buffer,
        "recording": recording,
        "echo": echo,
        "rir": rir,
        "deconv_rir": deconv_rir,
        "peaks": peaks,
        "rt60": rt60,
        "snr_db": snr_db,
        "sample_rate": SAMPLE_RATE,
    }

    if save_recording:
        _save_wav("loopback_recording.wav", recording, SAMPLE_RATE)
        print(f"\nRecording saved → loopback_recording.wav")

    return result


def plot_results(result: dict) -> None:
    """Generate diagnostic plots from the loopback test."""
    if not MPL_AVAILABLE:
        print("[SKIP] matplotlib not available — skipping plots.")
        return

    sr = result["sample_rate"]
    chirp = result["chirp"]
    recording = result["recording"]
    rir = result["rir"]
    peaks = result["peaks"]

    t_rec = np.linspace(0, len(recording) / sr, len(recording)) * 1000  # ms
    t_rir = np.linspace(0, len(rir) / sr, len(rir)) * 1000              # ms
    freqs = np.fft.rfftfreq(len(recording), 1 / sr) / 1000              # kHz

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Micro-Sonar Hardware Loopback Diagnostic", fontsize=14, fontweight="bold")

    # 1. Emitted chirp waveform
    t_chirp = np.linspace(0, len(chirp) / sr, len(chirp)) * 1000
    axes[0, 0].plot(t_chirp, chirp, color="steelblue", linewidth=0.8)
    axes[0, 0].set(title="Emitted Chirp (18–22 kHz)", xlabel="Time (ms)", ylabel="Amplitude")

    # 2. Recorded signal
    axes[0, 1].plot(t_rec, recording, color="darkorange", linewidth=0.5, alpha=0.8)
    axes[0, 1].set(title="Recorded Echo Signal", xlabel="Time (ms)", ylabel="Amplitude")

    # 3. Frequency spectrum of recording
    spec = 20 * np.log10(np.abs(np.fft.rfft(recording)) + 1e-9)
    axes[1, 0].plot(freqs, spec, color="navy", linewidth=0.8)
    axes[1, 0].axvspan(18, 22, alpha=0.2, color="green", label="18–22 kHz band")
    axes[1, 0].set(title="Recording Spectrum", xlabel="Frequency (kHz)", ylabel="Magnitude (dB)", xlim=[0, 25])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Frequency spectrum of chirp
    chirp_full = np.pad(chirp, (0, len(recording) - len(chirp)))
    chirp_spec = 20 * np.log10(np.abs(np.fft.rfft(chirp_full)) + 1e-9)
    axes[1, 1].plot(freqs, chirp_spec, color="green", linewidth=0.8)
    axes[1, 1].axvspan(18, 22, alpha=0.2, color="green")
    axes[1, 1].set(title="Chirp Reference Spectrum", xlabel="Frequency (kHz)", ylabel="Magnitude (dB)", xlim=[0, 25])
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Matched filter RIR
    axes[2, 0].plot(t_rir[:len(rir)], rir, color="crimson", linewidth=0.6)
    if len(peaks["indices"]) > 0:
        axes[2, 0].scatter(
            peaks["times_ms"],
            rir[peaks["indices"]],
            color="gold", s=60, zorder=5, label="Detected peaks"
        )
        axes[2, 0].legend()
    axes[2, 0].set(title="Room Impulse Response (Matched Filter)", xlabel="Time (ms)", ylabel="Amplitude")
    axes[2, 0].set_xlim([0, min(200, t_rir[-1])])

    # 6. Energy decay curve (RT60)
    energy = rir.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-20) + 1e-20)
    axes[2, 1].plot(t_rir[:len(edc_db)], edc_db, color="purple", linewidth=0.8)
    axes[2, 1].axhline(-60, color="gray", linestyle="--", label="-60 dB")
    axes[2, 1].set(title=f"Energy Decay (RT60≈{result['rt60']:.3f}s)" if not np.isnan(result['rt60']) else "Energy Decay Curve",
                   xlabel="Time (ms)", ylabel="Level (dB)")
    axes[2, 1].set_ylim([-80, 5])
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig("loopback_diagnostic.png", dpi=150, bbox_inches="tight")
    print("Diagnostic plot saved → loopback_diagnostic.png")
    plt.show()


def _save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Save audio as a WAV file (without scipy dependency)."""
    try:
        import wave, struct
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
    except Exception as e:
        print(f"[WARNING] Could not save WAV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Micro-Sonar Laptop Hardware Loopback Test")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
    parser.add_argument("--sweeps", type=int, default=3, help="Number of averaging sweeps")
    parser.add_argument("--no-save", action="store_true", help="Do not save recording WAV")
    parser.add_argument("--no-plot", action="store_true", help="Skip diagnostic plot")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    result = run_loopback_test(
        input_device=args.input_device,
        output_device=args.output_device,
        n_sweeps=args.sweeps,
        save_recording=not args.no_save,
    )

    if result and not args.no_plot:
        plot_results(result)

