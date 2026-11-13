"""
simulation/room_sim.py
======================
Synthetic Room Impulse Response (RIR) generator using pyroomacoustics.

Generates shoebox rooms with:
 - Randomised dimensions (width, length, height)
 - Randomised absorption materials from a preset library
 - Randomised microphone/speaker placement (simulating hand-held device)
 - Image Source Method up to specified reflection order
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import pyroomacoustics as pra
    PRA_AVAILABLE = True
except ImportError:
    PRA_AVAILABLE = False
    print("[WARNING] pyroomacoustics not installed. Run: pip install pyroomacoustics")


# ---------------------------------------------------------------------------
# Material absorption coefficients at octave-band frequencies (Hz):
# 125, 250, 500, 1000, 2000, 4000
# Source: Kuttruff "Room Acoustics", typical measured values
# ---------------------------------------------------------------------------
MATERIAL_LIBRARY = {
    "concrete": {
        "coeffs": [0.02, 0.03, 0.03, 0.03, 0.04, 0.07],
        "label": 0,
    },
    "drywall": {
        "coeffs": [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
        "label": 1,
    },
    "wood": {
        "coeffs": [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
        "label": 2,
    },
    "carpet": {
        "coeffs": [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
        "label": 3,
    },
    "glass": {
        "coeffs": [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],
        "label": 4,
    },
}


@dataclass
class RoomConfig:
    """Container for a single room configuration."""
    width: float               # metres
    length: float              # metres
    height: float              # metres
    material: str              # material name key from MATERIAL_LIBRARY
    material_label: int        # integer class label
    mic_pos: np.ndarray        # shape (3,) - microphone position
    src_pos: np.ndarray        # shape (3,) - speaker/source position
    rir: Optional[np.ndarray] = field(default=None, repr=False)
    rt60_estimated: float = 0.0


def _random_position_in_room(
    width: float,
    length: float,
    height: float,
    margin: float = 0.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Sample a random 3-D point inside the room with margin from walls."""
    if rng is None:
        rng = np.random.default_rng()
    return np.array([
        rng.uniform(margin, width - margin),
        rng.uniform(margin, length - margin),
        rng.uniform(0.8, min(height - 0.3, 2.0)),  # realistic hand/ear height
    ])


def simulate_room(
    config: RoomConfig,
    sample_rate: int = 48_000,
    max_order: int = 3,
) -> np.ndarray:
    """
    Simulate the Room Impulse Response for a given RoomConfig.

    Parameters
    ----------
    config : RoomConfig
        Room geometry, material, and source/mic positions.
    sample_rate : int
        Audio sample rate in Hz.
    max_order : int
        Image source reflection order. Higher = more accurate but slower.

    Returns
    -------
    np.ndarray
        RIR array, float32, length varies with RT60 and room size.
    """
    if not PRA_AVAILABLE:
        raise RuntimeError("pyroomacoustics is required. pip install pyroomacoustics")

    mat_data = MATERIAL_LIBRARY[config.material]
    # Build absorption material using pra.Material
    absorption = pra.Material(
        energy_absorption=dict(
            coeffs=mat_data["coeffs"],
            center_freqs=[125, 250, 500, 1000, 2000, 4000],
        )
    )

    room_dims = [config.width, config.length, config.height]
    room = pra.ShoeBox(
        room_dims,
        fs=sample_rate,
        materials=absorption,
        max_order=max_order,
        ray_tracing=False,
        air_absorption=True,
    )

    # Add a Dirac impulse as the source signal (we want the RIR, not convolution)
    n_samples = int(sample_rate * 1.5)  # 1.5s buffer — enough for large rooms
    impulse = np.zeros(n_samples, dtype=np.float32)
    impulse[0] = 1.0
    room.add_source(config.src_pos.tolist(), signal=impulse)
    mic_array = config.mic_pos[:, np.newaxis]  # shape (3, 1)
    room.add_microphone(mic_array)
    room.simulate()

    rir = room.rir[0][0].astype(np.float32)
    # Normalise
    peak = np.max(np.abs(rir))
    if peak > 1e-10:
        rir /= peak
    return rir


def generate_random_room(
    rng: np.random.Generator = None,
    width_range: Tuple[float, float] = (2.5, 10.0),
    length_range: Tuple[float, float] = (2.5, 12.0),
    height_range: Tuple[float, float] = (2.4, 4.0),
    materials: List[str] = None,
) -> RoomConfig:
    """
    Sample a random room configuration.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy RNG instance (for reproducibility).
    *_range : (float, float)
        (min, max) dimension ranges.
    materials : list of str
        Material names to choose from (defaults to all in MATERIAL_LIBRARY).

    Returns
    -------
    RoomConfig
    """
    if rng is None:
        rng = np.random.default_rng()
    if materials is None:
        materials = list(MATERIAL_LIBRARY.keys())

    w = rng.uniform(*width_range)
    l = rng.uniform(*length_range)
    h = rng.uniform(*height_range)
    mat = rng.choice(materials)

    mic_pos = _random_position_in_room(w, l, h, rng=rng)
    src_pos = _random_position_in_room(w, l, h, rng=rng)
    # Ensure mic and source are at least 0.5m apart
    while np.linalg.norm(mic_pos - src_pos) < 0.5:
        src_pos = _random_position_in_room(w, l, h, rng=rng)

    return RoomConfig(
        width=w,
        length=l,
        height=h,
        material=mat,
        material_label=MATERIAL_LIBRARY[mat]["label"],
        mic_pos=mic_pos,
        src_pos=src_pos,
    )
