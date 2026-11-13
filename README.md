# Micro-Sonar: Active Acoustic Spatial Mapping Engine

A smartphone-based micro-sonar system that uses near-ultrasonic chirps (18–22 kHz)
to map physical room dimensions and wall materials — no camera, GPS, or internet needed.

## Quick Start (Laptop Testing)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test your laptop hardware (speaker + mic at 18-22 kHz)
python scripts/test_hardware_loopback.py

# 3. List audio devices if needed
python scripts/test_hardware_loopback.py --list-devices

# 4. Generate synthetic training dataset
python -m simulation.dataset_builder --n_rooms 500 --output data/dataset.npz

# 5. Train the model
python scripts/train_model.py --dataset data/dataset.npz --backbone custom_cnn

# 6. Run live sonar demo (DSP only)
python scripts/live_demo.py

# 7. Run with trained model
python scripts/live_demo.py --model checkpoints/best_sonar_model.pt --backbone custom_cnn
```

## Project Structure

```
micro-sonar/
├── config/sonar_config.yaml      ← All tunable parameters
├── dsp/
│   ├── chirp.py                  ← Log/linear/Golay chirp generators
│   ├── matched_filter.py         ← Cross-correlation RIR extraction + RT60
│   └── feature_extractor.py      ← Mel-spectrogram computation
├── simulation/
│   ├── room_sim.py               ← pyroomacoustics room generator
│   └── dataset_builder.py        ← Batch dataset builder (run as script)
├── models/
│   ├── sonar_net.py              ← PyTorch multi-task CNN
│   └── export_tflite.py          ← ONNX + TFLite export pipeline
├── scripts/
│   ├── test_hardware_loopback.py ← FIRST RUN THIS — hardware validation
│   ├── train_model.py            ← Training loop
│   └── live_demo.py              ← Interactive live sonar demo
└── android/                      ← Phase 3: C++/Oboe/JNI scaffold (TODO)
```

## Phase Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Hardware loopback validation | ✅ Ready |
| 2 | Synthetic dataset + ML training | ✅ Ready |
| 3 | Live laptop demo | ✅ Ready |
| 4 | C++ DSP + Android NDK/Oboe | 🔜 Next |
| 5 | TFLite inference on Android | 🔜 Planned |
| 6 | Unity/Godot C-FFI bridge | 🔜 Planned |

## Configuration

All parameters are in [`config/sonar_config.yaml`](config/sonar_config.yaml).
Key parameters:

- `chirp.f_start / f_end`: 18000–22000 Hz (adjust if laptop cuts off earlier)
- `chirp.mode`: `logarithmic` (best for room acoustics) or `linear`
- `simulation.n_rooms`: Number of synthetic training rooms
- `model.backbone`: `custom_cnn` (fast, 200K params) or `mobilenet_v3_small`

## Hardware Notes

- **Laptop Speaker Cutoff**: Most consumer laptop speakers roll off above 16 kHz.
  Run `test_hardware_loopback.py` first — if SNR < 5 dB, lower `chirp.f_start` to 15000.
- **External Speakers**: A USB or Bluetooth speaker improves high-frequency output significantly.
- **Android**: The Oboe/NDK backend (Phase 4) handles high-frequency output much better
  than laptop audio.

