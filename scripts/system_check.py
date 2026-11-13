"""Full system check script."""
import sys
sys.path.insert(0, '.')
results = {}

try:
    import numpy as np, scipy, matplotlib, sounddevice, pyroomacoustics, yaml, tqdm, torch
    results['Core Packages'] = 'OK'
except Exception as e:
    results['Core Packages'] = f'FAIL: {e}'

try:
    from dsp.chirp import log_chirp, linear_chirp, golay_pair
    from dsp.matched_filter import matched_filter, extract_rir_peaks, compute_rt60, deconvolve_rir
    from dsp.feature_extractor import compute_mel_spectrogram, compute_energy_envelope
    results['DSP Modules'] = 'OK'
except Exception as e:
    results['DSP Modules'] = f'FAIL: {e}'

try:
    from simulation.room_sim import generate_random_room, simulate_room
    from simulation.dataset_builder import build_dataset
    results['Simulation'] = 'OK'
except Exception as e:
    results['Simulation'] = f'FAIL: {e}'

try:
    from models.sonar_net import build_model, SonarLoss
    m = build_model('custom_cnn')
    n = sum(p.numel() for p in m.parameters())
    results['Model custom_cnn'] = f'OK ({n:,} params)'
except Exception as e:
    results['Model custom_cnn'] = f'FAIL: {e}'

try:
    import torch
    x = torch.randn(2, 1, 128, 128)
    dims, mat = m(x)
    assert dims.shape == (2, 3)
    assert mat.shape == (2, 5)
    results['Forward Pass'] = 'OK (dims=(2,3) mat=(2,5))'
except Exception as e:
    results['Forward Pass'] = f'FAIL: {e}'

try:
    loss_fn = SonarLoss()
    target_dims = torch.rand(2, 3) * 10
    target_mat = torch.randint(0, 5, (2,))
    total, dl, ml = loss_fn(dims, mat, target_dims, target_mat)
    results['Loss Function'] = f'OK (total={total.item():.4f})'
except Exception as e:
    results['Loss Function'] = f'FAIL: {e}'

try:
    chirp = log_chirp(sample_rate=48000)
    assert len(chirp) == 4800
    a, b = golay_pair()
    results['Chirp Generation'] = f'OK (log={len(chirp)} samp, golay={len(a)} samp)'
except Exception as e:
    results['Chirp Generation'] = f'FAIL: {e}'

try:
    import numpy as np
    rng = np.random.default_rng(1)
    cfg = generate_random_room(rng=rng)
    rir_raw = simulate_room(cfg, sample_rate=48000, max_order=2)
    echo = np.convolve(chirp, rir_raw)[:len(chirp) * 6]
    echo = echo + rng.standard_normal(len(echo)).astype(np.float32) * 0.01
    rir = matched_filter(echo, chirp)
    mel = compute_mel_spectrogram(rir)
    env = compute_energy_envelope(rir)
    mshape = str(mel.shape)
    eshape = str(env.shape)
    results['Full DSP Pipeline'] = f'OK mel={mshape} env={eshape}'
except Exception as e:
    results['Full DSP Pipeline'] = f'FAIL: {e}'

try:
    import numpy as np
    d = np.load('data/test_dataset.npz')
    n_samples = d['features'].shape[0]
    fshape = str(d['features'].shape)
    results['Existing Dataset'] = f'OK ({n_samples} samples, {fshape})'
except Exception as e:
    results['Existing Dataset'] = f'FAIL: {e}'

try:
    import os
    wav_exists = os.path.exists('loopback_recording.wav')
    results['Loopback WAV'] = 'OK (file exists)' if wav_exists else 'MISSING'
except Exception as e:
    results['Loopback WAV'] = f'FAIL: {e}'

print()
print('=' * 60)
print('  MICRO-SONAR FULL SYSTEM CHECK')
print('=' * 60)
all_ok = True
for k, v in results.items():
    status = '[OK  ]' if v.startswith('OK') else '[FAIL]'
    if not v.startswith('OK'):
        all_ok = False
    print(f'  {status}  {k}: {v}')
print('=' * 60)
print(f'  OVERALL: {"ALL SYSTEMS GO - READY TO PROCEED" if all_ok else "ISSUES DETECTED"}')
print('=' * 60)

