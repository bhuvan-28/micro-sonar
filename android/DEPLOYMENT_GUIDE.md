# Micro-Sonar: Android Porting & Deployment Guide

This guide walks through deploying the Active Acoustic Micro-Sonar engine onto an **Android phone** using **C++ (NDK + Oboe)** and **ONNX Runtime Mobile / TFLite**.

---

## 1. Architectural Architecture on Android

```
[ Android App UI (Kotlin / Jetpack Compose) ]
                      │  (JNI Calls)
                      ▼
[ C++ Native Core (libmicrosonar.so) ]
   ├── Audio I/O   : Google Oboe (AAudio backend, low-latency, 48 kHz PCM)
   ├── Chirp Syn   : 18–22 kHz Log Chirp buffer generator
   ├── DSP Engine  : KissFFT (Cross-correlation matched filter & Mel-filterbank)
   └── ML Runtime  : ONNX Runtime Mobile C++ API (sonar_model.onnx)
```

---

## 2. Prerequisites & Tools

1. **Android Studio** (Koala / Ladybug or newer).
2. **Android NDK** (version 26+ recommended) and **CMake 3.22+** (install via Android Studio SDK Manager).
3. **Android Device** running Android 9+ (API Level 28+) supporting `AAudio`.
4. Exported model: `models/sonar_model.onnx` (already generated, **180 KB**).

---

## 3. Step-by-Step Implementation

### Step 3.1: Permissions (`AndroidManifest.xml`)

Add audio recording and low-latency feature flags:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <!-- Mandatory for micro-sonar echo recording -->
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />

    <!-- Request low-latency hardware path -->
    <uses-feature android:name="android.hardware.audio.low_latency" android:required="true" />
    <uses-feature android:name="android.hardware.audio.pro" android:required="false" />
    ...
</manifest>
```

---

### Step 3.2: Native Audio Engine with Oboe

Google's **Oboe** handles device fragmentation and automatically picks the lowest latency audio path (`AAudio` or fallback `OpenSL ES`).

In your `android/app/build.gradle.kts`:
```kotlin
dependencies {
    implementation("com.google.oboe:oboe:1.9.0")
    // ONNX Runtime Mobile for Android
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
}
```

In `cpp/sonar_engine.cpp`, configure the full-duplex Oboe stream:
```cpp
#include <oboe/Oboe.h>

oboe::AudioStreamBuilder builder;
builder.setDirection(oboe::Direction::Input)
       ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
       ->setSharingMode(oboe::SharingMode::Exclusive)
       ->setFormat(oboe::AudioFormat::Float)
       ->setChannelCount(oboe::ChannelCount::Mono)
       ->setSampleRate(48000)
       ->setDataCallback(myAudioCallback);
```

---

### Step 3.3: DSP Cross-Correlation in C++

Port the matched-filter deconvolution logic from `dsp/matched_filter.py` using **KissFFT**:

1. Multiply frequency-domain echo with conjugate of the reference chirp:
   $$\mathbf{H}[f] = \mathbf{ECHO}[f] \times \overline{\mathbf{CHIRP}}[f]$$
2. Execute inverse FFT to extract the raw Room Impulse Response ($h[t]$).
3. Compute the $128 \times 128$ Mel-Spectrogram tensor.

---

### Step 3.4: On-Device Model Inference

Place `sonar_model.onnx` into `app/src/main/assets/`. Load and execute using either:

#### Option A: Kotlin / Java (Quickest)
```kotlin
import ai.onnxruntime.*

val env = OrtEnvironment.getEnvironment()
val session = env.createSession(assets.open("sonar_model.onnx").readBytes())

val inputTensor = OnnxTensor.createTensor(env, melSpectrogramFloatBuffer, longArrayOf(1, 1, 128, 128))
val results = session.run(mapOf("mel_spectrogram" to inputTensor))

val dimensions = (results[0].value as Array<FloatArray>)[0] // [Width, Length, Height]
val materialLogits = (results[1].value as Array<FloatArray>)[0] // 5 classes
```

#### Option B: C++ Native (Zero-copy in audio thread)
Use ONNX Runtime C++ API directly inside `sonar_engine.cpp` to run end-to-end inference without JNI data marshalling.

---

## 4. Acoustic Calibration on Phones

* **Mic & Speaker Placement:** On phones, the earpiece/bottom speaker and primary mic are separated by only 10–15 cm. The direct-path arrival occurs within $0.3\text{–}0.5\text{ ms}$, followed by room surface echoes.
* **Frequency Headroom:** Smartphone speakers typically handle 18–22 kHz with higher fidelity than laptops due to smaller speaker membranes designed for crisp high frequencies.

