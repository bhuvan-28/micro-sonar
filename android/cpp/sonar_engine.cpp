/**
 * android/cpp/sonar_engine.cpp
 * =============================
 * Stub implementation of the Micro-Sonar native C++ engine.
 *
 * Phase 3 TODO:
 *  - Replace stubs with real Oboe audio stream management
 *  - Integrate KissFFT for matched filtering
 *  - Integrate TFLite C++ API for model inference
 *
 * The Python prototype (scripts/) is the reference implementation.
 * Port the DSP logic from dsp/matched_filter.py and dsp/feature_extractor.py.
 */

#include "sonar_engine.h"
#include <cmath>
#include <cstdio>
#include <cstring>

// Placeholder state
static float g_last_width    = 0.0f;
static float g_last_length   = 0.0f;
static float g_last_height   = 0.0f;
static int   g_last_material = -1;
static bool  g_has_result    = false;
static bool  g_initialized   = false;

int sonar_init(int sample_rate, int chirp_f_start, int chirp_f_end) {
    // TODO: Open Oboe stream
    // OboeStreamBuilder builder;
    // builder.setSampleRate(sample_rate)
    //        .setChannelCount(1)
    //        .setFormat(AudioFormat::Float)
    //        .setPerformanceMode(PerformanceMode::LowLatency)
    //        ...
    printf("[MicroSonar] sonar_init(%d Hz, %d-%d Hz chirp) — stub\n",
           sample_rate, chirp_f_start, chirp_f_end);
    g_initialized = true;
    return 0;
}

int sonar_trigger_sweep(void) {
    if (!g_initialized) return -1;
    // TODO:
    //  1. Generate log chirp in [f_start, f_end] using KissFFT
    //  2. Write chirp to Oboe output stream
    //  3. Record echo via Oboe input stream (synchronised)
    //  4. Run matched filter (cross-correlation via KissFFT)
    //  5. Extract Mel-spectrogram
    //  6. Run TFLite inference
    //  7. Store result in g_last_* fields
    printf("[MicroSonar] sonar_trigger_sweep() — stub (returning synthetic data)\n");
    // Synthetic placeholder result
    g_last_width    = 4.5f;
    g_last_length   = 6.2f;
    g_last_height   = 2.8f;
    g_last_material = 1;  // drywall
    g_has_result    = true;
    return 0;
}

int sonar_get_room_metrics(
    float* out_width,
    float* out_length,
    float* out_height,
    int*   out_material_class
) {
    if (!g_has_result) return -1;
    *out_width          = g_last_width;
    *out_length         = g_last_length;
    *out_height         = g_last_height;
    *out_material_class = g_last_material;
    return 0;
}

void sonar_destroy(void) {
    // TODO: Close Oboe stream and free TFLite interpreter
    printf("[MicroSonar] sonar_destroy() — stub\n");
    g_initialized = false;
    g_has_result  = false;
}

