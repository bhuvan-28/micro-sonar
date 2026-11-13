/**
 * android/cpp/sonar_engine.h
 * ===========================
 * Public C API for the Micro-Sonar native engine.
 * Exposed via JNI to Kotlin/Java and via C-FFI to Unity/Godot.
 *
 * Phase 3 (TODO): Full implementation using Oboe + KissFFT + TFLite.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** Initialise the sonar engine (opens Oboe audio stream at 48 kHz). */
int sonar_init(int sample_rate, int chirp_f_start, int chirp_f_end);

/** Trigger a single chirp sweep and block until echo is captured. */
int sonar_trigger_sweep(void);

/**
 * Get the last inference result.
 * Writes predicted [width, length, height] in metres and material class index.
 * Returns 0 on success, -1 if no result is available.
 */
int sonar_get_room_metrics(
    float* out_width,
    float* out_length,
    float* out_height,
    int*   out_material_class
);

/** Shut down the sonar engine and release Oboe stream. */
void sonar_destroy(void);

#ifdef __cplusplus
}
#endif

