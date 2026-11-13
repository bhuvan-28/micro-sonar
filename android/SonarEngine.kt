package com.example.microsonar

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import kotlin.math.*

data class RoomMetrics(
    val width: Float,
    val length: Float,
    val height: Float,
    val volume: Float,
    val material: String,
    val confidence: Float
)

class SonarEngine(private val context: Context) {

    private val sampleRate = 48000
    private val chirpDuration = 0.1f // 100 ms
    private val fStart = 18000f
    private val fEnd = 22000f

    private val nFft = 2048
    private val hopLength = 512
    private val nMels = 128
    private val fMin = 17000f
    private val fMax = 23000f

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession
    private val materials = arrayOf("Concrete", "Drywall", "Wood", "Carpet", "Glass")

    init {
        val modelBytes = context.assets.open("sonar_model.onnx").readBytes()
        ortSession = ortEnv.createSession(modelBytes)
    }

    private fun generateLogChirp(): FloatArray {
        val nSamples = (chirpDuration * sampleRate).toInt()
        val chirp = FloatArray(nSamples)
        val logRatio = ln(fEnd / fStart)
        for (i in 0 until nSamples) {
            val t = i.toFloat() / sampleRate
            val phase = 2.0 * Math.PI * fStart * chirpDuration / logRatio * (exp(t / chirpDuration * logRatio) - 1.0)
            val window = 0.5 * (1.0 - cos(2.0 * Math.PI * i / nSamples))
            chirp[i] = (0.85 * sin(phase) * window).toFloat()
        }
        return chirp
    }

    suspend fun runSweep(): RoomMetrics = withContext(Dispatchers.Default) {
        val chirp = generateLogChirp()
        val totalRecordSamples = (0.6 * sampleRate).toInt()
        val recordedBuffer = FloatArray(totalRecordSamples)

        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .build())
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build())
            .setBufferSizeInBytes(chirp.size * 4)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        track.write(chirp, 0, chirp.size, AudioTrack.WRITE_BLOCKING)

        val minBuf = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
            max(minBuf, totalRecordSamples * 4)
        )

        recorder.startRecording()
        track.play()

        var readTotal = 0
        while (readTotal < totalRecordSamples) {
            val read = recorder.read(recordedBuffer, readTotal, totalRecordSamples - readTotal, AudioRecord.READ_BLOCKING)
            if (read > 0) readTotal += read else break
        }

        try { recorder.stop() } catch (_: Exception) {}
        recorder.release()
        track.release()

        // 1. Matched filter cross-correlation
        val rir = matchedFilter(recordedBuffer, chirp)

        // 2. Compute proper Mel-Spectrogram (128x128) matching Python training
        val melSpec = computeMelSpectrogram(rir)

        // 3. Run ONNX Inference
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(melSpec), longArrayOf(1, 1, 128, 128))
        val output = ortSession.run(mapOf("mel_spectrogram" to tensor))

        val dims = (output[0].value as Array<FloatArray>)[0]
        val logits = (output[1].value as Array<FloatArray>)[0]

        // Softmax & argmax
        var maxLogit = logits[0]
        var bestIdx = 0
        for (i in 1 until logits.size) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i]
                bestIdx = i
            }
        }

        var sumExp = 0.0
        for (v in logits) sumExp += exp(v.toDouble())
        val confidence = (exp(maxLogit.toDouble()) / sumExp).toFloat()

        RoomMetrics(
            width = dims[0],
            length = dims[1],
            height = dims[2],
            volume = dims[0] * dims[1] * dims[2],
            material = materials[bestIdx],
            confidence = confidence
        )
    }

    private fun matchedFilter(echo: FloatArray, ref: FloatArray): FloatArray {
        val out = FloatArray(echo.size)
        val step = 1
        val maxRefLen = min(ref.size, 2400)
        var maxVal = 1e-9f

        for (i in 0 until echo.size - maxRefLen step step) {
            var sum = 0f
            for (j in 0 until maxRefLen) {
                sum += echo[i + j] * ref[j]
            }
            out[i] = sum
            if (abs(sum) > maxVal) maxVal = abs(sum)
        }

        for (i in out.indices) out[i] /= maxVal
        return out
    }

    private fun computeMelSpectrogram(signal: FloatArray): FloatArray {
        val nFrames = 128
        val hop = max(1, (signal.size - nFft) / nFrames)
        val spec2D = Array(nMels) { FloatArray(nFrames) }

        // Window
        val window = FloatArray(nFft) { 0.5f * (1.0f - cos(2.0f * Math.PI.toFloat() * it / nFft)) }

        // Simple Mel filter centers
        val melMin = 2595.0f * log10(1.0f + fMin / 700.0f)
        val melMax = 2595.0f * log10(1.0f + fMax / 700.0f)
        val melBins = FloatArray(nMels + 2) {
            val m = melMin + (melMax - melMin) * it / (nMels + 1)
            700.0f * (10.0f.pow(m / 2595.0f) - 1.0f)
        }
        val binIdx = IntArray(nMels + 2) {
            ((nFft + 1) * melBins[it] / sampleRate).toInt().coerceIn(0, nFft / 2)
        }

        var sMin = Float.MAX_VALUE
        var sMax = -Float.MAX_VALUE

        for (f in 0 until nFrames) {
            val start = (f * hop).coerceIn(0, signal.size - nFft)
            // Power spectrum via DFT on band
            val power = FloatArray(nFft / 2 + 1)
            for (k in binIdx.first() until min(binIdx.last() + 1, nFft / 2)) {
                var re = 0f
                var im = 0f
                val omega = 2.0 * Math.PI * k / nFft
                for (n in 0 until nFft) {
                    val sample = signal[start + n] * window[n]
                    re += (sample * cos(omega * n)).toFloat()
                    im -= (sample * sin(omega * n)).toFloat()
                }
                power[k] = re * re + im * im
            }

            // Mel filterbank integration
            for (m in 0 until nMels) {
                val fLeft = binIdx[m]
                val fCenter = binIdx[m + 1]
                val fRight = binIdx[m + 2]
                var melEnergy = 0f

                for (k in fLeft until fCenter) {
                    if (fCenter > fLeft) melEnergy += power[k] * (k - fLeft).toFloat() / (fCenter - fLeft)
                }
                for (k in fCenter until fRight) {
                    if (fRight > fCenter) melEnergy += power[k] * (fRight - k).toFloat() / (fRight - fCenter)
                }

                val logMel = (10.0f * log10(melEnergy + 1e-9f))
                spec2D[m][f] = logMel
                if (logMel < sMin) sMin = logMel
                if (logMel > sMax) sMax = logMel
            }
        }

        // Normalize to [-1.0, 1.0]
        val range = if (sMax > sMin) sMax - sMin else 1f
        val outTensor = FloatArray(nMels * nFrames)
        var p = 0
        for (m in 0 until nMels) {
            for (f in 0 until nFrames) {
                val norm = 2.0f * (spec2D[m][f] - sMin) / range - 1.0f
                outTensor[p++] = norm
            }
        }
        return outTensor
    }
}

