from __future__ import annotations

import numpy as np
from scipy import signal


TARGET_BEAT_SAMPLES = 256


def bandpass_ecg(samples: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < max(16, int(fs)):
        return samples - np.mean(samples)

    nyquist = 0.5 * fs
    high = min(high, nyquist - 1.0)
    sos = signal.butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, samples).astype(np.float32)


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32)
    beat = beat - np.mean(beat)
    std = np.std(beat)
    if std < 1e-6:
        return beat
    return beat / std


def resample_beat(beat: np.ndarray, target_samples: int = TARGET_BEAT_SAMPLES) -> np.ndarray:
    return signal.resample(beat, target_samples).astype(np.float32)


def extract_beat_window(
    samples: np.ndarray,
    peak_index: int,
    fs: float,
    before_sec: float = 0.35,
    after_sec: float = 0.45,
    target_samples: int = TARGET_BEAT_SAMPLES,
) -> np.ndarray | None:
    before = int(before_sec * fs)
    after = int(after_sec * fs)
    start = peak_index - before
    end = peak_index + after
    if start < 0 or end > len(samples):
        return None

    beat = samples[start:end]
    beat = resample_beat(beat, target_samples)
    return normalize_beat(beat)


def detect_r_peaks(samples: np.ndarray, fs: float) -> np.ndarray:
    filtered = bandpass_ecg(np.asarray(samples, dtype=np.float32), fs)
    centered = filtered - np.median(filtered)

    if abs(np.min(centered)) > abs(np.max(centered)):
        centered = -centered

    min_distance = int(0.25 * fs)
    prominence = max(0.05, float(np.std(centered) * 0.8))
    peaks, _ = signal.find_peaks(centered, distance=min_distance, prominence=prominence)
    return peaks.astype(int)


def bpm_from_peaks(peaks: np.ndarray, fs: float) -> float | None:
    if len(peaks) < 2:
        return None
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.25) & (rr < 2.5)]
    if rr.size == 0:
        return None
    return float(60.0 / np.median(rr))

