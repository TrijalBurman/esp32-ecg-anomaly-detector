from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import wfdb

from preprocess import bandpass_ecg, bpm_from_peaks, detect_r_peaks, extract_beat_window
from serial_ecg import available_ports, read_samples, samples_to_arrays


BASE_DIR = Path(__file__).resolve().parents[1]
MITBIH_DIR = BASE_DIR / "data" / "mitdb"
MODEL_PATH = BASE_DIR / "models" / "ecg_aami_cnn.keras"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map.json"
DEFAULT_SAMPLE_RATE = 250.0


@st.cache_resource
def load_classifier() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    labels = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))["labels"]
    return tf.keras.models.load_model(MODEL_PATH), labels


def estimate_sample_rate(millis: np.ndarray, fallback: float) -> float:
    if millis.size < 3:
        return fallback

    diffs = np.diff(millis)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return fallback

    fs = 1000.0 / float(np.median(diffs))
    if not np.isfinite(fs) or fs < 50.0:
        return fallback
    return fs


def plot_ecg(times: np.ndarray, ecg: np.ndarray, peaks: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=ecg, mode="lines", name="ECG"))
    if peaks.size:
        fig.add_trace(
            go.Scatter(
                x=times[peaks],
                y=ecg[peaks],
                mode="markers",
                name="R-peaks",
                marker=dict(size=8),
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Seconds",
        yaxis_title="Filtered ECG",
    )
    return fig


@st.cache_data
def load_preview_segment(record_name: str, start_sec: float, duration_sec: float) -> tuple[np.ndarray, np.ndarray, float]:
    record_path = MITBIH_DIR / record_name
    if not (record_path.with_suffix(".dat")).exists():
        raise FileNotFoundError(f"MIT-BIH record not found: {record_path.with_suffix('.dat')}")

    header = wfdb.rdheader(str(record_path))
    fs = float(header.fs)
    start = int(start_sec * fs)
    stop = int((start_sec + duration_sec) * fs)
    record = wfdb.rdrecord(str(record_path), sampfrom=start, sampto=stop)

    channel_names = [name.upper() for name in record.sig_name]
    channel = channel_names.index("MLII") if "MLII" in channel_names else 0
    ecg = record.p_signal[:, channel].astype(np.float32)
    times = np.arange(ecg.size, dtype=np.float32) / fs
    return times, ecg, fs


def classify_beats(model: tf.keras.Model, labels: list[str], ecg: np.ndarray, peaks: np.ndarray, fs: float):
    rows = []
    beat_tensors = []
    beat_peak_indices = []

    for peak in peaks:
        beat = extract_beat_window(ecg, int(peak), fs)
        if beat is not None:
            beat_tensors.append(beat)
            beat_peak_indices.append(int(peak))

    if not beat_tensors:
        return rows

    x = np.asarray(beat_tensors, dtype=np.float32)[..., np.newaxis]
    probs = model.predict(x, verbose=0)
    pred_indices = np.argmax(probs, axis=1)

    for peak, pred_index, prob_row in zip(beat_peak_indices, pred_indices, probs):
        rows.append(
            {
                "time_sec": round(float(peak / fs), 3),
                "class": labels[int(pred_index)],
                "confidence": round(float(np.max(prob_row)), 3),
                **{f"p_{label}": round(float(prob_row[idx]), 3) for idx, label in enumerate(labels)},
            }
        )

    return rows


def summarize_predictions(predictions: list[dict], labels: list[str]) -> tuple[str, float, float]:
    if not predictions:
        return "None", 0.0, 0.0

    vote_counts = {label: 0 for label in labels}
    confidence_sums = {label: 0.0 for label in labels}
    for prediction in predictions:
        label = prediction["class"]
        vote_counts[label] += 1
        confidence_sums[label] += float(prediction["confidence"])

    overall_class = max(
        labels,
        key=lambda label: (vote_counts[label], confidence_sums[label]),
    )
    matching = [prediction for prediction in predictions if prediction["class"] == overall_class]
    agreement = len(matching) / len(predictions)
    overall_confidence = float(np.median([prediction["confidence"] for prediction in matching]))
    return overall_class, agreement, overall_confidence


def main() -> None:
    st.set_page_config(page_title="ESP32 ECG Anomaly Dashboard", layout="wide")
    st.title("ESP32 ECG Anomaly Dashboard")

    ports = available_ports()
    with st.sidebar:
        st.header("Mode")
        mode = st.radio("Data source", ["MIT-BIH preview sample", "Live ESP32 serial"])

        st.header("Serial")
        port = st.selectbox("Port", ports if ports else ["No serial ports found"], disabled=mode != "Live ESP32 serial")
        baud = st.number_input(
            "Baud",
            min_value=9600,
            max_value=921600,
            value=115200,
            step=9600,
            disabled=mode != "Live ESP32 serial",
        )
        sample_rate = st.number_input("Sample rate Hz", min_value=100.0, max_value=1000.0, value=DEFAULT_SAMPLE_RATE)
        window_seconds = st.slider("Capture window seconds", min_value=5, max_value=30, value=10)

        st.header("Preview")
        preview_record = st.selectbox("MIT-BIH record", ["100", "101", "103", "105", "106", "200", "208"])
        preview_start = st.slider("Preview start second", min_value=0, max_value=120, value=10)

        run_label = "Run MIT-BIH preview" if mode == "MIT-BIH preview sample" else "Start live classification"
        run_disabled = mode == "Live ESP32 serial" and not ports
        run = st.button(run_label, type="primary", disabled=run_disabled)

    try:
        model, labels = load_classifier()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train first with: python src\\train_mitbih.py")
        return

    status = st.empty()
    chart = st.empty()
    metrics = st.container()
    table = st.empty()

    if not run:
        status.info("Run the MIT-BIH preview, or connect the ESP32 and start live classification.")
        return

    if mode == "MIT-BIH preview sample":
        status.info(f"Loading MIT-BIH record {preview_record} preview...")
        try:
            times, volts, actual_sample_rate = load_preview_segment(
                preview_record,
                float(preview_start),
                float(window_seconds),
            )
        except FileNotFoundError as exc:
            status.error(str(exc))
            st.info("Train first with: python src\\train_mitbih.py")
            return

        lead_off_pct = 0.0
        sample_count = len(volts)
    else:
        status.info(f"Reading {window_seconds} seconds from {port}...")
        samples = read_samples(port, int(baud), float(window_seconds))

        if len(samples) < int(sample_rate * 2):
            status.error("Not enough samples received. Check COM port, baud rate, and Arduino Serial Monitor.")
            return

        millis, volts, lead_off = samples_to_arrays(samples)
        actual_sample_rate = estimate_sample_rate(millis, float(sample_rate))
        times = (millis - millis[0]) / 1000.0
        lead_off_pct = float(np.mean(lead_off) * 100.0)
        sample_count = len(samples)

    ecg = bandpass_ecg(volts, actual_sample_rate)
    peaks = detect_r_peaks(volts, actual_sample_rate)
    bpm = bpm_from_peaks(peaks, actual_sample_rate)
    predictions = classify_beats(model, labels, ecg, peaks, actual_sample_rate)

    status.success(f"Processed {sample_count} samples and detected {len(peaks)} beats.")
    chart.plotly_chart(plot_ecg(times, ecg, peaks), use_container_width=True)

    latest = predictions[-1]["class"] if predictions else "None"
    latest_confidence = predictions[-1]["confidence"] if predictions else 0.0
    overall_class, agreement, overall_confidence = summarize_predictions(predictions, labels)

    col1, col2, col3, col4, col5, col6 = metrics.columns(6)
    col1.metric("BPM", "--" if bpm is None else f"{bpm:.1f}")
    col2.metric("Overall class", overall_class)
    col3.metric("Agreement", f"{agreement * 100.0:.0f}%")
    col4.metric("Overall confidence", f"{overall_confidence:.2f}")
    col5.metric("Latest class", latest)
    col6.metric("Latest confidence", f"{latest_confidence:.2f}")
    st.caption(f"Sample rate: {actual_sample_rate:.1f} Hz | Lead-off samples: {lead_off_pct:.1f}%")

    if predictions:
        table.dataframe(predictions, use_container_width=True, hide_index=True)
    else:
        table.warning("No complete beat windows were available for classification.")


if __name__ == "__main__":
    main()
