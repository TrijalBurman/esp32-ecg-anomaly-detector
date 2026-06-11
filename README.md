# ESP32 ECG Anomaly Detector

ESP32 + AD8232 heartbeat anomaly detector with a MIT-BIH-trained 1D CNN model and a live Streamlit dashboard.

The ESP32 streams ECG samples from an AD8232 sensor at 250 Hz to a PC over UART. The Python dashboard filters the ECG, detects R-peaks, calculates BPM, extracts beat windows, and classifies each beat into one of five AAMI classes:

| Class | Meaning |
| --- | --- |
| `N` | Normal and bundle branch beats |
| `S` | Supraventricular ectopic beats |
| `V` | Ventricular ectopic beats |
| `F` | Fusion beats |
| `Q` | Unknown / paced / unclassifiable beats |

> **This is an educational prototype, not a medical device.**

---

## Hardware

- ESP32 development board
- AD8232 ECG module
- Three ECG electrode pads (`RA`, `LA`, `RL`)
- USB cable
- PC with Python 3.11 or 3.12

## Wiring

| AD8232 pin | ESP32 pin | Purpose |
| --- | --- | --- |
| `OUTPUT` | `GPIO34` | Analog ECG signal |
| `LO+` | `GPIO26` | Lead-off detection |
| `LO-` | `GPIO27` | Lead-off detection |
| `3.3V` | `3V3` | Sensor power |
| `GND` | `GND` | Common ground |

Electrode placement:

| Electrode | Location |
| --- | --- |
| `RA` | Right wrist |
| `LA` | Left wrist |
| `RL` | Right ankle (reference) |

Use adhesive Ag/AgCl ECG pads for the cleanest signal.

---

## Quick Start

### 1. Flash ESP32

Open in Arduino IDE (`ESP32 Dev Module`, baud `115200`):

```text
firmware/esp32_ad8232_stream/esp32_ad8232_stream.ino
```

### 2. Install Python Dependencies

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Requires Python 3.11 or 3.12 on Windows for TensorFlow compatibility.

### 3. Train Model (first time only)

```powershell
python src\train_mitbih.py
```

Downloads MIT-BIH, applies SMOTE to balance minority classes, trains the 1D CNN, and saves model files under `pc_app/models/`. Expected test accuracy: **~98.2%**.

### 4. Run Dashboard

```powershell
streamlit run src\dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## Dashboard Modes

### MIT-BIH Preview (no hardware needed)

1. Select record (e.g. `100`) and start second
2. Click **Run MIT-BIH preview**

### Live ESP32 Streaming

1. Close Arduino Serial Monitor
2. Select COM port and baud `115200`
3. Click **▶ Start live classification**
4. Click **⏹ Stop streaming** to halt

The dashboard captures in 10-second windows continuously. The ECG chart and beat table stay visible between windows (no blank screen).

---

## Model Performance

Trained on MIT-BIH Arrhythmia Database with SMOTE oversampling to balance minority classes (S, F):

| Metric | Value |
| --- | --- |
| Test accuracy | **98.2%** |
| Validation accuracy | 99.8% |
| Training epochs | 23 (early stopping) |

Training report saved to `pc_app/models/training_report.json`.

---

## Signal Processing

### R-Peak Detection (`preprocess.py`)

- Butterworth bandpass filter: **0.5–40 Hz**
- Artifact suppression: signal clipped to **±4σ** before peak detection
- Physiological refractory period: **0.40 s** (prevents T-wave double-detection)
- Height floor: peaks must exceed **mean + 1σ** to reject noise blips
- BPM range: **30–150 BPM**

### ECG Chart

- Display smoothing: 9-point moving average (display only, does not affect model input)
- ECG monitor style: teal line, red R-peak markers, dark transparent background

---

## Project Structure

```text
firmware/
  esp32_ad8232_stream/
    esp32_ad8232_stream.ino
pc_app/
  requirements.txt
  src/
    dashboard.py        # Streamlit UI and live streaming logic
    preprocess.py       # ECG filtering, R-peak detection, beat extraction
    train_mitbih.py     # MIT-BIH download, SMOTE balancing, CNN training
    model.py            # 1D CNN architecture definition
    serial_ecg.py       # ESP32 serial reader
    mitbih_labels.py    # AAMI label mapping
  data/
    mitdb/              # Downloaded MIT-BIH records
  models/
    ecg_aami_cnn.keras
    ecg_aami_cnn.tflite
    label_map.json
    training_report.json
GUIDE.md
README.md
```

---

## Notes

- Classification uses ECG waveform morphology, not BPM alone.
- `Overall class` uses majority voting across all beats in the capture window.
- `Latest class` shows the most recently classified beat.
- Close Arduino Serial Monitor before using live mode — only one process can use a COM port at a time.
- For best signal quality, use adhesive ECG electrode pads rather than bare metal contact.

## Documentation

For full wiring details, firmware explanation, preprocessing pipeline, model architecture, training steps, dashboard code flow, and troubleshooting:

[GUIDE.md](GUIDE.md)
