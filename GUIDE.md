# ESP32 Heartbeat Anomaly Detector

This project builds an end-to-end heartbeat anomaly detector using:

- ESP32 development board
- AD8232 ECG front-end module
- Three ECG electrodes: `RA`, `LA`, `RL`
- UART serial streaming from ESP32 to PC
- MIT-BIH Arrhythmia Database for ML training
- Python, TensorFlow, SciPy, WFDB, Streamlit, and Plotly

The ESP32 reads the analog ECG waveform from the AD8232 and sends raw samples to the PC. The PC dashboard filters the ECG, detects R-peaks, calculates BPM, extracts each beat as a waveform window, and classifies the beat into one of five AAMI classes:

- `N`: normal and bundle branch beats
- `S`: supraventricular ectopic beats
- `V`: ventricular ectopic beats
- `F`: fusion beats
- `Q`: unknown, paced, or unclassifiable beats

Important: the model does not classify from BPM alone. BPM is useful for display, but `N/S/V/F/Q` classification needs ECG morphology, which means the shape of the waveform around each heartbeat.

This is an educational prototype. It is not a certified medical device and should not be used for diagnosis.

## Current Project Status

The project has already been set up and trained locally.

Generated files:

- `pc_app/models/ecg_aami_cnn.keras`: trained TensorFlow/Keras model for the dashboard
- `pc_app/models/ecg_aami_cnn.tflite`: exported TensorFlow Lite model
- `pc_app/models/label_map.json`: class labels and beat input size
- `pc_app/models/training_report.json`: test metrics and confusion matrix
- `pc_app/data/mitdb`: downloaded MIT-BIH records

Last training run result:

- Test accuracy: about `94.31%`
- Classes: `N`, `S`, `V`, `F`, `Q`
- Note: `S` and `F` are harder because they have fewer examples in MIT-BIH.

## High-Level Working

The complete data flow is:

```text
Body electrodes
  -> AD8232 ECG front-end
  -> ESP32 ADC on GPIO34
  -> UART CSV stream over USB
  -> Python serial reader
  -> ECG filtering and R-peak detection
  -> Beat window extraction
  -> TensorFlow CNN model
  -> Streamlit dashboard
```

The system has two operating modes:

1. `MIT-BIH preview sample`
   - Uses a stored MIT-BIH ECG segment.
   - Does not need ESP32 hardware.
   - Useful for checking the dashboard and trained model.

2. `Live ESP32 serial`
   - Reads real ECG samples from the ESP32 over a COM port.
   - Requires the AD8232 wiring and electrodes.
   - Shows live-ish capture windows and beat classifications.

## Hardware Required

- ESP32 development board
- AD8232 ECG sensor module
- Three ECG electrode pads or clips
- AD8232 electrode cable marked `RA`, `LA`, and `RL`
- USB cable for ESP32
- PC running Windows with Python environment installed
- Arduino IDE or PlatformIO for flashing the ESP32

## Wiring

### AD8232 To ESP32

| AD8232 pin | ESP32 pin | Purpose |
| --- | --- | --- |
| `OUTPUT` | `GPIO34` | Analog ECG signal into ESP32 ADC |
| `LO+` | `GPIO26` | Lead-off detection signal |
| `LO-` | `GPIO27` | Lead-off detection signal |
| `3.3V` | `3V3` | Sensor power |
| `GND` | `GND` | Common ground |

Use `3.3V`, not `5V`, for the AD8232 module.

### Electrode Placement

For your three-electrode setup:

| Electrode | Placement |
| --- | --- |
| `RA` | Right wrist |
| `LA` | Left wrist |
| `RL` | Right ankle |

`RA` and `LA` measure the ECG potential difference. `RL` is the reference/right-leg drive electrode and helps stabilize the signal.

### Step-By-Step Wiring

1. Disconnect the ESP32 from USB before wiring.
2. Connect AD8232 `GND` to ESP32 `GND`.
3. Connect AD8232 `3.3V` to ESP32 `3V3`.
4. Connect AD8232 `OUTPUT` to ESP32 `GPIO34`.
5. Connect AD8232 `LO+` to ESP32 `GPIO26`.
6. Connect AD8232 `LO-` to ESP32 `GPIO27`.
7. Attach `RA` electrode to the right wrist.
8. Attach `LA` electrode to the left wrist.
9. Attach `RL` electrode to the right ankle.
10. Connect ESP32 to the PC with USB.
11. Flash the ESP32 firmware.
12. Close Arduino Serial Monitor before opening the dashboard.

Only one program can use the ESP32 COM port at a time. If Arduino Serial Monitor is open, Streamlit cannot read the same COM port.

## Signal Notes

- AD8232 `OUTPUT` gives an ECG-like analog waveform, not BPM.
- ESP32 reads this waveform using its ADC.
- The PC calculates BPM after detecting R-peaks.
- `LO+` and `LO-` indicate whether electrodes are loose or disconnected.
- Noisy ECG usually comes from loose electrodes, dry skin, moving wires, or USB/power noise.
- If the waveform appears upside down, the dashboard still tries to handle it by checking signal polarity.

## Project Layout

```text
.
|-- README.md
|-- dashboard_preview.png
|-- firmware/
|   `-- esp32_ad8232_stream/
|       `-- esp32_ad8232_stream.ino
`-- pc_app/
    |-- requirements.txt
    |-- data/
    |   `-- mitdb/
    |-- models/
    |   |-- ecg_aami_cnn.keras
    |   |-- ecg_aami_cnn.tflite
    |   |-- label_map.json
    |   `-- training_report.json
    `-- src/
        |-- dashboard.py
        |-- mitbih_labels.py
        |-- model.py
        |-- preprocess.py
        |-- serial_ecg.py
        `-- train_mitbih.py
```

## Software Setup

The local environment was created under:

```text
C:\Users\Trijal\Documents\IOT Project\pc_app\.venv
```

Because the system Python is Python 3.14, this project used a Python 3.12 runtime for TensorFlow compatibility. To activate the environment:

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
.\.venv\Scripts\Activate.ps1
```

If you ever need to recreate the environment:

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If TensorFlow fails on Python 3.14, install/use Python 3.12 or 3.11.

## Python Dependencies

`pc_app/requirements.txt` contains:

- `numpy`: numerical arrays
- `scipy`: filtering, resampling, and peak detection
- `pandas`: table/data handling support
- `scikit-learn`: train/test split, class weights, metrics
- `tensorflow`: CNN model training and inference
- `wfdb`: MIT-BIH dataset download and reading
- `pyserial`: COM port reading from ESP32
- `streamlit`: dashboard UI
- `plotly`: interactive ECG chart

## ESP32 Firmware

Firmware file:

```text
firmware/esp32_ad8232_stream/esp32_ad8232_stream.ino
```

### Firmware Purpose

The ESP32 firmware samples the AD8232 signal at `250 Hz` and prints each sample as a CSV line over USB serial.

CSV format:

```text
millis,raw_adc,lead_off_plus,lead_off_minus
```

Example:

```text
15342,1845,0,0
15346,1851,0,0
15350,1839,0,0
```

Meaning:

- `millis`: ESP32 timestamp in milliseconds
- `raw_adc`: 12-bit ADC reading from GPIO34, from `0` to `4095`
- `lead_off_plus`: digital value from AD8232 `LO+`
- `lead_off_minus`: digital value from AD8232 `LO-`

### Firmware Code Explanation

Important constants:

```cpp
constexpr uint8_t ECG_PIN = 34;
constexpr uint8_t LEAD_OFF_PLUS_PIN = 26;
constexpr uint8_t LEAD_OFF_MINUS_PIN = 27;
constexpr uint32_t BAUD_RATE = 115200;
constexpr uint16_t SAMPLE_RATE_HZ = 250;
```

These define the wiring and serial speed.

In `setup()`:

- `Serial.begin(BAUD_RATE)` starts UART over USB.
- `pinMode(...)` configures ECG and lead-off pins.
- `analogReadResolution(12)` sets ADC values to `0..4095`.
- `analogSetPinAttenuation(ECG_PIN, ADC_11db)` allows a wider ADC voltage range.
- `nextSampleAt = micros()` initializes the sampling timer.

In `loop()`:

- The firmware waits until the next sampling time.
- It reads GPIO34 using `analogRead`.
- It reads `LO+` and `LO-` using `digitalRead`.
- It prints one CSV row to serial.

The firmware uses a timed sampling loop instead of `delay()` so the sample interval stays more consistent.

## Flashing The ESP32

1. Open Arduino IDE.
2. Install ESP32 board support.
3. Open:

```text
firmware/esp32_ad8232_stream/esp32_ad8232_stream.ino
```

4. Select board: `ESP32 Dev Module`.
5. Select the correct COM port.
6. Upload the sketch.
7. Open Serial Monitor only for quick testing.
8. Close Serial Monitor before running the Streamlit dashboard.

The Arduino sketch is already in the correct Arduino IDE folder format:

```text
firmware/esp32_ad8232_stream/esp32_ad8232_stream.ino
```

Arduino IDE requires the `.ino` file and parent folder to have the same name. This project already follows that rule.

Inside the sketch, this line controls whether a CSV header is printed:

```cpp
const bool PRINT_CSV_HEADER = false;
```

Keep it `false` for the cleanest dashboard serial stream. You can set it to `true` while testing in Arduino Serial Monitor.

Arduino IDE board manager URL:

```text
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```

### Compile With Arduino CLI

Arduino CLI can also compile the firmware without opening Arduino IDE.

Install/update the ESP32 core:

```powershell
arduino-cli core update-index
arduino-cli core install esp32:esp32
```

Compile the sketch:

```powershell
arduino-cli compile --fqbn esp32:esp32:esp32 firmware\esp32_ad8232_stream
```

This project was successfully compiled using Arduino CLI `1.5.1` and ESP32 core `3.3.10`.

## Machine Learning Dataset

The model is trained on the MIT-BIH Arrhythmia Database using the `wfdb` Python package.

The training script downloads records into:

```text
pc_app/data/mitdb
```

MIT-BIH provides:

- ECG signal files such as `.dat`
- Header files such as `.hea`
- Annotation files such as `.atr`

The annotation files contain beat locations and symbols. These symbols are mapped into the five AAMI classes.

## Label Mapping

File:

```text
pc_app/src/mitbih_labels.py
```

The dashboard and training code both use:

```python
LABELS = ["N", "S", "V", "F", "Q"]
```

MIT-BIH symbols are mapped like this:

| AAMI class | MIT-BIH symbols used |
| --- | --- |
| `N` | `N`, `L`, `R`, `e`, `j` |
| `S` | `A`, `a`, `J`, `S` |
| `V` | `V`, `E` |
| `F` | `F` |
| `Q` | `/`, `f`, `Q`, `?`, `P` |

Functions:

- `label_to_index()`: converts labels like `N` into numeric classes like `0`
- `index_to_label()`: converts numeric classes back to labels
- `map_symbol(symbol)`: converts MIT-BIH annotation symbols to AAMI labels

## ECG Preprocessing

File:

```text
pc_app/src/preprocess.py
```

This file contains the shared signal processing code used by both training and live inference.

### `bandpass_ecg(samples, fs)`

Applies a Butterworth band-pass filter from `0.5 Hz` to `40 Hz`.

Purpose:

- Reduce baseline drift
- Reduce high-frequency noise
- Keep the main ECG morphology useful for R-peak detection and classification

### `detect_r_peaks(samples, fs)`

Detects likely R-peaks using `scipy.signal.find_peaks`.

It:

- Filters the ECG
- Centers the signal around its median
- Flips the signal if peaks appear inverted
- Uses minimum peak distance of about `0.25 seconds`
- Uses prominence based on signal standard deviation

Output:

```python
np.ndarray
```

containing sample indices where heartbeats were detected.

### `bpm_from_peaks(peaks, fs)`

Calculates BPM from R-R intervals.

It:

- Finds differences between consecutive peaks
- Converts sample differences to seconds
- Rejects impossible intervals outside `0.25` to `2.5` seconds
- Uses the median R-R interval
- Converts it to BPM using `60 / RR`

### `extract_beat_window(samples, peak_index, fs)`

Extracts a beat-centered ECG segment:

- `0.35 seconds` before the R-peak
- `0.45 seconds` after the R-peak

Then it:

- Resamples every beat to `256` samples
- Normalizes it by subtracting mean and dividing by standard deviation

This is important because the CNN expects every beat to have the same shape:

```text
256 samples x 1 channel
```

## ML Model Architecture

File:

```text
pc_app/src/model.py
```

The model is a compact 1D CNN for ECG beat classification.

Input:

```text
(256, 1)
```

Architecture:

```text
Input ECG beat
  -> Conv1D(32 filters, kernel 7, ReLU)
  -> BatchNormalization
  -> MaxPooling1D
  -> Conv1D(64 filters, kernel 5, ReLU)
  -> BatchNormalization
  -> MaxPooling1D
  -> Conv1D(128 filters, kernel 3, ReLU)
  -> BatchNormalization
  -> GlobalAveragePooling1D
  -> Dropout(0.3)
  -> Dense(5, softmax)
```

Output:

```text
[p_N, p_S, p_V, p_F, p_Q]
```

The class with the highest probability becomes the predicted beat class.

The model uses:

- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Metric: accuracy

## Training Pipeline

File:

```text
pc_app/src/train_mitbih.py
```

Run training:

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
.\.venv\Scripts\Activate.ps1
python src\train_mitbih.py
```

### Training Steps

1. Create `pc_app/data/mitdb` if needed.
2. Download MIT-BIH records if `100.dat` is not present.
3. Read each ECG record with `wfdb.rdrecord`.
4. Read annotations with `wfdb.rdann`.
5. Choose ECG channel:
   - Prefer `MLII`
   - Then `II`
   - Otherwise use channel `0`
6. Band-pass filter the ECG.
7. For each annotation:
   - Convert MIT-BIH symbol to AAMI class
   - Extract the ECG beat window around the annotated beat
   - Resample to `256` samples
   - Normalize the beat
8. Split into train and test sets using stratified split.
9. Compute class weights to reduce class imbalance problems.
10. Train the 1D CNN.
11. Evaluate on the test set.
12. Save the model and reports.

### Training Outputs

Saved under:

```text
pc_app/models
```

Files:

- `ecg_aami_cnn.keras`: Keras model used by Streamlit
- `ecg_aami_cnn.tflite`: TensorFlow Lite model for possible embedded/mobile use
- `label_map.json`: label list and beat input size
- `training_report.json`: precision, recall, F1 score, accuracy, confusion matrix

## Serial Reading Code

File:

```text
pc_app/src/serial_ecg.py
```

This module handles communication with the ESP32.

### `EcgSample`

A dataclass representing one serial sample:

```python
EcgSample(
    millis: int,
    raw_adc: int,
    lead_off_plus: int,
    lead_off_minus: int,
)
```

### `available_ports()`

Lists COM ports available on the PC.

The dashboard uses this to populate the port dropdown.

### `parse_line(line)`

Parses one ESP32 CSV line:

```text
millis,raw_adc,lead_off_plus,lead_off_minus
```

If a line is malformed, it returns `None` instead of crashing.

### `read_samples(port, baud, seconds)`

Opens the selected COM port and reads samples for a fixed time window.

For example, if the capture window is `10` seconds, it reads serial data for about `10` seconds and returns a list of `EcgSample` objects.

### `samples_to_arrays(samples)`

Converts serial samples into NumPy arrays:

- `millis`: timestamps
- `volts`: ADC converted to voltage using `(raw / 4095) * 3.3`
- `lead_off`: Boolean array showing whether either lead-off pin was active

## Dashboard

File:

```text
pc_app/src/dashboard.py
```

Run:

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
.\.venv\Scripts\Activate.ps1
streamlit run src\dashboard.py
```

Then open:

```text
http://localhost:8501
```

### Dashboard Modes

#### MIT-BIH Preview Sample

This mode loads a short ECG segment from the downloaded MIT-BIH dataset.

Use this mode to:

- Preview the dashboard
- Check that the model loads
- Verify charts and classification without ESP32 hardware

Workflow:

1. Select `MIT-BIH preview sample`.
2. Choose a MIT-BIH record such as `100`.
3. Choose preview start second.
4. Click `Run MIT-BIH preview`.

#### Live ESP32 Serial

This mode reads real ECG from the ESP32.

Workflow:

1. Flash ESP32 firmware.
2. Connect the AD8232 and electrodes.
3. Close Arduino Serial Monitor.
4. Start the dashboard.
5. Select `Live ESP32 serial`.
6. Select the ESP32 COM port.
7. Set baud to `115200`.
8. Click `Start live classification`.

### Dashboard Code Flow

1. `load_classifier()` loads:
   - `models/ecg_aami_cnn.keras`
   - `models/label_map.json`
2. Sidebar controls select:
   - data source mode
   - serial port
   - baud rate
   - capture duration
   - MIT-BIH preview record
3. If preview mode is selected:
   - `load_preview_segment()` reads ECG from MIT-BIH
4. If live mode is selected:
   - `read_samples()` reads serial data from ESP32
   - `samples_to_arrays()` converts ADC readings to volts
   - `estimate_sample_rate()` estimates actual sample rate from ESP32 timestamps
5. The ECG is filtered with `bandpass_ecg()`.
6. R-peaks are detected with `detect_r_peaks()`.
7. BPM is calculated using `bpm_from_peaks()`.
8. For every detected R-peak:
   - `extract_beat_window()` extracts the beat segment
   - The model predicts class probabilities
   - The dashboard stores time, class, confidence, and per-class probabilities
9. Plotly displays ECG and R-peak markers.
10. Streamlit displays:
   - BPM
   - overall class based on the majority of classified beats in the capture window
   - agreement percentage for the overall class
   - overall median confidence
   - latest class
   - latest confidence
   - sample rate in the status caption
   - lead-off percentage
   - classification table

The `Overall class` is intended to be the stable result for the full capture window. It uses majority voting across detected beats, with summed confidence as a tie-breaker. One or two incorrect beat predictions therefore do not normally change the overall label. `Latest class` remains available to show the most recently detected beat.

## How All Files Work Together

```text
esp32_ad8232_stream.ino
  sends CSV serial ECG data

serial_ecg.py
  reads CSV data from ESP32 and converts ADC values to volts

preprocess.py
  filters ECG, detects R-peaks, extracts normalized beat windows

mitbih_labels.py
  maps MIT-BIH annotation symbols to N/S/V/F/Q classes

model.py
  defines the CNN architecture

train_mitbih.py
  downloads MIT-BIH, prepares beat windows, trains the CNN, saves model files

dashboard.py
  loads model, reads live or preview ECG, preprocesses beats, predicts classes, displays results
```

In short:

- The firmware creates the live ECG stream.
- The training script creates the trained model.
- The preprocessing module keeps training and live inference consistent.
- The dashboard connects everything for user interaction.

## Running The Complete Project

### 1. Flash ESP32

Open and upload:

```text
firmware/esp32_ad8232_stream/esp32_ad8232_stream.ino
```

### 2. Activate Python Environment

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
.\.venv\Scripts\Activate.ps1
```

### 3. Train Model

Only needed if model files are missing or you want to retrain:

```powershell
python src\train_mitbih.py
```

### 4. Run Dashboard

```powershell
streamlit run src\dashboard.py
```

### 5. Test Preview Mode

1. Select `MIT-BIH preview sample`.
2. Choose record `100`.
3. Click `Run MIT-BIH preview`.

### 6. Test Live Mode

1. Select `Live ESP32 serial`.
2. Pick the ESP32 COM port.
3. Confirm baud is `115200`.
4. Click `Start live classification`.

## Dashboard Preview

A preview image is saved here:

```text
dashboard_preview.png
```

It shows the dashboard running with a MIT-BIH preview sample, ECG plot, detected R-peaks, metrics, and classification output.

## Model Performance Notes

The final local training report is saved in:

```text
pc_app/models/training_report.json
```

The model performs well on common classes such as `N`, `V`, and `Q`. `S` and `F` are less reliable because there are fewer examples and these classes are more subtle.

For a serious research-quality model, improvements could include:

- Patient-wise train/test split
- More robust R-peak detection
- Data augmentation
- Better handling of class imbalance
- Larger CNN or CNN-LSTM architecture
- Calibration of probabilities
- Validation on live AD8232-collected data

## Troubleshooting

### Dashboard says model not found

Run:

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
.\.venv\Scripts\Activate.ps1
python src\train_mitbih.py
```

### No serial ports found

Check:

- ESP32 is connected by USB
- USB cable supports data, not only charging
- ESP32 driver is installed
- Arduino Serial Monitor is closed

### Port opens but no data arrives

Check:

- Correct firmware is flashed
- Baud is `115200`
- Correct COM port is selected
- ESP32 is not stuck in bootloader mode

### Lead-off percentage is high

Check:

- Electrode pads are attached firmly
- `RA`, `LA`, and `RL` cables are connected
- Skin contact is good
- Electrode gel/pads are not dry

### ECG is noisy

Try:

- Keep wires still
- Sit still during capture
- Use fresh electrode pads
- Clean skin before attaching electrodes
- Use a shorter USB cable
- Run laptop on battery
- Keep the sensor away from laptop chargers and power adapters

### BPM looks wrong

Possible reasons:

- R-peaks are not detected correctly
- Signal is too noisy
- Electrode placement is unstable
- The capture window is too short

Try increasing the capture window to `15` or `20` seconds.

### Classification looks wrong on live data

Possible reasons:

- Live AD8232 signal quality differs from MIT-BIH clinical ECG recordings
- R-peak detection is selecting wrong peaks
- Electrodes are loose
- The model is trained on MIT-BIH morphology, not personalized data

Use MIT-BIH preview mode first to confirm the software path is working.

## Safety Notes

- This is not a medical device.
- Do not use it to diagnose disease.
- Do not connect the AD8232 or ESP32 to mains-powered experimental circuits.
- Power the ESP32 from USB only.
- Do not use damaged electrodes or wires.
- Stop testing if there is discomfort.

## Quick Command Reference

Activate environment:

```powershell
cd "C:\Users\Trijal\Documents\IOT Project\pc_app"
.\.venv\Scripts\Activate.ps1
```

Train:

```powershell
python src\train_mitbih.py
```

Run dashboard:

```powershell
streamlit run src\dashboard.py
```

Expected ESP32 serial format:

```text
millis,raw_adc,lead_off_plus,lead_off_minus
```

Expected model input:

```text
256 ECG samples x 1 channel
```

Expected model output:

```text
N, S, V, F, Q probabilities
```
