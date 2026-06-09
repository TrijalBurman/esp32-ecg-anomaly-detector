from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import wfdb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from mitbih_labels import LABELS, label_to_index, map_symbol
from model import build_model
from preprocess import TARGET_BEAT_SAMPLES, bandpass_ecg, extract_beat_window


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "mitdb"
MODEL_DIR = BASE_DIR / "models"

RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def ensure_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "100.dat").exists():
        wfdb.dl_database("mitdb", dl_dir=str(DATA_DIR), records=RECORDS)


def choose_ecg_channel(record: wfdb.Record) -> int:
    names = [name.upper() for name in record.sig_name]
    for preferred in ("MLII", "II"):
        if preferred in names:
            return names.index(preferred)
    return 0


def load_beats() -> tuple[np.ndarray, np.ndarray]:
    ensure_dataset()
    x_beats: list[np.ndarray] = []
    y_labels: list[int] = []
    to_index = label_to_index()

    for record_name in RECORDS:
        record_path = str(DATA_DIR / record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")
        channel = choose_ecg_channel(record)
        fs = float(record.fs)

        ecg = record.p_signal[:, channel].astype(np.float32)
        filtered = bandpass_ecg(ecg, fs)

        for sample_index, symbol in zip(annotation.sample, annotation.symbol):
            label = map_symbol(symbol)
            if label is None:
                continue

            beat = extract_beat_window(filtered, int(sample_index), fs)
            if beat is None:
                continue

            x_beats.append(beat)
            y_labels.append(to_index[label])

    x = np.asarray(x_beats, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y_labels, dtype=np.int64)
    return x, y


def save_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    print("Loading MIT-BIH beats...")
    x, y = load_beats()
    print(f"Loaded {len(x)} beats with shape {x.shape[1:]}.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    model = build_model(num_classes=len(LABELS))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
            min_lr=1e-5,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    model.save(MODEL_DIR / "ecg_aami_cnn.keras")
    save_tflite(model, MODEL_DIR / "ecg_aami_cnn.tflite")

    (MODEL_DIR / "label_map.json").write_text(
        json.dumps({"labels": LABELS, "target_beat_samples": TARGET_BEAT_SAMPLES}, indent=2),
        encoding="utf-8",
    )
    (MODEL_DIR / "training_report.json").write_text(
        json.dumps({"classification_report": report, "confusion_matrix": matrix}, indent=2),
        encoding="utf-8",
    )

    print("Saved model artifacts to:", MODEL_DIR)


if __name__ == "__main__":
    main()

