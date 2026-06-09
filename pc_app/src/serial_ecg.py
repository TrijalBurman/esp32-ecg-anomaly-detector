from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np
import serial
from serial.tools import list_ports


@dataclass
class EcgSample:
    millis: int
    raw_adc: int
    lead_off_plus: int
    lead_off_minus: int


def available_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def parse_line(line: bytes) -> EcgSample | None:
    try:
        text = line.decode("ascii", errors="ignore").strip()
        millis, raw, lo_plus, lo_minus = text.split(",")
        return EcgSample(int(millis), int(raw), int(lo_plus), int(lo_minus))
    except ValueError:
        return None


def read_samples(port: str, baud: int, seconds: float) -> list[EcgSample]:
    samples: list[EcgSample] = []
    deadline = monotonic() + seconds

    with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
        ser.reset_input_buffer()
        while monotonic() < deadline:
            sample = parse_line(ser.readline())
            if sample is not None:
                samples.append(sample)

    return samples


def samples_to_arrays(samples: list[EcgSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    millis = np.array([sample.millis for sample in samples], dtype=np.float32)
    raw = np.array([sample.raw_adc for sample in samples], dtype=np.float32)
    lead_off = np.array(
        [sample.lead_off_plus or sample.lead_off_minus for sample in samples],
        dtype=bool,
    )

    volts = (raw / 4095.0) * 3.3
    return millis, volts, lead_off

