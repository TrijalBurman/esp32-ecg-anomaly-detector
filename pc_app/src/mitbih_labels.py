from __future__ import annotations

from typing import Dict


LABELS = ["N", "S", "V", "F", "Q"]

SYMBOL_TO_AAMI: Dict[str, str] = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "?": "Q",
    "P": "Q",
}


def label_to_index() -> dict[str, int]:
    return {label: idx for idx, label in enumerate(LABELS)}


def index_to_label() -> dict[int, str]:
    return {idx: label for idx, label in enumerate(LABELS)}


def map_symbol(symbol: str) -> str | None:
    return SYMBOL_TO_AAMI.get(symbol)

