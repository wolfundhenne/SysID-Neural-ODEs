"""Small helpers for explicit Optuna search-space definitions."""

from __future__ import annotations

from typing import Any


def categorical(choices: list[Any]) -> dict[str, Any]:
    return {"type": "categorical", "choices": choices}


def log_float(
    low: float,
    high: float,
) -> dict[str, Any]:
    return {"type": "float", "low": low, "high": high, "log": True}


def linear_float(
    low: float,
    high: float,
) -> dict[str, Any]:
    return {"type": "float", "low": low, "high": high, "log": False}


def integer(
    low: int,
    high: int,
    step: int = 1,
    log: bool = False,
) -> dict[str, Any]:
    return {"type": "int", "low": low, "high": high, "step": step, "log": log}


def fixed(value: Any) -> dict[str, Any]:
    return {"type": "fixed", "value": value}
