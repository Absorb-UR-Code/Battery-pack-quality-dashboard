"""Inference adapter for the five-model LSTM autoencoder fault bank.

This package is intentionally registered as a validation candidate. The
supplied models were trained with Test05-Test09 events and derived
augmentations, so their Test01-Test09 replay scores are not independent
generalization evidence.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


WINDOW = 100
STRIDE = 20
INFERENCE_BATCH_SIZE = 256
EXPECTED_CV_COUNT = 176
EXPECTED_TEMP_COUNT = 32

CATEGORIES = (
    "정상",
    "용량 불량",
    "복합 불량",
    "센싱와이어 불량",
    "온도 센서 불량",
)
MODEL_FILES = (
    "lstm_ae_정상_v1.keras",
    "lstm_ae_용량불량_v1.keras",
    "lstm_ae_복합불량_v1.keras",
    "lstm_ae_센싱와이어불량_v1.keras",
    "lstm_ae_온도센서불량_v1.keras",
)


def _natural_sensor_key(column: str) -> tuple[int, ...]:
    return tuple(int(value) for value in re.findall(r"\d+", column))


def _sensor_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cv_columns = sorted(
        [column for column in df.columns if re.fullmatch(r"M\d{2}CV\d{2}", str(column))],
        key=_natural_sensor_key,
    )
    temp_columns = sorted(
        [column for column in df.columns if re.fullmatch(r"M\d{2}T\d{2}", str(column))],
        key=_natural_sensor_key,
    )
    if len(cv_columns) != EXPECTED_CV_COUNT or len(temp_columns) != EXPECTED_TEMP_COUNT:
        raise ValueError(
            "LSTM-AE 후보 모델은 셀 전압 176개와 온도 32개가 필요합니다. "
            f"현재 입력은 전압 {len(cv_columns)}개, 온도 {len(temp_columns)}개입니다."
        )
    return cv_columns, temp_columns


def _numeric_sensor_values(df: pd.DataFrame, columns: list[str], label: str) -> np.ndarray:
    numeric = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_missing = numeric.columns[numeric.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"{label} 센서가 전체 결측입니다: {', '.join(all_missing[:5])}")
    numeric = numeric.interpolate(axis=0, limit_direction="both").ffill().bfill()
    if numeric.isna().any().any():
        raise ValueError(f"{label} 센서 결측치를 보완하지 못했습니다.")
    return numeric.to_numpy(dtype=np.float32)


def _relative_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    cv_columns, temp_columns = _sensor_columns(df)
    cv_values = _numeric_sensor_values(df, cv_columns, "셀 전압")
    temp_values = _numeric_sensor_values(df, temp_columns, "온도")
    cv_relative = cv_values - cv_values.mean(axis=1, keepdims=True)
    temp_relative = temp_values - temp_values.mean(axis=1, keepdims=True)
    return np.concatenate([cv_relative, temp_relative], axis=1), cv_columns, temp_columns


@lru_cache(maxsize=2)
def _load_assets(
    root_text: str,
    asset_signature: tuple[int, ...],
) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
    del asset_signature
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras

    root = Path(root_text)
    model_paths = [root / file_name for file_name in MODEL_FILES]
    scaler_path = root / "scaler.npz"
    for path in [*model_paths, scaler_path]:
        if not path.exists():
            raise FileNotFoundError(f"LSTM-AE 구성 파일이 없습니다: {path.name}")

    models = tuple(keras.models.load_model(path, compile=False) for path in model_paths)
    with np.load(scaler_path, allow_pickle=False) as scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)

    expected_shape = (EXPECTED_CV_COUNT + EXPECTED_TEMP_COUNT,)
    if mean.shape != expected_shape:
        raise ValueError(f"스케일러 평균 벡터 크기가 올바르지 않습니다: {mean.shape}")
    if std.shape != expected_shape or np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("스케일러 표준편차 벡터가 올바르지 않습니다.")
    return models, mean, std


def _assets(root: Path) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
    paths = [root / file_name for file_name in MODEL_FILES] + [root / "scaler.npz"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"LSTM-AE 구성 파일이 없습니다: {path.name}")
    signature = tuple(path.stat().st_mtime_ns for path in paths)
    return _load_assets(str(root), signature)


def _reconstruction_errors(
    models: tuple[Any, ...],
    scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = len(scaled)
    endpoints = np.arange(WINDOW - 1, row_count, STRIDE, dtype=int)
    errors = np.empty((len(endpoints), len(models)), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled,
        window_shape=WINDOW,
        axis=0,
    ).transpose(0, 2, 1)[::STRIDE]

    for batch_start in range(0, len(endpoints), INFERENCE_BATCH_SIZE):
        batch_endpoints = endpoints[batch_start : batch_start + INFERENCE_BATCH_SIZE]
        batch_windows = np.ascontiguousarray(
            windows[batch_start : batch_start + len(batch_endpoints)],
            dtype=np.float32,
        )
        for model_index, model in enumerate(models):
            reconstruction = np.asarray(model(batch_windows, training=False))
            errors[batch_start : batch_start + len(batch_endpoints), model_index] = np.mean(
                (batch_windows - reconstruction) ** 2,
                axis=(1, 2),
            )
    return errors, endpoints


def _similarity_probabilities(errors: np.ndarray) -> np.ndarray:
    """Return display-only similarities; raw argmin remains the decision rule."""
    centered = errors - errors.min(axis=1, keepdims=True)
    scale = errors.std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    logits = -centered / scale
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _candidate_sensors(
    raw_relative_window: np.ndarray,
    cv_columns: list[str],
    temp_columns: list[str],
    fault_type: str,
) -> list[str]:
    cv_values = raw_relative_window[:, : len(cv_columns)]
    temp_values = raw_relative_window[:, len(cv_columns) :]

    def top(columns: list[str], values: np.ndarray, count: int) -> list[str]:
        magnitude = np.nanmax(np.abs(values), axis=0)
        order = np.argsort(magnitude)[::-1][:count]
        return [columns[int(index)] for index in order if np.isfinite(magnitude[int(index)])]

    if fault_type == "온도 센서 불량":
        return top(temp_columns, temp_values, 5)
    if fault_type == "복합 불량":
        return top(cv_columns, cv_values, 3) + top(temp_columns, temp_values, 2)
    return top(cv_columns, cv_values, 5)


def _fault_payload(
    class_index: int,
    probabilities: np.ndarray,
    raw_window: np.ndarray,
    cv_columns: list[str],
    temp_columns: list[str],
    window_end_row: int,
) -> dict[str, Any]:
    fault_type = CATEGORIES[class_index]
    return {
        "fault_type": fault_type,
        "fault_confidence": float(probabilities[class_index]),
        "fault_probabilities": {
            category: float(probability)
            for category, probability in zip(CATEGORIES[1:], probabilities[1:])
        },
        "suspect_sensors": _candidate_sensors(
            raw_window,
            cv_columns,
            temp_columns,
            fault_type,
        ),
        "severity": "검증 후보 판정",
        "type_window_start_row": window_end_row - WINDOW + 1,
        "type_window_end_row": window_end_row,
        "confidence_notice": "재구성오차 유사도를 확률처럼 정규화한 참고값이며 보정 확률이 아닙니다.",
    }


def predict(df: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    """Return operational-stride row predictions and report-compatible pack output."""
    threshold = float(context["spec"].get("threshold", 0.5))
    root = Path(context["spec"]["root"])
    relative, cv_columns, temp_columns = _relative_features(df)

    if len(df) < WINDOW:
        return {
            "scores": np.full(len(df), np.nan, dtype=float),
            "threshold": threshold,
            "predictions": np.zeros(len(df), dtype=bool),
            "details": {
                "compatibility": 1.0,
                "insufficient_rows": True,
                "required_rows": WINDOW,
                "available_rows": len(df),
                "decision_rule": "five-model raw reconstruction-error argmin",
            },
        }

    models, mean, std = _assets(root)
    scaled = ((relative - mean) / std).astype(np.float32, copy=False)
    errors, endpoints = _reconstruction_errors(models, scaled)
    probabilities = _similarity_probabilities(errors)
    class_indices = np.argmin(errors, axis=1)
    window_predictions = class_indices != 0
    winning_probabilities = probabilities[np.arange(len(probabilities)), class_indices]

    scores = np.full(len(df), np.nan, dtype=float)
    predictions = np.zeros(len(df), dtype=bool)
    scores[endpoints] = np.where(
        window_predictions,
        0.5 + (0.5 * winning_probabilities),
        0.5 * (1.0 - winning_probabilities),
    )
    predictions[endpoints] = window_predictions

    peak_position = int(np.argmax(errors[:, 0]))
    pack_class_index = int(class_indices[peak_position])
    pack_endpoint = int(endpoints[peak_position])
    pack_is_anomaly = pack_class_index != 0

    details: dict[str, Any] = {
        "compatibility": 1.0,
        "window_size": WINDOW,
        "warmup_rows": WINDOW - 1,
        "valid_window_count": int(len(endpoints)),
        "decision_rule": "정상 AE 오차 최대 윈도우에서 5개 AE raw argmin",
        "pack_class": CATEGORIES[pack_class_index],
        "pack_window_end_row": pack_endpoint + 1,
        "pack_normal_reconstruction_error": float(errors[peak_position, 0]),
        "probability_notice": "표시 확률은 재구성오차 기반 상대 유사도이며 보정 확률이 아닙니다.",
        "summary_override": {
            "status": "NG_REVIEW" if pack_is_anomaly else "NORMAL",
            "file_prediction": int(pack_is_anomaly),
            "trigger": "LSTM-AE raw argmin 파일 판정",
        },
    }

    if pack_is_anomaly:
        details.update(
            _fault_payload(
                pack_class_index,
                probabilities[peak_position],
                relative[pack_endpoint - WINDOW + 1 : pack_endpoint + 1],
                cv_columns,
                temp_columns,
                pack_endpoint + 1,
            )
        )

    run_start_positions = np.flatnonzero(
        window_predictions & ~np.r_[False, window_predictions[:-1]]
    )
    details["fault_by_row"] = {
        str(int(endpoints[position]) + 1): _fault_payload(
            int(class_indices[position]),
            probabilities[position],
            relative[
                endpoints[position] - WINDOW + 1 : endpoints[position] + 1
            ],
            cv_columns,
            temp_columns,
            int(endpoints[position]) + 1,
        )
        for position in run_start_positions
    }

    return {
        "scores": scores,
        "threshold": threshold,
        "predictions": predictions,
        "details": details,
    }
