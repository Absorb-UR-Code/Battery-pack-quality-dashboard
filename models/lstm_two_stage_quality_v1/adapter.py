"""Two-stage LSTM inference adapter for the battery pack dashboard.

Stage 1 provides the operational OK/NG probability. Stage 2 is invoked only
when Stage 1 finds an anomalous 100-row window and supplies the fault type
probabilities consumed by the fault-log tab.
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
INFERENCE_BATCH_SIZE = 256
EXPECTED_CV_COUNT = 176
EXPECTED_TEMP_COUNT = 32
CLASS_NAMES = (
    "용량 불량",
    "용접·접촉 불량",
    "센싱와이어 불량",
    "온도 센서 불량",
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
            "2단계 LSTM은 셀 전압 176개와 온도 32개가 필요합니다. "
            f"현재 입력은 전압 {len(cv_columns)}개, 온도 {len(temp_columns)}개입니다."
        )
    return cv_columns, temp_columns


def _numeric_sensor_values(df: pd.DataFrame, columns: list[str], label: str) -> np.ndarray:
    numeric = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_missing = numeric.columns[numeric.isna().all()].tolist()
    if all_missing:
        preview = ", ".join(all_missing[:5])
        raise ValueError(f"{label} 센서가 전체 결측입니다: {preview}")
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


@lru_cache(maxsize=4)
def _load_assets(
    stage1_path: str,
    stage2_path: str,
    scaler_path: str,
    stage1_mtime: int,
    stage2_mtime: int,
    scaler_mtime: int,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    del stage1_mtime, stage2_mtime, scaler_mtime
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras

    stage1_model = keras.models.load_model(stage1_path, compile=False)
    stage2_model = keras.models.load_model(stage2_path, compile=False)
    with np.load(scaler_path, allow_pickle=False) as scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
    if mean.shape != (EXPECTED_CV_COUNT + EXPECTED_TEMP_COUNT,):
        raise ValueError(f"스케일러 평균 벡터 크기가 올바르지 않습니다: {mean.shape}")
    if std.shape != mean.shape or np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("스케일러 표준편차 벡터가 올바르지 않습니다.")
    return stage1_model, stage2_model, mean, std


def _assets(root: Path) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    stage1_path = root / "stage1_binary.keras"
    stage2_path = root / "stage2_fault_type.keras"
    scaler_path = root / "scaler.npz"
    for path in (stage1_path, stage2_path, scaler_path):
        if not path.exists():
            raise FileNotFoundError(f"모델 구성 파일이 없습니다: {path.name}")
    return _load_assets(
        str(stage1_path),
        str(stage2_path),
        str(scaler_path),
        stage1_path.stat().st_mtime_ns,
        stage2_path.stat().st_mtime_ns,
        scaler_path.stat().st_mtime_ns,
    )


def _predict_stage1_rows(model: Any, scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_count = len(scaled)
    scores = np.full(row_count, np.nan, dtype=float)
    endpoints = np.arange(WINDOW - 1, row_count, dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled,
        window_shape=WINDOW,
        axis=0,
    ).transpose(0, 2, 1)
    for batch_start in range(0, len(endpoints), INFERENCE_BATCH_SIZE):
        batch_endpoints = endpoints[batch_start : batch_start + INFERENCE_BATCH_SIZE]
        batch_windows = np.ascontiguousarray(
            windows[batch_start : batch_start + len(batch_endpoints)],
            dtype=np.float32,
        )
        batch_scores = np.asarray(model(batch_windows, training=False)).reshape(-1)
        scores[batch_endpoints] = batch_scores
    return scores, endpoints


def _candidate_sensors(
    raw_relative_window: np.ndarray,
    cv_columns: list[str],
    temp_columns: list[str],
    fault_type: str,
) -> list[str]:
    if fault_type == "온도 센서 불량":
        values = raw_relative_window[:, len(cv_columns) :]
        columns = temp_columns
    else:
        values = raw_relative_window[:, : len(cv_columns)]
        columns = cv_columns
    magnitude = np.nanmax(np.abs(values), axis=0)
    order = np.argsort(magnitude)[::-1][:5]
    return [columns[int(index)] for index in order if np.isfinite(magnitude[int(index)])]


def _fault_payload(
    type_probabilities: np.ndarray,
    raw_window: np.ndarray,
    cv_columns: list[str],
    temp_columns: list[str],
    window_end_row: int,
) -> dict[str, Any]:
    best_class = int(np.argmax(type_probabilities))
    fault_type = CLASS_NAMES[best_class]
    return {
        "fault_type": fault_type,
        "fault_confidence": float(type_probabilities[best_class]),
        "fault_probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(CLASS_NAMES, type_probabilities)
        },
        "suspect_sensors": _candidate_sensors(
            raw_window,
            cv_columns,
            temp_columns,
            fault_type,
        ),
        "severity": "검토 필요",
        "type_window_start_row": window_end_row - WINDOW + 1,
        "type_window_end_row": window_end_row,
    }


def predict(df: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    """Return row-length Stage-1 scores and file-level Stage-2 metadata."""
    threshold = float(context["spec"].get("threshold", 0.5))
    root = Path(context["spec"]["root"])
    relative, cv_columns, temp_columns = _relative_features(df)
    stage1_model, stage2_model, mean, std = _assets(root)
    scaled = ((relative - mean) / std).astype(np.float32, copy=False)

    if len(df) < WINDOW:
        scores = np.full(len(df), np.nan, dtype=float)
        return {
            "scores": scores,
            "threshold": threshold,
            "predictions": np.zeros(len(df), dtype=bool),
            "details": {
                "compatibility": 1.0,
                "insufficient_rows": True,
                "required_rows": WINDOW,
                "available_rows": len(df),
                "stage1_model_role": "정상·불량 이진 판정",
                "stage2_model_role": "불량 유형 분류",
            },
        }

    scores, endpoints = _predict_stage1_rows(stage1_model, scaled)
    predictions = np.isfinite(scores) & (scores >= threshold)
    valid_scores = scores[endpoints]
    best_position = int(np.nanargmax(valid_scores))
    best_endpoint = int(endpoints[best_position])
    best_probability = float(valid_scores[best_position])

    details: dict[str, Any] = {
        "compatibility": 1.0,
        "window_size": WINDOW,
        "warmup_rows": WINDOW - 1,
        "valid_window_count": int(len(endpoints)),
        "stage1_model_role": "정상·불량 이진 판정",
        "stage2_model_role": "불량 유형 분류",
        "binary_probability": best_probability,
        "binary_window_end_row": best_endpoint + 1,
    }

    if best_probability >= threshold:
        best_window = scaled[best_endpoint - WINDOW + 1 : best_endpoint + 1][None, ...]
        type_probabilities = np.asarray(stage2_model(best_window, training=False))[0]
        raw_window = relative[best_endpoint - WINDOW + 1 : best_endpoint + 1]
        details.update(
            _fault_payload(
                type_probabilities,
                raw_window,
                cv_columns,
                temp_columns,
                best_endpoint + 1,
            )
        )

        # 실시간 재생에서는 각 이상 구간이 처음 시작된 시점까지만 사용해
        # 유형을 분류한다. 파일 뒤쪽의 미래 윈도우가 초기 경보에 섞이지 않는다.
        valid_predictions = predictions[endpoints]
        run_start_positions = np.flatnonzero(
            valid_predictions & ~np.r_[False, valid_predictions[:-1]]
        )
        run_start_endpoints = endpoints[run_start_positions]
        all_windows = np.lib.stride_tricks.sliding_window_view(
            scaled,
            window_shape=WINDOW,
            axis=0,
        ).transpose(0, 2, 1)
        run_windows = np.ascontiguousarray(
            all_windows[run_start_positions],
            dtype=np.float32,
        )
        run_probabilities = np.asarray(stage2_model(run_windows, training=False))
        details["fault_by_row"] = {
            str(int(endpoint) + 1): _fault_payload(
                probabilities,
                relative[endpoint - WINDOW + 1 : endpoint + 1],
                cv_columns,
                temp_columns,
                int(endpoint) + 1,
            )
            for endpoint, probabilities in zip(run_start_endpoints, run_probabilities)
        }

    return {
        "scores": scores,
        "threshold": threshold,
        "predictions": predictions,
        "details": details,
    }
