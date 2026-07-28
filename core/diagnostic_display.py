from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_catalog import detect_sensor_columns, read_csv_resilient
from .fault_log import extract_fault_metadata
from .features import build_sensor_kpis


KPI_SPECS: dict[str, dict[str, Any]] = {
    "셀 전압 평균 (V)": {
        "metric": "cv_mean",
        "domain": "voltage",
        "digits": 4,
        "unit": "V",
    },
    "셀 전압 편차 (V)": {
        "metric": "cv_std",
        "domain": "voltage",
        "digits": 5,
        "unit": "V",
    },
    "온도 평균 (°C)": {
        "metric": "temp_mean",
        "domain": "temperature",
        "digits": 2,
        "unit": "°C",
    },
    "온도 범위 (°C)": {
        "metric": "temp_range",
        "domain": "temperature",
        "digits": 2,
        "unit": "°C",
    },
    "온도 편차 (°C)": {
        "metric": "temp_std",
        "domain": "temperature",
        "digits": 3,
        "unit": "°C",
    },
}

NORMAL_CELL_STYLE = (
    "background-color:#e9f7ef;color:#155b3d;font-weight:650;white-space:nowrap;"
)
ALERT_CELL_STYLE = (
    "background-color:#fde8e8;color:#8f2424;font-weight:750;white-space:nowrap;"
)
ALERT_UP_STYLE = (
    "background-color:#fde8e8;color:#087a55;font-weight:800;white-space:nowrap;"
)
ALERT_DOWN_STYLE = (
    "background-color:#fde8e8;color:#b42318;font-weight:800;white-space:nowrap;"
)


def _expanded_bounds(low: float, high: float, minimum_half_width: float) -> dict[str, float]:
    if not np.isfinite(low) or not np.isfinite(high):
        return {"low": np.nan, "high": np.nan}
    center = (float(low) + float(high)) / 2.0
    half_width = max((float(high) - float(low)) / 2.0, float(minimum_half_width))
    return {"low": center - half_width, "high": center + half_width}


def _summarize_reference(
    sensor_parts: list[pd.DataFrame],
    kpi_parts: list[pd.DataFrame],
) -> dict[str, dict[str, dict[str, float]]]:
    sensor_reference: dict[str, dict[str, float]] = {}
    kpi_reference: dict[str, dict[str, float]] = {}

    if sensor_parts:
        sensor_values = pd.concat(sensor_parts, ignore_index=True, sort=False)
        quantiles = sensor_values.quantile([0.01, 0.99], numeric_only=True)
        for column in quantiles.columns:
            low = float(quantiles.loc[0.01, column])
            high = float(quantiles.loc[0.99, column])
            minimum_half_width = 0.002 if "CV" in str(column).upper() else 0.5
            sensor_reference[str(column)] = _expanded_bounds(low, high, minimum_half_width)

    if kpi_parts:
        kpi_values = pd.concat(kpi_parts, ignore_index=True, sort=False)
        quantiles = kpi_values.quantile([0.01, 0.99], numeric_only=True)
        minimum_widths = {
            "cv_mean": 0.002,
            "cv_std": 0.0005,
            "temp_mean": 0.5,
            "temp_range": 0.5,
            "temp_std": 0.2,
        }
        for metric in minimum_widths:
            if metric not in quantiles.columns:
                continue
            low = float(quantiles.loc[0.01, metric])
            high = float(quantiles.loc[0.99, metric])
            kpi_reference[metric] = _expanded_bounds(low, high, minimum_widths[metric])

    return {"sensor": sensor_reference, "kpi": kpi_reference}


def build_normal_reference(
    records: list[dict[str, Any]],
    max_rows_per_file: int = 800,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Build robust CHG/DCHG reference bands from Train sensor observations."""
    sensor_parts: dict[str, list[pd.DataFrame]] = {"CHG": [], "DCHG": [], "ALL": []}
    kpi_parts: dict[str, list[pd.DataFrame]] = {"CHG": [], "DCHG": [], "ALL": []}

    for record in records:
        path = Path(str(record.get("path", "")))
        if not path.exists() or not path.is_file():
            continue
        try:
            frame = read_csv_resilient(path)
        except Exception:
            continue
        if frame.empty:
            continue

        if len(frame) > int(max_rows_per_file):
            sample_index = np.linspace(0, len(frame) - 1, int(max_rows_per_file), dtype=int)
            frame = frame.iloc[np.unique(sample_index)].copy()

        cv_cols, temp_cols = detect_sensor_columns(frame.columns)
        sensor_cols = cv_cols + temp_cols
        if not sensor_cols:
            continue
        numeric = frame[sensor_cols].apply(pd.to_numeric, errors="coerce")
        residual_parts = []
        if cv_cols:
            cv_values = numeric[cv_cols]
            residual_parts.append(cv_values.sub(cv_values.median(axis=1), axis=0))
        if temp_cols:
            temp_values = numeric[temp_cols]
            residual_parts.append(temp_values.sub(temp_values.median(axis=1), axis=0))
        sensor_residual = pd.concat(residual_parts, axis=1)
        kpis = build_sensor_kpis(frame)[
            ["cv_mean", "cv_std", "temp_mean", "temp_range", "temp_std"]
        ]
        mode = str(record.get("mode", "UNKNOWN")).upper()
        if mode not in {"CHG", "DCHG"}:
            mode = "ALL"
        sensor_parts[mode].append(sensor_residual)
        kpi_parts[mode].append(kpis)
        if mode != "ALL":
            sensor_parts["ALL"].append(sensor_residual)
            kpi_parts["ALL"].append(kpis)

    return {
        mode: _summarize_reference(sensor_parts[mode], kpi_parts[mode])
        for mode in ("CHG", "DCHG", "ALL")
    }


def reference_for_mode(
    reference: dict[str, Any],
    mode: str,
) -> dict[str, dict[str, dict[str, float]]]:
    selected = reference.get(str(mode).upper(), {}) if isinstance(reference, dict) else {}
    if selected.get("sensor") or selected.get("kpi"):
        return selected
    return reference.get("ALL", {"sensor": {}, "kpi": {}}) if isinstance(reference, dict) else {
        "sensor": {},
        "kpi": {},
    }


def _fault_domains(payload: dict[str, Any]) -> set[str]:
    metadata = extract_fault_metadata({"details": payload})
    fault_type = str(metadata.get("fault_type", "")).lower()
    sensor_text = str(metadata.get("suspect_sensors", "")).upper()
    domains: set[str] = set()

    if any(token in fault_type for token in ["온도", "열 관리", "temperature", "thermal"]):
        domains.add("temperature")
    if any(
        token in fault_type
        for token in [
            "전압",
            "용량",
            "저항",
            "용접",
            "접촉",
            "센싱와이어",
            "voltage",
            "capacity",
            "resistance",
            "weld",
            "wire",
        ]
    ):
        domains.add("voltage")
    if "CV" in sensor_text:
        domains.add("voltage")
    if any(f"T{index:02d}" in sensor_text for index in range(1, 3)):
        domains.add("temperature")
    if "센서 불량" in fault_type and not domains:
        domains.update({"voltage", "temperature"})
    return domains


def latest_scored_prediction(
    row_result: pd.DataFrame | None,
    position: int,
) -> tuple[bool, float, int | None]:
    """Return the latest evaluated row at or before the live playback cursor."""
    if not isinstance(row_result, pd.DataFrame) or row_result.empty:
        return False, float("nan"), None
    if "predicted_anomaly" not in row_result.columns:
        return False, float("nan"), None

    visible_end = max(0, min(int(position), len(row_result)))
    if visible_end == 0:
        return False, float("nan"), None

    visible = row_result.iloc[:visible_end]
    if "score" in visible.columns:
        scores = pd.to_numeric(visible["score"], errors="coerce")
        evaluated = scores.notna().to_numpy()
    else:
        scores = pd.Series(np.nan, index=visible.index, dtype=float)
        evaluated = np.ones(len(visible), dtype=bool)

    evaluated_positions = np.flatnonzero(evaluated)
    if not len(evaluated_positions):
        return False, float("nan"), None

    latest_position = int(evaluated_positions[-1])
    raw_flag = visible["predicted_anomaly"].iloc[latest_position]
    if pd.isna(raw_flag):
        flag = False
    elif isinstance(raw_flag, str):
        flag = raw_flag.strip().casefold() in {"1", "true", "yes", "y", "ng", "anomaly"}
    else:
        flag = bool(raw_flag)

    score = pd.to_numeric(pd.Series([scores.iloc[latest_position]]), errors="coerce").iloc[0]
    return flag, float(score) if np.isfinite(score) else float("nan"), latest_position + 1


def evaluated_row_positions(
    row_result: pd.DataFrame | None,
    row_count: int,
    *,
    end_position: int | None = None,
) -> np.ndarray:
    """Return zero-based source rows for which the model produced a finite score."""
    row_count = max(0, int(row_count))
    if not isinstance(row_result, pd.DataFrame) or row_result.empty or row_count == 0:
        return np.asarray([], dtype=int)

    if "score" in row_result.columns:
        evaluated = pd.to_numeric(row_result["score"], errors="coerce").notna().to_numpy()
    elif "predicted_anomaly" in row_result.columns:
        evaluated = row_result["predicted_anomaly"].notna().to_numpy()
    else:
        return np.asarray([], dtype=int)

    if "row_index" in row_result.columns:
        source_positions = (
            pd.to_numeric(row_result["row_index"], errors="coerce").to_numpy(dtype=float) - 1
        )
    else:
        source_positions = np.arange(len(row_result), dtype=float)

    valid = evaluated & np.isfinite(source_positions)
    positions = source_positions[valid].astype(int, copy=False)
    positions = positions[(positions >= 0) & (positions < row_count)]
    if end_position is not None:
        positions = positions[positions < max(0, min(int(end_position), row_count))]
    return np.unique(positions)


def reached_fault_payloads(
    result: dict[str, Any] | None,
    end_position: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Return fault-run payloads whose detection rows have reached the cursor."""
    if not result:
        return []

    visible_end = max(0, int(end_position))
    details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
    fault_by_row = details.get("fault_by_row", {})
    explicit: list[tuple[int, dict[str, Any]]] = []
    if isinstance(fault_by_row, dict):
        for row_key, payload in fault_by_row.items():
            try:
                row_number = int(float(str(row_key)))
            except (TypeError, ValueError):
                continue
            if 1 <= row_number <= visible_end and isinstance(payload, dict):
                explicit.append((row_number, payload))
    if explicit:
        return sorted(explicit, key=lambda item: item[0])

    row_result = result.get("row_result")
    if not isinstance(row_result, pd.DataFrame) or "predicted_anomaly" not in row_result:
        return []

    visible = row_result.iloc[: min(visible_end, len(row_result))]
    if visible.empty:
        return []
    if "score" in visible.columns:
        scores = pd.to_numeric(visible["score"], errors="coerce")
        evaluated_positions = np.flatnonzero(scores.notna().to_numpy())
    else:
        evaluated_positions = np.arange(len(visible), dtype=int)
    if not len(evaluated_positions):
        return []

    evaluated_flags = (
        visible["predicted_anomaly"]
        .iloc[evaluated_positions]
        .fillna(0)
        .astype(bool)
        .to_numpy()
    )
    run_start_mask = evaluated_flags & ~np.r_[False, evaluated_flags[:-1]]
    run_start_rows = evaluated_positions[np.flatnonzero(run_start_mask)] + 1
    return [(int(row_number), details) for row_number in run_start_rows]


def latest_fault_payload(
    result: dict[str, Any] | None,
    end_position: int,
) -> tuple[int | None, dict[str, Any]]:
    """Return the most recent fault-run payload visible at the cursor."""
    payloads = reached_fault_payloads(result, end_position)
    if payloads:
        return payloads[-1]
    details = result.get("details", {}) if isinstance(result, dict) else {}
    return None, details if isinstance(details, dict) else {}


def fault_domain_coverage(
    result: dict[str, Any] | None,
    row_count: int,
    end_position: int | None = None,
) -> pd.DataFrame:
    """Propagate each Stage-2 type over every row covered by its anomalous window."""
    row_count = max(0, int(row_count))
    empty = pd.DataFrame(
        {
            "voltage_fault": np.zeros(row_count, dtype=bool),
            "temperature_fault": np.zeros(row_count, dtype=bool),
        }
    )
    if not result or row_count == 0:
        return empty

    row_result = result.get("row_result")
    if not isinstance(row_result, pd.DataFrame) or "predicted_anomaly" not in row_result:
        return empty
    flags = row_result["predicted_anomaly"].fillna(0).astype(bool).to_numpy()[:row_count].copy()
    if len(flags) < row_count:
        flags = np.pad(flags, (0, row_count - len(flags)), constant_values=False)
    if end_position is not None:
        visible_end = max(0, min(int(end_position), row_count))
        flags[visible_end:] = False
    if not flags.any():
        return empty

    details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
    window_size = max(1, int(details.get("window_size", 100)))
    default_payload = extract_fault_metadata(result)
    visible_end = row_count if end_position is None else max(0, min(int(end_position), row_count))
    payload_entries = reached_fault_payloads(result, visible_end)
    payload_rows = np.asarray([row for row, _ in payload_entries], dtype=int)
    diff = {
        "voltage": np.zeros(row_count + 1, dtype=int),
        "temperature": np.zeros(row_count + 1, dtype=int),
    }

    for endpoint in np.flatnonzero(flags):
        endpoint_row = int(endpoint) + 1
        payload = default_payload
        if len(payload_rows):
            payload_index = int(np.searchsorted(payload_rows, endpoint_row, side="right") - 1)
            if payload_index >= 0:
                payload = payload_entries[payload_index][1]
        domains = _fault_domains(payload if isinstance(payload, dict) else default_payload)
        if not domains:
            domains = _fault_domains(default_payload)
        if not domains:
            continue
        covered_start = max(0, int(endpoint) - window_size + 1)
        covered_end = min(row_count - 1, int(endpoint))
        for domain in domains:
            diff[domain][covered_start] += 1
            diff[domain][covered_end + 1] -= 1

    return pd.DataFrame(
        {
            "voltage_fault": np.cumsum(diff["voltage"][:-1]) > 0,
            "temperature_fault": np.cumsum(diff["temperature"][:-1]) > 0,
        }
    )


def _adjustment(
    value: float,
    bounds: dict[str, float] | None,
    digits: int,
    unit: str,
    include_unit: bool = True,
) -> tuple[str, str]:
    if not np.isfinite(value):
        return " · 결측", "down"
    if not bounds:
        return "", "inside"
    low = float(bounds.get("low", np.nan))
    high = float(bounds.get("high", np.nan))
    unit_text = f" {unit}" if include_unit else ""
    if np.isfinite(low) and value < low:
        return f" ▲(+{low - value:.{digits}f}{unit_text})", "up"
    if np.isfinite(high) and value > high:
        return f" ▼(-{value - high:.{digits}f}{unit_text})", "down"
    return "", "inside"


def kpi_log_styler(
    log_frame: pd.DataFrame,
    kpi_frame: pd.DataFrame,
    source_positions: np.ndarray | list[int],
    domain_coverage: pd.DataFrame,
    normal_reference: dict[str, Any],
    mode: str,
) -> pd.io.formats.style.Styler:
    display = log_frame.copy().astype(object)
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    positions = np.asarray(source_positions, dtype=int)
    kpis = kpi_frame.reset_index(drop=True)
    reference = reference_for_mode(normal_reference, mode)
    kpi_reference = reference.get("kpi", {})

    for display_row, source_position in enumerate(positions):
        if source_position < 0 or source_position >= len(domain_coverage):
            continue
        for column, spec in KPI_SPECS.items():
            if column not in display.columns or spec["metric"] not in kpis.columns:
                continue
            domain_active = bool(
                domain_coverage.iloc[source_position][f"{spec['domain']}_fault"]
            )
            if not domain_active:
                continue
            value = pd.to_numeric(
                pd.Series([kpis.iloc[display_row][spec["metric"]]]), errors="coerce"
            ).iloc[0]
            adjustment, direction = _adjustment(
                float(value) if pd.notna(value) else np.nan,
                kpi_reference.get(spec["metric"]),
                int(spec["digits"]),
                str(spec["unit"]),
                include_unit=False,
            )
            if spec["metric"] == "temp_range":
                base_text = str(display.iloc[display_row][column]).replace(" °C", "").replace(" ", "")
            elif pd.notna(value):
                base_text = f"{float(value):.{int(spec['digits'])}f}"
            else:
                base_text = "-"
            display.iat[display_row, display.columns.get_loc(column)] = (
                f"{base_text}{adjustment}"
            )
            styles.iat[display_row, styles.columns.get_loc(column)] = (
                ALERT_UP_STYLE
                if direction == "up"
                else ALERT_DOWN_STYLE
                if direction == "down"
                else ALERT_CELL_STYLE
            )

    return display.style.apply(lambda _: styles, axis=None)


def sensor_matrix_styler(
    matrix: pd.DataFrame,
    group: str,
    normal_reference: dict[str, Any],
    mode: str,
) -> pd.io.formats.style.Styler:
    display = matrix.copy().astype(object)
    styles = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
    reference = reference_for_mode(normal_reference, mode).get("sensor", {})
    digits = 4 if group == "voltage" else 2
    unit = "V" if group == "voltage" else "°C"
    matrix_numeric = matrix.apply(pd.to_numeric, errors="coerce")
    peer_median = float(np.nanmedian(matrix_numeric.to_numpy(dtype=float)))

    for module in matrix.index:
        for sensor in matrix.columns:
            sensor_name = f"{module}{sensor}"
            value = pd.to_numeric(pd.Series([matrix.loc[module, sensor]]), errors="coerce").iloc[0]
            if pd.isna(value):
                display.loc[module, sensor] = "결측"
                styles.loc[module, sensor] = ALERT_DOWN_STYLE
                continue
            residual = float(value) - peer_median if np.isfinite(peer_median) else np.nan
            adjustment, direction = _adjustment(
                residual,
                reference.get(sensor_name),
                digits,
                unit,
                include_unit=False,
            )
            display.loc[module, sensor] = f"{float(value):.{digits}f}{adjustment}"
            styles.loc[module, sensor] = (
                ALERT_UP_STYLE
                if direction == "up"
                else ALERT_DOWN_STYLE
                if direction == "down"
                else NORMAL_CELL_STYLE
            )

    return display.style.apply(lambda _: styles, axis=None)
