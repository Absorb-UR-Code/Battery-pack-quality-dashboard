from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .data_catalog import detect_sensor_columns, infer_mode, make_elapsed_seconds


PACK_COLUMNS = [
    "Voltage",
    "Current",
    "RSOCmin",
    "RSOCmax",
    "RSOCavg",
    "USOCmin",
    "USOCmax",
    "USOCavg",
    "SOH",
    "Power",
    "ChgPmax",
    "DchgPmax",
    "ChgImax",
    "DchgImax",
    "Vmin",
    "Vmax",
    "DV",
    "Tmin",
    "Tmax",
    "Tavg",
]

ALIASES = {
    "Current": ["CURRENT", "current", "Current_R2", "Current_Smooth"],
    "Voltage": ["VOLTAGE", "voltage", "PackVoltage", "pack_voltage"],
    "Power": ["POWER", "power", "PackPower", "pack_power"],
    "RSOCavg": ["SOC", "soc", "SOCavg", "SOC_R2_01"],
}


def _entropy_from_abs_residual(values: np.ndarray) -> np.ndarray:
    denom = np.nansum(values, axis=1, keepdims=True)
    probability = np.divide(values, denom, out=np.zeros_like(values), where=denom != 0)
    log_probability = np.zeros_like(probability)
    positive = probability > 0
    log_probability[positive] = np.log(probability[positive])
    entropy = -np.nansum(probability * log_probability, axis=1)
    max_entropy = math.log(values.shape[1]) if values.shape[1] > 1 else 1.0
    return entropy / max_entropy


def _numeric_from_alias(df: pd.DataFrame, target: str) -> pd.Series | None:
    candidates = [target, *ALIASES.get(target, [])]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def operation_state(current: pd.Series | None, eps: float = 1e-9) -> pd.Series:
    if current is None:
        return pd.Series(dtype="object")
    return pd.Series(
        np.select(
            [current > eps, current < -eps],
            ["charge", "discharge"],
            default="rest",
        ),
        index=current.index,
    )


def build_sensor_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the five operator KPIs from synchronized CV and temperature sensors."""
    cv_cols, temp_cols = detect_sensor_columns(df.columns)
    out = pd.DataFrame(index=df.index)

    if cv_cols:
        cv = df[cv_cols].apply(pd.to_numeric, errors="coerce")
        out["cv_mean"] = cv.mean(axis=1)
        out["cv_std"] = cv.std(axis=1, ddof=0)
    else:
        out["cv_mean"] = np.nan
        out["cv_std"] = np.nan

    if temp_cols:
        temp = df[temp_cols].apply(pd.to_numeric, errors="coerce")
        out["temp_mean"] = temp.mean(axis=1)
        out["temp_min"] = temp.min(axis=1)
        out["temp_max"] = temp.max(axis=1)
        out["temp_range"] = out["temp_max"] - out["temp_min"]
        out["temp_std"] = temp.std(axis=1, ddof=0)
    else:
        out["temp_mean"] = np.nan
        out["temp_min"] = np.nan
        out["temp_max"] = np.nan
        out["temp_range"] = np.nan
        out["temp_std"] = np.nan

    return out


def sensor_snapshot_matrix(
    row: pd.Series,
    sensor_columns: list[str],
    group: str,
) -> pd.DataFrame:
    """Pivot one measurement row into module-by-sensor matrices for field inspection."""
    if group not in {"voltage", "temperature"}:
        raise ValueError("group must be 'voltage' or 'temperature'")

    pattern = r"M(\d{2})CV(\d{2})" if group == "voltage" else r"M(\d{2})T(\d{2})"
    prefix = "CV" if group == "voltage" else "T"
    records: list[dict[str, Any]] = []
    for column in sensor_columns:
        match = re.fullmatch(pattern, str(column), re.I)
        if not match:
            continue
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        records.append(
            {
                "모듈": f"M{match.group(1)}",
                "센서": f"{prefix}{match.group(2)}",
                "값": value,
            }
        )

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .pivot(index="모듈", columns="센서", values="값")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


def build_row_features(df: pd.DataFrame, source_file: str = "uploaded.csv") -> pd.DataFrame:
    cv_cols, temp_cols = detect_sensor_columns(df.columns)
    features: dict[str, Any] = {
        "elapsed_sec": make_elapsed_seconds(df).astype(float),
    }

    for col in PACK_COLUMNS:
        series = _numeric_from_alias(df, col)
        if series is not None:
            features[col] = series

    current = features.get("Current")
    if current is not None:
        features["Current_abs"] = current.abs()
        features["operation_state"] = operation_state(current)
    else:
        features["operation_state"] = pd.Series("unknown", index=df.index)

    if "Voltage" in features and "Current" in features:
        features["Power_calc"] = features["Voltage"] * features["Current"]
        if "Power" in features:
            features["Power_error"] = features["Power"] - features["Power_calc"]

    if "RSOCmin" in features and "RSOCmax" in features:
        features["RSOC_spread"] = features["RSOCmax"] - features["RSOCmin"]
    if "USOCmin" in features and "USOCmax" in features:
        features["USOC_spread"] = features["USOCmax"] - features["USOCmin"]

    if cv_cols:
        cv = df[cv_cols].apply(pd.to_numeric, errors="coerce")
        values = cv.to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1)
        residual = values - mean[:, None]
        abs_residual = np.abs(residual)
        cv_range = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        features.update(
            {
                "cv_mean": mean,
                "cv_min": np.nanmin(values, axis=1),
                "cv_max": np.nanmax(values, axis=1),
                "cv_std": np.nanstd(values, axis=1),
                "cv_range": cv_range,
                "cv_resid_rmse": np.sqrt(np.nanmean(residual**2, axis=1)),
                "cv_resid_maxabs": np.nanmax(abs_residual, axis=1),
                "cv_resid_entropy": _entropy_from_abs_residual(abs_residual),
                "slope_spread": cv.diff().abs().max(axis=1) - cv.diff().abs().min(axis=1),
                "cv_range_delta": pd.Series(cv_range, index=df.index).diff().fillna(0.0),
            }
        )
        module_frames: list[pd.Series] = []
        kmap_modules = sorted({col[:3] for col in cv_cols if re.fullmatch(r"M\d{2}CV\d{2}", col, re.I)})
        for module in kmap_modules:
            cols = [col for col in cv_cols if col.startswith(module)]
            module_frames.append(cv[cols].mean(axis=1).rename(module))
        if module_frames:
            modules = pd.concat(module_frames, axis=1)
            features["module_range"] = modules.max(axis=1) - modules.min(axis=1)
        else:
            features["module_range"] = pd.Series(cv_range, index=df.index)
        pack_mean = pd.Series(mean, index=df.index)
        corr = cv.corrwith(pack_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        features["min_corr_to_pack_mean"] = float(corr.min()) if len(corr) else 1.0

    if temp_cols:
        temp = df[temp_cols].apply(pd.to_numeric, errors="coerce")
        values = temp.to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1)
        residual = values - mean[:, None]
        abs_residual = np.abs(residual)
        temp_range = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        features.update(
            {
                "temp_mean": mean,
                "temp_min": np.nanmin(values, axis=1),
                "temp_max": np.nanmax(values, axis=1),
                "temp_std": np.nanstd(values, axis=1),
                "temp_range": temp_range,
                "temp_resid_rmse": np.sqrt(np.nanmean(residual**2, axis=1)),
                "temp_resid_maxabs": np.nanmax(abs_residual, axis=1),
                "temp_resid_entropy": _entropy_from_abs_residual(abs_residual),
                "temp_range_delta": pd.Series(temp_range, index=df.index).diff().fillna(0.0),
            }
        )
        pair_gaps = []
        for module in range(1, 17):
            c1 = f"M{module:02d}T01"
            c2 = f"M{module:02d}T02"
            if c1 in temp.columns and c2 in temp.columns:
                pair_gaps.append((temp[c2] - temp[c1]).abs().rename(f"M{module:02d}"))
        if pair_gaps:
            pair_df = pd.concat(pair_gaps, axis=1)
            features["temp_pair_gap_max"] = pair_df.max(axis=1)
            features["temp_pair_gap_mean"] = pair_df.mean(axis=1)

    out = pd.DataFrame(features, index=df.index)
    derivative_cols = [
        "Voltage",
        "Current",
        "Power",
        "RSOCavg",
        "USOCavg",
        "SOH",
        "cv_range",
        "temp_range",
        "module_range",
    ]
    derived_parts: dict[str, pd.Series] = {}
    for col in derivative_cols:
        if col in out.columns:
            derived_parts[f"d_{col}"] = pd.to_numeric(out[col], errors="coerce").diff().fillna(0.0)
            derived_parts[f"{col}_ma30"] = pd.to_numeric(out[col], errors="coerce").rolling(30, min_periods=1).mean()
    if derived_parts:
        out = pd.concat([out, pd.DataFrame(derived_parts, index=out.index)], axis=1)
    out["source_file"] = source_file
    out["mode"] = infer_mode(source_file, df)
    return out.copy()


def prepare_feature_matrix(
    features: pd.DataFrame,
    columns: list[str],
    fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(index=features.index)
    missing: list[str] = []
    for col in columns:
        if col in features.columns:
            frame[col] = pd.to_numeric(features[col], errors="coerce")
        else:
            frame[col] = np.nan
            missing.append(col)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for col in columns:
        configured = (fill_values or {}).get(col)
        fallback = float(configured) if configured is not None else frame[col].median()
        if not np.isfinite(fallback):
            fallback = 0.0
        frame[col] = frame[col].fillna(fallback)
    return frame.astype(float), missing


def sensor_deviation_ranking(df: pd.DataFrame, group: str = "voltage") -> pd.DataFrame:
    cv_cols, temp_cols = detect_sensor_columns(df.columns)
    cols = cv_cols if group == "voltage" else temp_cols
    if not cols:
        return pd.DataFrame(columns=["sensor", "mean_abs_residual", "p95_abs_residual", "max_abs_residual", "missing_rate"])
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    center = values.median(axis=1)
    residual = values.sub(center, axis=0).abs()
    ranking = pd.DataFrame(
        {
            "sensor": cols,
            "mean_abs_residual": residual.mean(axis=0).to_numpy(),
            "p95_abs_residual": residual.quantile(0.95, axis=0).to_numpy(),
            "max_abs_residual": residual.max(axis=0).to_numpy(),
            "missing_rate": values.isna().mean(axis=0).to_numpy(),
            "zero_rate": values.eq(0).mean(axis=0).to_numpy(),
        }
    )
    return ranking.sort_values(["p95_abs_residual", "max_abs_residual"], ascending=False).reset_index(drop=True)


def module_temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for module in range(1, 17):
        c1 = f"M{module:02d}T01"
        c2 = f"M{module:02d}T02"
        if c1 not in df.columns or c2 not in df.columns:
            continue
        t1 = pd.to_numeric(df[c1], errors="coerce")
        t2 = pd.to_numeric(df[c2], errors="coerce")
        gap = t2 - t1
        rows.append(
            {
                "module": f"M{module:02d}",
                "T01_mean": float(t1.mean()),
                "T02_mean": float(t2.mean()),
                "signed_gap_mean": float(gap.mean()),
                "abs_gap_p95": float(gap.abs().quantile(0.95)),
                "abs_gap_max": float(gap.abs().max()),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_gap_p95", ascending=False) if rows else pd.DataFrame()
