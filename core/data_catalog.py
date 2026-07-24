from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


CV_PATTERNS = [
    re.compile(r"M\d{2}CV\d{2}$", re.IGNORECASE),
    re.compile(r"Voltages_R2_\d+$", re.IGNORECASE),
    re.compile(r"(?:cell[_ -]?)?(?:voltage|volt|cv)[_ -]?\d+$", re.IGNORECASE),
]
TEMP_PATTERNS = [
    re.compile(r"M\d{2}T\d{2}$", re.IGNORECASE),
    re.compile(r"CellTemp_R_\d+$", re.IGNORECASE),
    re.compile(r"(?:cell|module)?[_ -]?(?:temperature|temp|mt)[_ -]?\d+$", re.IGNORECASE),
]
LABEL_SUFFIXES = ("_label.csv",)
SKIP_FILE_TOKENS = ("manifest", "summary", "static", "threshold", "prediction", "metric")


@dataclass(frozen=True)
class DataFileRecord:
    source_name: str
    source_role: str
    path: str
    file_name: str
    size_mb: float
    rows: int
    columns: int
    cv_sensors: int
    temp_sensors: int
    mode: str
    label_hint: str
    schema: str
    modified_at: str
    readable: bool
    issue: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _natural_key(name: str) -> tuple[int | str, ...]:
    parts = re.split(r"(\d+)", str(name))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _matches_any(name: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.fullmatch(str(name)) for pattern in patterns)


def detect_sensor_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    cols = [str(c) for c in columns]
    cv_cols = sorted([c for c in cols if _matches_any(c, CV_PATTERNS)], key=_natural_key)
    temp_cols = sorted([c for c in cols if _matches_any(c, TEMP_PATTERNS)], key=_natural_key)
    return cv_cols, temp_cols


def infer_mode(file_name: str, df: pd.DataFrame | None = None) -> str:
    lower = str(file_name).lower()
    if "dchg" in lower or "discharge" in lower:
        return "DCHG"
    if "chg" in lower or "charge" in lower:
        return "CHG"
    if df is not None:
        current_col = next((c for c in ["Current", "CURRENT", "current", "Current_R2"] if c in df.columns), None)
        if current_col:
            current = pd.to_numeric(df[current_col], errors="coerce").dropna()
            active = current[current.abs() > 1e-9]
            if not active.empty:
                return "DCHG" if float(active.median()) > 0 else "CHG"
    return "UNKNOWN"


def infer_label_hint(file_name: str) -> str:
    upper = str(file_name).upper()
    if re.search(r"(?:^|[_ -])NG(?:[_ .-]|$)", upper):
        return "NG"
    if re.search(r"(?:^|[_ -])OK(?:[_ .-]|$)", upper):
        return "OK"
    if any(token in upper for token in ["BLOCKED", "FANOFF", "FAULT", "HIGHFAN"]):
        return "FAULT_SCENARIO"
    return "UNLABELED"


def infer_schema(cv_count: int, temp_count: int, columns: Iterable[str]) -> str:
    cols = {str(c) for c in columns}
    if any(re.fullmatch(r"M\d{2}CV\d{2}", c, re.IGNORECASE) for c in cols):
        return "KMAP_16M"
    if any(c.startswith("Voltages_R2_") for c in cols):
        return "SBL_72CELL"
    if cv_count or temp_count:
        return "GENERIC_SENSOR_ARRAY"
    return "PACK_SIGNAL_ONLY"


def read_csv_resilient(path_or_buffer: Any, nrows: int | None = None) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, nrows=nrows, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV를 읽지 못했습니다. " + " | ".join(errors[-2:]))


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
    return max(0, lines - 1)


def should_catalog(path: Path) -> bool:
    lower = path.name.lower()
    if not lower.endswith(".csv") or lower.endswith(LABEL_SUFFIXES):
        return False
    return not any(token in lower for token in SKIP_FILE_TOKENS)


def inspect_csv(path: Path, source_name: str, source_role: str) -> DataFileRecord:
    try:
        header = read_csv_resilient(path, nrows=4)
        cv_cols, temp_cols = detect_sensor_columns(header.columns)
        mode = infer_mode(path.name, header)
        return DataFileRecord(
            source_name=source_name,
            source_role=source_role,
            path=str(path),
            file_name=path.name,
            size_mb=round(path.stat().st_size / (1024**2), 3),
            rows=count_csv_rows(path),
            columns=len(header.columns),
            cv_sensors=len(cv_cols),
            temp_sensors=len(temp_cols),
            mode=mode,
            label_hint=infer_label_hint(path.name),
            schema=infer_schema(len(cv_cols), len(temp_cols), header.columns),
            modified_at=pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            readable=True,
        )
    except Exception as exc:
        return DataFileRecord(
            source_name=source_name,
            source_role=source_role,
            path=str(path),
            file_name=path.name,
            size_mb=round(path.stat().st_size / (1024**2), 3) if path.exists() else 0.0,
            rows=0,
            columns=0,
            cv_sensors=0,
            temp_sensors=0,
            mode="UNKNOWN",
            label_hint=infer_label_hint(path.name),
            schema="UNREADABLE",
            modified_at="",
            readable=False,
            issue=f"{type(exc).__name__}: {exc}",
        )


def build_catalog(data_sources: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for source in data_sources:
        if not source.get("enabled", True):
            continue
        root = Path(str(source.get("path", ""))).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        iterator = root.rglob("*.csv") if source.get("recursive", False) else root.glob("*.csv")
        for path in sorted(iterator, key=lambda p: _natural_key(p.name)):
            if should_catalog(path):
                records.append(
                    inspect_csv(
                        path,
                        source_name=str(source.get("name", root.name)),
                        source_role=str(source.get("role", "external")),
                    ).to_dict()
                )
    if not records:
        return pd.DataFrame(columns=list(DataFileRecord.__annotations__))
    return pd.DataFrame(records).sort_values(
        ["source_role", "source_name", "file_name"],
        kind="stable",
    ).reset_index(drop=True)


def resolve_time_axis(df: pd.DataFrame) -> tuple[pd.Series, str, pd.Series | None]:
    if {"Date", "Time"}.issubset(df.columns):
        timestamp = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            errors="coerce",
        )
        if timestamp.notna().mean() >= 0.5:
            return timestamp, "측정 시간", timestamp
    for col in ["Timestamp", "timestamp", "DATETIME", "datetime"]:
        if col in df.columns:
            timestamp = pd.to_datetime(df[col], errors="coerce")
            if timestamp.notna().mean() >= 0.5:
                return timestamp, "측정 시간", timestamp
    for col in ["Temptime", "TimeLine", "Timeline", "order", "Order", "TIME", "Time", "time"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().mean() >= 0.5:
                return numeric, col, None
    axis = pd.Series(np.arange(1, len(df) + 1), index=df.index, name="order")
    return axis, "순서", None


def make_elapsed_seconds(df: pd.DataFrame) -> pd.Series:
    axis, _, timestamp = resolve_time_axis(df)
    if timestamp is not None and timestamp.notna().any():
        first = timestamp.dropna().iloc[0]
        return (timestamp - first).dt.total_seconds().ffill().bfill().fillna(0.0)
    numeric = pd.to_numeric(axis, errors="coerce")
    if numeric.notna().any():
        return (numeric - numeric.dropna().iloc[0]).ffill().bfill().fillna(0.0)
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)


def _max_consecutive_true(values: pd.Series | np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def audit_data_quality(df: pd.DataFrame, policy: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cv_cols, temp_cols = detect_sensor_columns(df.columns)
    sensor_cols = cv_cols + temp_cols
    numeric_sensor = df[sensor_cols].apply(pd.to_numeric, errors="coerce") if sensor_cols else pd.DataFrame(index=df.index)
    missing_rate = float(numeric_sensor.isna().to_numpy().mean()) if sensor_cols and len(df) else 0.0
    zero_rate = float(numeric_sensor.eq(0).to_numpy().mean()) if sensor_cols and len(df) else 0.0

    duplicate_cols = [c for c in df.columns if c.lower() not in {"order", "timeline", "date", "time"}]
    duplicate_rate = float(df.duplicated(subset=duplicate_cols).mean()) if duplicate_cols and len(df) else 0.0

    _, _, timestamp = resolve_time_axis(df)
    gaps = pd.Series(dtype=float)
    non_monotonic = 0
    if timestamp is not None:
        gaps = timestamp.diff().dt.total_seconds()
        non_monotonic = int((gaps < 0).sum())
    time_gap_threshold = float(policy.get("time_gap_seconds", 30.0))
    large_gap_count = int((gaps > time_gap_threshold).sum()) if not gaps.empty else 0

    cv = df[cv_cols].apply(pd.to_numeric, errors="coerce") if cv_cols else pd.DataFrame(index=df.index)
    tt = df[temp_cols].apply(pd.to_numeric, errors="coerce") if temp_cols else pd.DataFrame(index=df.index)
    cv_invalid = int(
        ((cv < float(policy.get("min_cell_voltage", 1.5))) | (cv > float(policy.get("max_cell_voltage", 5.0)))).sum().sum()
    ) if cv_cols else 0
    temp_invalid = int(
        ((tt < float(policy.get("min_temperature", -40.0))) | (tt > float(policy.get("max_temperature", 100.0)))).sum().sum()
    ) if temp_cols else 0
    constant_sensors = int((numeric_sensor.nunique(dropna=True) <= 1).sum()) if sensor_cols else 0

    checks = [
        ("센서 컬럼", len(sensor_cols) > 0, f"전압 {len(cv_cols)}개, 온도 {len(temp_cols)}개"),
        ("센서 결측률", missing_rate <= float(policy.get("max_missing_sensor_rate", 0.01)), f"{missing_rate:.3%}"),
        ("중복 행", duplicate_rate <= float(policy.get("max_duplicate_rate", 0.01)), f"{duplicate_rate:.3%}"),
        ("시간 간격", large_gap_count == 0, f"> {time_gap_threshold:g}초: {large_gap_count}행"),
        ("시간 역행", non_monotonic == 0, f"{non_monotonic}행"),
        ("전압 물리범위", cv_invalid == 0, f"범위 밖 {cv_invalid:,}값"),
        ("온도 물리범위", temp_invalid == 0, f"범위 밖 {temp_invalid:,}값"),
        ("고정 센서", constant_sensors == 0, f"{constant_sensors}개"),
    ]
    check_df = pd.DataFrame(checks, columns=["점검 항목", "통과", "결과"])
    failed = int((~check_df["통과"]).sum())
    if failed == 0:
        status = "PASS"
    elif any(name in {"센서 컬럼", "전압 물리범위", "온도 물리범위"} and not passed for name, passed, _ in checks):
        status = "FAIL"
    else:
        status = "REVIEW"
    summary = {
        "status": status,
        "failed_checks": failed,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "cv_sensors": len(cv_cols),
        "temp_sensors": len(temp_cols),
        "missing_rate": missing_rate,
        "zero_rate": zero_rate,
        "duplicate_rate": duplicate_rate,
        "large_gap_count": large_gap_count,
        "max_gap_seconds": float(gaps.max()) if not gaps.empty and gaps.notna().any() else np.nan,
        "constant_sensors": constant_sensors,
    }
    return check_df, summary


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, max_points, dtype=int)


def max_consecutive_true(values: pd.Series | np.ndarray) -> int:
    return _max_consecutive_true(values)
