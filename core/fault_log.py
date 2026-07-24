from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import BATCH_DIR
from .storage import load_deleted_fault_event_ids, load_fault_actions, load_fault_event_log


FAULT_STATUSES = {"NG", "NG_REVIEW", "ANOMALY", "FAULT"}
FAULT_EVENT_COLUMNS = [
    "event_id",
    "detected_at",
    "origin",
    "source_file",
    "source_path",
    "mode",
    "model_id",
    "model_version",
    "model_status",
    "fault_type",
    "fault_confidence",
    "fault_probabilities",
    "severity",
    "suspect_sensors",
    "suspect_modules",
    "suspect_cells",
    "detected_row",
    "fire_rate",
    "score_p95",
    "score_max",
    "max_consecutive_rows",
    "recommended_action",
    "disposition_guide",
    "recommendation_reason",
    "action_status",
    "final_action",
    "assignee",
    "action_notes",
    "action_updated_at",
]


FAULT_TYPE_ALIASES = {
    "capacity": "용량 불량",
    "capacity fault": "용량 불량",
    "용량불량": "용량 불량",
    "composite": "복합 불량",
    "composite fault": "복합 불량",
    "복합불량": "복합 불량",
    "weld": "용접·접촉 불량",
    "welding": "용접·접촉 불량",
    "contact": "용접·접촉 불량",
    "용접불량": "용접·접촉 불량",
    "sensing wire": "센싱와이어 불량",
    "sensing_wire": "센싱와이어 불량",
    "wire": "센싱와이어 불량",
    "센싱와이어불량": "센싱와이어 불량",
    "temperature sensor": "온도 센서 불량",
    "temp_sensor": "온도 센서 불량",
    "온도센서불량": "온도 센서 불량",
    "voltage sensor": "전압 센서 불량",
    "voltage_sensor": "전압 센서 불량",
    "전압센서불량": "전압 센서 불량",
    "thermal": "열 관리 이상",
    "thermal management": "열 관리 이상",
    "열관리이상": "열 관리 이상",
    "internal short": "내부 단락·안전 위험",
    "short circuit": "내부 단락·안전 위험",
    "내부단락": "내부 단락·안전 위험",
}


ACTION_GUIDE = {
    "용량 불량": (
        "용량 재시험",
        "재시험 후 기준 미달 시 출하 제외",
        "동일 조건 충·방전으로 용량과 에너지 유지율을 재확인합니다.",
    ),
    "복합 불량": (
        "출하 보류 및 복합 진단",
        "원인별 재작업 후 전체 충·방전 재시험",
        "용량·접촉·센서 계통이 함께 의심되므로 단일 부품 교체 전에 원인 분리 진단을 수행합니다.",
    ),
    "용접·접촉 불량": (
        "출하 보류 및 접합부 점검",
        "재작업 후 전압강하·발열 재검사",
        "접촉저항 증가 가능성이 있으므로 버스바와 용접부를 우선 점검합니다.",
    ),
    "센싱와이어 불량": (
        "하네스 점검 후 재계측",
        "배선 수리 또는 교체 후 재시험",
        "센싱와이어 단선·접촉 불량과 커넥터 체결 상태를 확인합니다.",
    ),
    "온도 센서 불량": (
        "온도 센서 교차검증",
        "센서 교체 후 재시험",
        "인접 센서와 기준 온도계로 오프셋 또는 단선 여부를 확인합니다.",
    ),
    "전압 센서 불량": (
        "전압 센싱 회로 재계측",
        "센싱 회로 수리 후 재시험",
        "독립 계측기와 비교해 센서·배선·BMS 입력 회로를 분리 점검합니다.",
    ),
    "열 관리 이상": (
        "팩 격리 및 냉각계통 점검",
        "원인 해소와 열 안정성 확인 전 출하 보류",
        "국부 발열, 냉각 유량, 열접촉 및 센서 위치를 함께 확인합니다.",
    ),
    "내부 단락·안전 위험": (
        "즉시 격리",
        "안전 절차에 따른 폐기 검토",
        "추가 충·방전을 중지하고 안전 담당자의 승인 절차를 적용합니다.",
    ),
    "유형 분석 대기": (
        "격리 후 유형 분류 및 재시험",
        "자동 폐기 금지·담당자 검토 필요",
        "이진 불량 판정만 존재하므로 유형 분류와 원시 센서 확인이 필요합니다.",
    ),
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _first(payloads: Iterable[dict[str, Any]], keys: Iterable[str], default: Any = None) -> Any:
    for payload in payloads:
        for key in keys:
            if key in payload and not _is_missing(payload[key]):
                return payload[key]
    return default


def _as_list(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        raw = list(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
            raw = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (json.JSONDecodeError, TypeError):
            raw = re.split(r"[,;|]", text)
    else:
        raw = [value]
    result: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text.lower() not in {"nan", "none", "null"} and text not in result:
            result.append(text)
    return result


def parse_probabilities(value: Any) -> dict[str, float]:
    if _is_missing(value):
        return {}
    raw = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in raw.items():
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            result[normalize_fault_type(key)] = numeric
    return result


def normalize_fault_type(value: Any) -> str:
    if _is_missing(value):
        return "유형 분석 대기"
    text = str(value).strip()
    compact = re.sub(r"[ _·\-/]", "", text.lower())
    for alias, normalized in FAULT_TYPE_ALIASES.items():
        if compact == re.sub(r"[ _·\-/]", "", alias.lower()):
            return normalized
    return text


def _locations_from_sensors(sensors: list[str]) -> tuple[list[str], list[str]]:
    modules: list[str] = []
    cells: list[str] = []
    for sensor in sensors:
        module_match = re.search(r"M\d{2}", sensor, flags=re.IGNORECASE)
        if module_match:
            module = module_match.group(0).upper()
            if module not in modules:
                modules.append(module)
        if re.fullmatch(r"M\d{2}CV\d{2}", sensor, flags=re.IGNORECASE):
            cell = sensor.upper()
            if cell not in cells:
                cells.append(cell)
    return modules, cells


def recommendation_for(fault_type: str) -> dict[str, str]:
    action, disposition, reason = ACTION_GUIDE.get(
        normalize_fault_type(fault_type),
        ACTION_GUIDE["유형 분석 대기"],
    )
    return {
        "recommended_action": action,
        "disposition_guide": disposition,
        "recommendation_reason": reason,
    }


def extract_fault_metadata(result: dict[str, Any]) -> dict[str, Any]:
    summary = dict(result.get("summary", {})) if isinstance(result.get("summary"), dict) else dict(result)
    details = dict(result.get("details", {})) if isinstance(result.get("details"), dict) else {}
    payloads = [summary, details]
    for key in ["fault", "classification", "fault_classification", "type_classifier"]:
        nested = details.get(key)
        if isinstance(nested, dict):
            payloads.append(nested)

    probabilities = parse_probabilities(
        _first(payloads, ["fault_probabilities", "class_probabilities", "type_probabilities", "probabilities"])
    )
    raw_type = _first(payloads, ["fault_type", "predicted_fault_type", "fault_label", "class_name"])
    if _is_missing(raw_type) and probabilities:
        raw_type = max(probabilities, key=probabilities.get)
    fault_type = normalize_fault_type(raw_type)

    raw_confidence = _first(payloads, ["fault_confidence", "type_confidence", "class_confidence"])
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = float(probabilities.get(fault_type, np.nan))
    if not np.isfinite(confidence):
        confidence = np.nan

    sensors = _as_list(
        _first(payloads, ["suspect_sensors", "problem_sensors", "fault_sensors", "top_sensors"])
    )
    modules = _as_list(_first(payloads, ["suspect_modules", "problem_modules", "fault_modules"]))
    cells = _as_list(_first(payloads, ["suspect_cells", "problem_cells", "fault_cells"]))
    inferred_modules, inferred_cells = _locations_from_sensors(sensors)
    modules = list(dict.fromkeys(modules + inferred_modules))
    cells = list(dict.fromkeys(cells + inferred_cells))

    raw_severity = _first(payloads, ["severity", "fault_severity", "risk_level"])
    severity = str(raw_severity).strip() if not _is_missing(raw_severity) else "검토 필요"
    guide = recommendation_for(fault_type)
    return {
        "fault_type": fault_type,
        "fault_confidence": confidence,
        "fault_probabilities": json.dumps(probabilities, ensure_ascii=False),
        "severity": severity,
        "suspect_sensors": ", ".join(sensors),
        "suspect_modules": ", ".join(modules),
        "suspect_cells": ", ".join(cells),
        **guide,
    }


def _event_id(*parts: Any) -> str:
    text = "::".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def build_fault_event(
    result: dict[str, Any],
    *,
    source_file: str,
    source_path: str = "",
    mode: str = "UNKNOWN",
    detected_at: str | None = None,
    detected_row: int | None = None,
    origin: str = "파일 판정",
    status_override: str | None = None,
    occurrence_key: str | None = None,
) -> dict[str, Any] | None:
    summary = dict(result.get("summary", {})) if isinstance(result.get("summary"), dict) else dict(result)
    status = str(status_override or summary.get("status", "")).upper()
    if status not in FAULT_STATUSES:
        return None
    detected = detected_at or datetime.now().isoformat(timespec="seconds")
    metadata_result = result
    details = dict(result.get("details", {})) if isinstance(result.get("details"), dict) else {}
    fault_by_row = details.get("fault_by_row", {})
    if detected_row is not None and isinstance(fault_by_row, dict):
        row_fault = fault_by_row.get(str(int(detected_row)))
        if isinstance(row_fault, dict):
            metadata_result = {
                "summary": summary,
                "details": {**details, **row_fault},
            }
    metadata = extract_fault_metadata(metadata_result)
    model_id = str(summary.get("model_id", ""))
    model_version = str(summary.get("model_version", ""))
    event_id_parts = [origin, source_file, model_id, model_version, detected_row or 0, detected]
    if occurrence_key:
        event_id_parts.append(occurrence_key)
    event_id = _event_id(*event_id_parts)
    return {
        "event_id": event_id,
        "detected_at": detected,
        "origin": origin,
        "source_file": source_file,
        "source_path": source_path,
        "mode": str(summary.get("mode", mode)).upper(),
        "model_id": model_id,
        "model_version": model_version,
        "model_status": status,
        "detected_row": detected_row if detected_row is not None else np.nan,
        "fire_rate": summary.get("fire_rate", np.nan),
        "score_p95": summary.get("score_p95", np.nan),
        "score_max": summary.get("score_max", np.nan),
        "max_consecutive_rows": summary.get("max_consecutive_rows", np.nan),
        "action_status": "신규",
        "final_action": "미결정",
        "assignee": "",
        "action_notes": "",
        "action_updated_at": "",
        **metadata,
    }


def batch_fault_events(frame: pd.DataFrame, *, batch_id: str, detected_at: str) -> pd.DataFrame:
    if frame.empty or "status" not in frame.columns:
        return pd.DataFrame(columns=FAULT_EVENT_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        status = str(row.get("status", "")).upper()
        if status not in FAULT_STATUSES:
            continue
        payload = row.to_dict()
        source_file = str(payload.get("source_file") or payload.get("file_name") or "")
        event = build_fault_event(
            {"summary": payload, "details": payload},
            source_file=source_file,
            source_path=str(payload.get("source_path") or payload.get("path") or ""),
            mode=str(payload.get("mode", "UNKNOWN")),
            detected_at=detected_at,
            detected_row=int(payload["detected_row"]) if pd.notna(payload.get("detected_row")) else None,
            origin="배치 판정",
            status_override=status,
        )
        if event:
            event["event_id"] = _event_id(batch_id, source_file, event["model_id"], event["model_version"])
            rows.append(event)
    return pd.DataFrame(rows)


def _batch_detected_at(csv_path: Path) -> str:
    metadata_path = csv_path.with_suffix(".json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            created_at = metadata.get("created_at")
            if created_at:
                return str(created_at)
        except (json.JSONDecodeError, OSError):
            pass
    return datetime.fromtimestamp(csv_path.stat().st_mtime).isoformat(timespec="seconds")


def load_fault_events(current_batch: pd.DataFrame | None = None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    live_events = load_fault_event_log()
    if not live_events.empty:
        parts.append(live_events)

    for csv_path in sorted(BATCH_DIR.glob("batch_result_*.csv")):
        try:
            batch = pd.read_csv(csv_path, encoding="utf-8-sig")
            events = batch_fault_events(
                batch,
                batch_id=csv_path.stem,
                detected_at=_batch_detected_at(csv_path),
            )
            if not events.empty:
                parts.append(events)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue

    if isinstance(current_batch, pd.DataFrame) and not current_batch.empty:
        events = batch_fault_events(
            current_batch,
            batch_id="current_session_batch",
            detected_at=datetime.now().isoformat(timespec="seconds"),
        )
        if not events.empty:
            parts.append(events)

    if not parts:
        return pd.DataFrame(columns=FAULT_EVENT_COLUMNS)
    events = pd.concat(parts, ignore_index=True, sort=False)
    for column in FAULT_EVENT_COLUMNS:
        if column not in events.columns:
            events[column] = ""
    events = events.drop_duplicates(subset=["event_id"], keep="last")
    deleted_event_ids = load_deleted_fault_event_ids()
    if deleted_event_ids:
        events = events[~events["event_id"].astype(str).isin(deleted_event_ids)].copy()

    actions = load_fault_actions()
    if not actions.empty and "event_id" in actions.columns:
        latest = actions.sort_values("updated_at").drop_duplicates("event_id", keep="last")
        action_columns = [
            "event_id",
            "action_status",
            "final_action",
            "assignee",
            "action_notes",
            "updated_at",
        ]
        latest = latest[[column for column in action_columns if column in latest.columns]].rename(
            columns={"updated_at": "action_updated_at"}
        )
        events = events.drop(
            columns=["action_status", "final_action", "assignee", "action_notes", "action_updated_at"],
            errors="ignore",
        ).merge(latest, on="event_id", how="left")

    for column, default in {
        "action_status": "신규",
        "final_action": "미결정",
        "assignee": "",
        "action_notes": "",
        "action_updated_at": "",
    }.items():
        if column not in events.columns:
            events[column] = default
        events[column] = events[column].fillna(default)
    return events[FAULT_EVENT_COLUMNS].sort_values("detected_at", ascending=False).reset_index(drop=True)
