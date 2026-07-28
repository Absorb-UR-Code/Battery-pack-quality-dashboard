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
from .storage import (
    load_deleted_fault_event_ids,
    load_fault_actions,
    load_fault_event_log,
)


FAULT_STATUSES = {"NG", "NG_REVIEW", "ANOMALY", "FAULT"}
FAULT_EVENT_COLUMNS = [
    "logged_at",
    "event_id",
    "detected_at",
    "origin",
    "source_file",
    "source_path",
    "mode",
    "mode_display",
    "model_id",
    "model_name",
    "model_version",
    "model_status",
    "model_trigger",
    "model_summary_json",
    "model_details_json",
    "fault_type",
    "fault_confidence",
    "fault_confidence_percent",
    "fault_probabilities",
    "severity",
    "rpn",
    "risk_level",
    "risk_label",
    "risk_color",
    "pfmea_ng_codes",
    "suspect_sensors",
    "suspect_modules",
    "suspect_cells",
    "detected_row",
    "fire_rate",
    "fire_rate_percent",
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
    "source_row_number",
    "source_window_start_row",
    "source_window_end_row",
    "source_window_row_count",
    "source_column_count",
    "source_columns_json",
    "source_row_json",
    "source_window_json",
]


FAULT_TYPE_ALIASES = {
    "capacity": "용량 불량",
    "capacity fault": "용량 불량",
    "용량불량": "용량 불량",
    "low capacity": "저용량 불량",
    "low_capacity": "저용량 불량",
    "low capacity fault": "저용량 불량",
    "저용량불량": "저용량 불량",
    "high resistance": "고저항 불량",
    "high_resistance": "고저항 불량",
    "high resistance fault": "고저항 불량",
    "고저항불량": "고저항 불량",
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


PFMEA_GUIDE = {
    "저용량 불량": {
        "pfmea_ng_codes": "NG5",
        "rpn_values": [84, 42, 63],
        "recommended_action": (
            "샘플링 비율 상향 및 배치별 grade 분리 · 인터락 강화 및 알람 도입 · "
            "매칭 알고리즘 tolerance 재산정"
        ),
        "disposition_guide": "용량 재시험 후 기준 미달 팩은 출하 보류하고 셀 매칭·열화 원인을 점검",
    },
    "고저항 불량": {
        "pfmea_ng_codes": "NG6",
        "rpn_values": [96, 63, 84, 54],
        "recommended_action": (
            "Power 관리 강화 및 검교정 주기 단축 · 지그 마모 주기 자동 관리 · "
            "체결 토크 자동 관리 및 이력 기록 · 세정 자동화 검토"
        ),
        "disposition_guide": "접속·용접부 전압강하와 국부 발열을 재검사하고 기준 초과 시 재작업",
    },
    "용량 불량": {
        "pfmea_ng_codes": "NG5",
        "rpn_values": [84, 42, 63],
        "recommended_action": (
            "샘플링 비율 상향 및 배치별 grade 분리 · 인터락 강화 및 알람 도입 · "
            "매칭 알고리즘 tolerance 재산정"
        ),
        "disposition_guide": "PFMEA 권고 조치 후 용량 재시험, 기준 미달 시 출하 보류",
    },
    "용접·접촉 불량": {
        "pfmea_ng_codes": "NG6",
        "rpn_values": [96, 63, 84, 54],
        "recommended_action": (
            "Power 관리 강화 및 검교정 주기 단축 · 지그 마모 주기 자동 관리 · "
            "체결 토크 자동 관리 및 이력 기록 · 세정 자동화 검토"
        ),
        "disposition_guide": "접합부 재작업 및 PFMEA 권고 조치 후 전압강하·발열 재검사",
    },
    "센싱와이어 불량": {
        "pfmea_ng_codes": "NG7",
        "rpn_values": [96, 54, 72, 96, 63],
        "recommended_action": (
            "outlier 자동 격리 시스템 구축 · IQC 자동화 시스템 도입 · 체결 확인 인터락 추가 · "
            "체결 토크 표준화 및 이력 관리 · 라우팅 지그 필수화"
        ),
        "disposition_guide": "배선·체결 상태를 보정하고 PFMEA 권고 조치 후 센서 재계측",
    },
    "온도 센서 불량": {
        "pfmea_ng_codes": "NG8, NG9",
        "rpn_values": [72, 96, 168, 63, 72, 63, 54],
        "recommended_action": (
            "도포량 정량화 및 자동 도포 검토 · 센서 수명 관리 시스템 구축 · "
            "레시피 자동 로딩 및 이중 확인 · 지그 및 부착 지그 정밀도 상향 · "
            "충진·탈기 자동화 검토 · 항온 챔버 도입 검토"
        ),
        "disposition_guide": "즉시 출하 보류 후 센서·레시피·지그를 점검하고 정상 확인 시 재시험",
    },
    "전압 센서 불량": {
        "pfmea_ng_codes": "NG7",
        "rpn_values": [96, 54, 72, 96, 63],
        "recommended_action": (
            "outlier 자동 격리 시스템 구축 · IQC 자동화 시스템 도입 · 체결 확인 인터락 추가 · "
            "체결 토크 표준화 및 이력 관리 · 라우팅 지그 필수화"
        ),
        "disposition_guide": "센싱 회로와 체결 상태를 보정하고 PFMEA 권고 조치 후 재계측",
    },
    "열 관리 이상": {
        "pfmea_ng_codes": "NG9",
        "rpn_values": [63, 72, 63, 54],
        "recommended_action": (
            "지그 정밀도 상향 · 부착 지그 정밀도 상향 · 충진·탈기 자동화 검토 · "
            "항온 챔버 도입 검토"
        ),
        "disposition_guide": "열 안정성과 냉각 조건 확인 전 출하 보류",
    },
    "복합 불량": {
        "pfmea_ng_codes": "ALL",
        "rpn_values": [180, 168, 120],
        "recommended_action": "AI 자동 판정 시스템 도입 · 관련 NG 유형별 PFMEA 권고 조치 병행",
        "disposition_guide": "즉시 출하 보류 후 원인별 조치와 전체 충·방전 재시험",
    },
    "내부 단락·안전 위험": {
        "pfmea_ng_codes": "ALL",
        "rpn_values": [180, 168, 120],
        "recommended_action": "AI 자동 판정 시스템 도입 · 안전 절차에 따른 즉시 격리 및 전문 진단",
        "disposition_guide": "추가 충·방전 금지, 안전 담당자 승인 전 이동·출하 금지",
    },
    "유형 분석 대기": {
        "pfmea_ng_codes": "미매핑",
        "rpn_values": [0],
        "recommended_action": "격리 후 불량 유형 분류 및 Pack_PFMEA 항목 매핑",
        "disposition_guide": "자동 폐기 금지, 담당자 검토 후 조치 결정",
    },
}


MODE_DISPLAY = {
    "CHG": "충전",
    "DCHG": "방전",
    "UNKNOWN": "미분류",
    "ALL": "충·방전",
}


def display_mode(value: Any) -> str:
    """Return an operator-facing Korean label while retaining mode codes internally."""
    if _is_missing(value):
        return "미분류"
    text = str(value).strip()
    return MODE_DISPLAY.get(text.upper(), text)


def _risk_from_rpn(rpn: int | float) -> tuple[int, str, str]:
    """Map Pack_PFMEA fill bands to dashboard risk levels."""
    numeric = float(rpn)
    if numeric >= 120:
        return 2, "빨강", "높음"
    if numeric >= 60:
        return 1, "노랑", "주의"
    return 0, "흰색", "낮음"


def _second_highest_distinct(values: Iterable[Any]) -> tuple[int, int]:
    """Return (selected, excluded_max) using the next distinct RPN below the maximum."""
    ordered = sorted(
        {
            int(float(value))
            for value in values
            if not _is_missing(value) and np.isfinite(float(value))
        },
        reverse=True,
    )
    if not ordered:
        return 0, 0
    return (ordered[1] if len(ordered) > 1 else ordered[0]), ordered[0]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _json_safe_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_safe_object(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [
            {str(key): _json_safe_object(item) for key, item in record.items()}
            for record in value.to_dict(orient="records")
        ]
    if isinstance(value, pd.Series):
        return {str(key): _json_safe_object(item) for key, item in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(key): _json_safe_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_safe_object(item) for item in list(value)]
    return _json_safe_scalar(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_safe_object(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _percent_text(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{numeric:.1%}" if np.isfinite(numeric) else ""


def _infer_detected_row(result: dict[str, Any]) -> int | None:
    summary = dict(result.get("summary", {})) if isinstance(result.get("summary"), dict) else {}
    details = dict(result.get("details", {})) if isinstance(result.get("details"), dict) else {}
    for payload in [summary, details]:
        for key in [
            "detected_row",
            "first_fault_row",
            "first_anomaly_row",
        ]:
            value = payload.get(key)
            try:
                row_number = int(float(value))
            except (TypeError, ValueError):
                continue
            if row_number >= 1:
                return row_number

    fault_by_row = details.get("fault_by_row", {})
    if isinstance(fault_by_row, dict):
        explicit_rows: list[int] = []
        for row_key in fault_by_row:
            try:
                explicit_rows.append(int(float(str(row_key))))
            except (TypeError, ValueError):
                continue
        valid_explicit_rows = [row for row in explicit_rows if row >= 1]
        if valid_explicit_rows:
            return min(valid_explicit_rows)

    row_result = result.get("row_result")
    if not isinstance(row_result, pd.DataFrame) or row_result.empty:
        return None
    if "predicted_anomaly" in row_result.columns:
        flags = row_result["predicted_anomaly"].fillna(0).astype(bool)
        matches = np.flatnonzero(flags.to_numpy())
        if len(matches):
            position = int(matches[0])
            if "row_index" in row_result.columns:
                try:
                    return int(float(row_result.iloc[position]["row_index"]))
                except (TypeError, ValueError):
                    pass
            return position + 1
    if "score" in row_result.columns:
        scores = pd.to_numeric(row_result["score"], errors="coerce")
        if scores.notna().any():
            position = int(np.nanargmax(scores.to_numpy(dtype=float)))
            if "row_index" in row_result.columns:
                try:
                    return int(float(row_result.iloc[position]["row_index"]))
                except (TypeError, ValueError):
                    pass
            return position + 1
    try:
        binary_window_end_row = int(float(details.get("binary_window_end_row")))
    except (TypeError, ValueError):
        binary_window_end_row = 0
    if binary_window_end_row >= 1:
        return binary_window_end_row
    return None


def source_snapshot_fields(
    source_frame: pd.DataFrame | None,
    *,
    detected_row: int | None,
    window_size: int = 1,
) -> dict[str, Any]:
    """Flatten the detection row and retain the full model-input window as JSON."""
    if not isinstance(source_frame, pd.DataFrame) or source_frame.empty or detected_row is None:
        return {}
    try:
        row_number = int(detected_row)
    except (TypeError, ValueError):
        return {}
    if row_number < 1:
        return {}

    end_position = min(row_number, len(source_frame))
    size = max(1, int(window_size))
    start_position = max(0, end_position - size)
    source_window = source_frame.iloc[start_position:end_position]
    if source_window.empty:
        return {}

    source_columns = [str(column) for column in source_frame.columns]
    source_row = source_window.iloc[-1]
    source_row_record = {
        str(column): _json_safe_scalar(value)
        for column, value in source_row.items()
    }
    source_window_records = [
        {str(column): _json_safe_scalar(value) for column, value in record.items()}
        for record in source_window.to_dict(orient="records")
    ]
    flattened_row = {
        f"raw__{column}": source_row_record.get(column)
        for column in source_columns
    }
    return {
        "source_row_number": end_position,
        "source_window_start_row": start_position + 1,
        "source_window_end_row": end_position,
        "source_window_row_count": len(source_window),
        "source_column_count": len(source_columns),
        "source_columns_json": _json_text(source_columns),
        "source_row_json": _json_text(source_row_record),
        "source_window_json": _json_text(source_window_records),
        **flattened_row,
    }


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


def recommendation_for(fault_type: str) -> dict[str, Any]:
    normalized = normalize_fault_type(fault_type)
    guide = PFMEA_GUIDE.get(normalized, PFMEA_GUIDE["유형 분석 대기"])
    rpn, excluded_max_rpn = _second_highest_distinct(guide["rpn_values"])
    risk_level, risk_color, severity = _risk_from_rpn(rpn)
    ng_codes = str(guide["pfmea_ng_codes"])
    return {
        "recommended_action": str(guide["recommended_action"]),
        "disposition_guide": str(guide["disposition_guide"]),
        "recommendation_reason": (
            f"FMEA_배터리팩.xlsx의 Pack_PFMEA {ng_codes} 관련 RPN에서 "
            f"최댓값 {excluded_max_rpn}을 제외하고 두 번째로 큰 서로 다른 값 "
            f"RPN {rpn}({risk_color})을 적용했습니다."
        ),
        "rpn": rpn,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "pfmea_ng_codes": ng_codes,
        "severity": severity,
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

    guide = recommendation_for(fault_type)
    return {
        "fault_type": fault_type,
        "fault_confidence": confidence,
        "fault_probabilities": json.dumps(probabilities, ensure_ascii=False),
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
    source_frame: pd.DataFrame | None = None,
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
    if detected_row is None:
        detected_row = _infer_detected_row(result)
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
    model_name = str(summary.get("model_name", ""))
    model_version = str(summary.get("model_version", ""))
    event_id_parts = [origin, source_file, model_id, model_version, detected_row or 0, detected]
    if occurrence_key:
        event_id_parts.append(occurrence_key)
    event_id = _event_id(*event_id_parts)
    event_mode = str(summary.get("mode", mode)).upper()
    try:
        window_size = max(1, int(details.get("window_size", 1)))
    except (TypeError, ValueError):
        window_size = 1
    snapshot = source_snapshot_fields(
        source_frame,
        detected_row=detected_row,
        window_size=window_size,
    )
    event = {
        "event_id": event_id,
        "detected_at": detected,
        "origin": origin,
        "source_file": source_file,
        "source_path": source_path,
        "mode": event_mode,
        "mode_display": display_mode(event_mode),
        "model_id": model_id,
        "model_name": model_name,
        "model_version": model_version,
        "model_status": status,
        "model_trigger": summary.get("trigger", ""),
        "model_summary_json": _json_text(summary),
        "model_details_json": _json_text(details),
        "detected_row": detected_row if detected_row is not None else np.nan,
        "fire_rate": summary.get("fire_rate", np.nan),
        "fire_rate_percent": _percent_text(summary.get("fire_rate", np.nan)),
        "score_p95": summary.get("score_p95", np.nan),
        "score_max": summary.get("score_max", np.nan),
        "max_consecutive_rows": summary.get("max_consecutive_rows", np.nan),
        "action_status": "신규",
        "final_action": "미결정",
        "assignee": "",
        "action_notes": "",
        "action_updated_at": "",
        **metadata,
        **snapshot,
    }
    event["fault_confidence_percent"] = _percent_text(event.get("fault_confidence"))
    event["risk_label"] = f"위험도 {int(event.get('risk_level', 0) or 0)}"
    return event


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
        live_events["fault_type"] = live_events["fault_type"].map(normalize_fault_type)
        live_guides = pd.DataFrame(
            [recommendation_for(fault_type) for fault_type in live_events["fault_type"]],
            index=live_events.index,
        )
        for column in live_guides.columns:
            live_events[column] = live_guides[column]
        live_events["mode_display"] = live_events["mode"].map(display_mode)
        live_events["risk_label"] = (
            "위험도 "
            + pd.to_numeric(live_events["risk_level"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )
        live_events["fault_confidence_percent"] = live_events["fault_confidence"].map(_percent_text)
        live_events["fire_rate_percent"] = live_events["fire_rate"].map(_percent_text)
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
    events["fault_type"] = events["fault_type"].map(normalize_fault_type)
    refreshed_guides = pd.DataFrame(
        [recommendation_for(fault_type) for fault_type in events["fault_type"]],
        index=events.index,
    )
    for column in refreshed_guides.columns:
        events[column] = refreshed_guides[column]
    events["mode"] = events["mode"].fillna("UNKNOWN").astype(str).str.upper()
    events["mode_display"] = events["mode"].map(display_mode)
    events["risk_label"] = (
        "위험도 "
        + pd.to_numeric(events["risk_level"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    events["fault_confidence_percent"] = events["fault_confidence"].map(_percent_text)
    events["fire_rate_percent"] = events["fire_rate"].map(_percent_text)
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
    ordered_columns = [column for column in FAULT_EVENT_COLUMNS if column in events.columns]
    extra_columns = [column for column in events.columns if column not in ordered_columns]
    return (
        events[ordered_columns + extra_columns]
        .sort_values("detected_at", ascending=False)
        .reset_index(drop=True)
    )
