from __future__ import annotations

from datetime import datetime
import html
import io
import json
from pathlib import Path
import time
from typing import Iterable
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import storage as storage_api
from core.config import APP_ROOT, MODEL_DIR, load_settings, save_settings
from core.data_catalog import (
    audit_data_quality,
    build_catalog,
    detect_sensor_columns,
    read_csv_resilient,
    resolve_time_axis,
)
from core.diagnostic_display import (
    build_normal_reference,
    evaluated_row_positions,
    fault_domain_coverage,
    kpi_log_styler,
    latest_fault_payload,
    latest_scored_prediction,
    sensor_matrix_styler,
)
from core.features import build_sensor_kpis, module_temperature_summary, sensor_deviation_ranking, sensor_snapshot_matrix
from core.fault_log import (
    build_completed_fault_event,
    build_fault_event,
    display_mode,
    extract_fault_metadata,
    load_fault_events,
    parse_probabilities,
    representative_fault_events,
)
from core.kpi_workspace_component import render_kpi_workspace
from core.model_registry import ModelSpec, discover_models, model_inventory, score_dataframe
from core.storage import (
    append_fault_action,
    append_review,
    dataframe_csv_bytes,
    delete_fault_actions,
    delete_fault_events,
    delete_reviews,
    load_fault_actions,
    load_reviews,
    safe_file_name,
    save_batch_result,
    save_uploads,
    upsert_fault_event,
)
from core.visuals import (
    AMBER,
    GRAPHITE,
    RED,
    TEAL,
    batch_status_figure,
    draggable_kpi_workspace_html,
    feature_timeline_figure,
    ranking_bar_figure,
    score_timeline_figure,
    sensor_envelope_figure,
    sensor_heatmap_figure,
)


st.set_page_config(
    page_title="배터리팩 품질 관제",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
:root {
  --ink: #172220;
  --muted: #64736f;
  --line: #d8e0de;
  --panel: #ffffff;
  --canvas: #f3f6f5;
  --teal: #0f766e;
  --teal-dark: #0b625b;
  --teal-soft: #e8f2ef;
  --red: #c43d3d;
  --amber: #a56818;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--canvas) !important;
  color: var(--ink) !important;
}
[data-testid="stHeader"] { background: rgba(243, 246, 245, 0.96) !important; }

/* Preserve the local light theme when Community Cloud applies browser defaults. */
button[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
  background: var(--teal) !important;
  border-color: var(--teal) !important;
  color: #ffffff !important;
}
button[data-testid="stBaseButton-primary"] *,
.stButton > button[kind="primary"] * {
  color: #ffffff !important;
}
button[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
  background: var(--teal-dark) !important;
  border-color: var(--teal-dark) !important;
}
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"],
[data-testid="stMain"] .stButton > button[kind="secondary"],
[data-testid="stMain"] .stDownloadButton > button {
  background: #f9fbfa !important;
  border-color: rgba(23, 34, 32, 0.20) !important;
  color: var(--ink) !important;
  opacity: 1 !important;
}
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] *,
[data-testid="stMain"] .stButton > button[kind="secondary"] *,
[data-testid="stMain"] .stDownloadButton > button * {
  color: var(--ink) !important;
}
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stMain"] .stButton > button[kind="secondary"]:hover,
[data-testid="stMain"] .stDownloadButton > button:hover {
  background: #eef4f2 !important;
  border-color: #91a39e !important;
}
[data-testid="stMain"] button:disabled,
[data-testid="stMain"] button[aria-disabled="true"] {
  background: #f4f7f6 !important;
  border-color: #d4dcda !important;
  color: #7a8783 !important;
  opacity: 1 !important;
}
[data-testid="stMain"] button:disabled *,
[data-testid="stMain"] button[aria-disabled="true"] * {
  color: #7a8783 !important;
}
[data-testid="stMain"] [data-baseweb="input"] > div,
[data-testid="stMain"] [data-baseweb="select"] > div,
[data-testid="stMain"] [data-baseweb="textarea"] > div,
[data-testid="stMain"] input,
[data-testid="stMain"] textarea {
  background: #ffffff !important;
  border-color: #cbd5d2 !important;
  color: var(--ink) !important;
}
[data-testid="stMain"] input::placeholder,
[data-testid="stMain"] textarea::placeholder {
  color: #87938f !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  background-color: var(--teal) !important;
}
[data-baseweb="slider"] [role="slider"] {
  background-color: var(--teal) !important;
  border-color: #f3f7f6 !important;
}
[data-baseweb="checkbox"] [aria-checked="true"],
[data-baseweb="radio"] [aria-checked="true"] {
  background-color: var(--teal) !important;
  border-color: var(--teal) !important;
}
/* Keep the previous frame fully visible while Streamlit replaces fragment data. */
[data-stale="true"] {
  opacity: 1 !important;
}
[data-testid="stSidebar"] { background: #182321; border-right: 1px solid #2d3b38; }
[data-testid="stSidebar"] * { color: #f3f7f6; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color: #c7d2cf !important; }
[data-testid="stSidebar"] input { color: #172220 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #172220 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #f8faf9 !important; border-color: #42534f !important; }
[data-testid="stSidebar"] .stButton > button { background: #263633; border-color: #51625e; color: #f3f7f6; }
[data-testid="stSidebar"] .stButton > button * { color: #f3f7f6 !important; }
.block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1680px; }
.ops-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; border-bottom: 2px solid #20312e; padding: 2px 2px 15px 2px; margin-bottom: 16px; }
.ops-kicker { color: var(--teal); font-size: 12px; font-weight: 800; letter-spacing: 1.2px; }
.ops-title { color: var(--ink); font-size: 28px; font-weight: 760; line-height: 1.15; margin: 3px 0 0; letter-spacing: 0; }
.ops-meta { color: var(--muted); font-size: 12px; text-align: right; line-height: 1.65; }
.section-label { color: var(--ink); font-weight: 760; font-size: 17px; margin: 4px 0 10px; }
.status-line { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 2px 0 12px; }
.status-pill { border: 1px solid var(--line); border-radius: 4px; padding: 5px 9px; font-size: 12px; font-weight: 760; background: #fff; }
.status-normal { color: #0f6b52; border-color: #8ec5b5; background: #edf8f4; }
.status-ng { color: #a42727; border-color: #e1a6a6; background: #fff1f1; }
.status-review { color: #82520e; border-color: #dfc38d; background: #fff8e9; }
.status-info { color: #314a45; border-color: #b8c8c4; background: #f4f8f7; }
.model-note { border-left: 3px solid var(--teal); background: #eef5f3; padding: 10px 12px; color: #30433f; font-size: 13px; margin: 6px 0 14px; }
.live-banner { display: flex; align-items: center; justify-content: space-between; gap: 16px; border: 1px solid #9dc7be; border-left: 5px solid var(--teal); background: #edf8f4; padding: 13px 16px; margin: 4px 0 14px; font-weight: 760; }
.live-banner-review { border-color: #dfc38d; border-left-color: var(--amber); background: #fff8e9; }
.live-banner-alert { border-color: #e1a6a6; border-left-color: var(--red); background: #fff1f1; }
.live-banner small { color: var(--muted); font-weight: 600; }
.product-count-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin: 0 0 18px; }
.product-count-card { position: relative; overflow: hidden; min-height: 128px; border: 1px solid var(--line); border-radius: 7px; padding: 17px 20px 15px; background: #fff; }
.product-count-card::after { content: ""; position: absolute; width: 112px; height: 112px; right: -32px; bottom: -48px; border-radius: 50%; background: var(--count-soft); }
.product-count-card.normal { --count-color: #0f6b52; --count-soft: #dcefe8; border-left: 5px solid var(--count-color); background: #f7fbf9; }
.product-count-card.anomaly { --count-color: #a92d2d; --count-soft: #f6dddd; border-left: 5px solid var(--count-color); background: #fff9f9; }
.product-count-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: #52605c; font-size: 13px; font-weight: 760; }
.product-count-state { display: inline-flex; align-items: center; gap: 6px; color: var(--count-color); font-size: 11px; font-weight: 760; }
.product-count-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--count-color); box-shadow: 0 0 0 4px var(--count-soft); }
.product-count-body { display: flex; align-items: baseline; gap: 7px; margin-top: 13px; color: var(--count-color); position: relative; z-index: 1; }
.product-count-value { font-size: 35px; line-height: 1; font-weight: 800; font-variant-numeric: tabular-nums; }
.product-count-unit { font-size: 14px; font-weight: 760; }
.product-count-meta { color: #6a7773; font-size: 11px; margin-top: 10px; position: relative; z-index: 1; }
.pack-live-toolbar { display:flex; align-items:center; justify-content:space-between; gap:16px; margin:0 0 10px; }
.ops-subtle { color: var(--muted); font-size: 12px; line-height: 1.55; }
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; min-height: 104px; }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--ink); font-size: 21px; white-space: normal; line-height: 1.2; overflow: visible; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; overflow: hidden; }
.stButton > button, .stDownloadButton > button { border-radius: 4px; min-height: 38px; font-weight: 700; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0; height: 42px; padding-left: 8px; padding-right: 8px; color: #42534f !important; font-size: 12px; }
.stTabs [data-baseweb="tab"] * { color: #42534f !important; }
.stTabs [aria-selected="true"] { background: #e8f2ef; color: #0b625b !important; }
.stTabs [aria-selected="true"] * { color: #0b625b !important; font-weight: 760; }
div[data-testid="stAlert"] * { color: #493b1d !important; }
[data-testid="stSidebar"] div[data-testid="stAlert"] { background: #203532; border-color: #3d5a54; }
[data-testid="stSidebar"] div[data-testid="stAlert"] * { color: #e6f2ef !important; }
div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 4px; background: white; }
@media (max-width: 900px) {
  .ops-header { align-items: flex-start; flex-direction: column; }
  .ops-meta { text-align: left; }
  .ops-title { font-size: 23px; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: 0 !important;
    flex: 1 1 calc(50% - 8px) !important;
    width: calc(50% - 8px) !important;
    max-width: calc(50% - 8px) !important;
  }
  .product-count-grid { grid-template-columns: 1fr; }
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=300)
def cached_catalog(settings_json: str) -> pd.DataFrame:
    settings = json.loads(settings_json)
    return build_catalog(settings.get("data_sources", []))


@st.cache_data(show_spinner=False, persist="disk", max_entries=24)
def cached_read_csv(path_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return read_csv_resilient(Path(path_text))


@st.cache_data(show_spinner=False, persist="disk", max_entries=8)
def cached_normal_reference(
    records_json: str,
    reference_schema: str = "sensor_residual_v2",
) -> dict[str, object]:
    del reference_schema
    records = json.loads(records_json)
    return build_normal_reference(records)


@st.cache_data(show_spinner=False, persist="disk", max_entries=16)
def cached_model_score(
    spec_json: str,
    path_text: str,
    modified_ns: int,
    source_file: str,
) -> dict[str, object]:
    """Reuse deterministic file-level inference across tabs, reruns, and restarts."""
    del modified_ns
    spec_payload = json.loads(spec_json)
    spec_payload["supported_modes"] = tuple(spec_payload.get("supported_modes", ()))
    spec = ModelSpec(**spec_payload)
    frame = cached_read_csv(path_text, Path(path_text).stat().st_mtime_ns)
    return score_dataframe(spec, frame, source_file)


def load_path(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    return cached_read_csv(str(path), path.stat().st_mtime_ns)


def load_model_score(spec: ModelSpec, record: pd.Series) -> dict[str, object]:
    path = Path(str(record["path"]))
    return cached_model_score(
        json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True),
        str(path),
        path.stat().st_mtime_ns,
        str(record["file_name"]),
    )


def file_fault_occurrence_key(record: pd.Series, spec: ModelSpec) -> str:
    return "::".join(
        [
            str(record["path"]),
            str(record.get("modified_at", "")),
            spec.model_id,
            spec.sha256,
        ]
    )


def persist_file_fault_event(
    result: dict[str, object],
    *,
    record: pd.Series,
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> dict[str, object] | None:
    occurrence_key = file_fault_occurrence_key(record, spec)

    detected_at = None
    time_labels = measurement_time_labels(frame)
    if not time_labels.empty:
        last_time = str(time_labels.iloc[-1]).strip()
        if last_time and last_time != "-":
            detected_at = last_time

    event = build_fault_event(
        result,
        source_file=str(record["file_name"]),
        source_path=str(record["path"]),
        source_frame=frame,
        mode=str(record["mode"]),
        detected_at=detected_at,
        origin="파일 판정",
        occurrence_key=occurrence_key,
    )
    if event:
        upsert_fault_event(event)
    return event


def status_html(status: str, label: str | None = None) -> str:
    normalized = str(status).upper()
    css = "status-normal" if normalized in {"NORMAL", "PASS", "READY"} else "status-ng" if normalized in {"NG", "NG_REVIEW", "FAIL", "ERROR"} else "status-review" if normalized in {"REVIEW", "FAULT_SCENARIO"} else "status-info"
    text = label or status
    return f'<span class="status-pill {css}">{text}</span>'


def fmt_float(value: object, digits: int = 3, fallback: str = "-") -> str:
    try:
        number = float(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback


def find_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {str(col).lower(): str(col) for col in frame.columns}
    return next((lookup[name.lower()] for name in candidates if name.lower() in lookup), None)


def measurement_time_labels(frame: pd.DataFrame) -> pd.Series:
    _, _, timestamp = resolve_time_axis(frame)
    if timestamp is not None and timestamp.notna().any():
        return timestamp.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")
    return pd.Series([f"행 {i:,}" for i in range(1, len(frame) + 1)], index=frame.index)


def serial_number_labels(frame: pd.DataFrame, source_file: str = "") -> pd.Series:
    serial_col = find_column(frame, "SerialNumber", "Serial_Number", "serial")
    fallback = Path(source_file).stem if source_file else "-"
    if not serial_col:
        return pd.Series(fallback, index=frame.index, dtype="object")

    def normalize(value: object) -> str:
        if pd.isna(value) or str(value).strip() == "":
            return fallback
        try:
            number = float(value)
            if np.isfinite(number) and number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    return frame[serial_col].map(normalize)


def sorted_serial_numbers(values: Iterable[object]) -> list[str]:
    normalized = {
        str(value).strip()
        for value in values
        if str(value).strip() and str(value).strip().casefold() not in {"nan", "none", "<na>"}
    }

    def serial_key(value: str) -> tuple[int, float | str]:
        try:
            return 0, float(value)
        except (TypeError, ValueError):
            return 1, value.casefold()

    return sorted(normalized, key=serial_key)


def battery_pack_schematic_html(
    *,
    position: int,
    total_rows: int,
    phase: str,
    serial_number: str,
    model_name: str = "배터리팩 판정 모델",
    fault_type: str = "",
    confidence: float = float("nan"),
    suspect_sensors: list[str] | None = None,
    event_row: int | None = None,
) -> str:
    """Render a 16-module, 176-cell, 32-temperature-sensor live pack schematic."""
    suspects = {str(sensor).strip().upper() for sensor in (suspect_sensors or []) if str(sensor).strip()}
    safe_serial = html.escape(str(serial_number))
    safe_model_name = html.escape(str(model_name or "배터리팩 판정 모델"))
    safe_fault_type = html.escape(str(fault_type or "불량 유형 확인 필요"))
    confidence_text = f"{confidence:.1%}" if np.isfinite(confidence) else "-"
    safe_suspects = html.escape(", ".join(sorted(suspects)) if suspects else "센서 후보 확인 필요")

    if phase == "fault":
        stage_class = "fault"
        stage_label = "불량 감지 · 관제 계속"
        stage_detail = (
            f"{safe_model_name} 불량 감지 · 유형 판정 완료 · 감지 행 {int(event_row or position):,}"
        )
        cause_html = f"""
          <div class="pack-stop-reason">
            <div class="pack-stop-label">불량 감지 정보</div>
            <div class="pack-stop-title">{safe_fault_type}</div>
            <div class="pack-stop-meta">유형 확신도 {confidence_text} · 의심 센서 {safe_suspects} · 실시간 검사는 계속됩니다.</div>
          </div>
        """
    elif phase == "complete":
        stage_class = "complete"
        stage_label = "검사 완료 · 정상"
        stage_detail = f"전체 측정 구간에서 {safe_model_name} 불량 윈도우가 검출되지 않았습니다."
        cause_html = ""
    elif phase == "idle":
        stage_class = "idle"
        stage_label = "검사 대기"
        stage_detail = f"가동을 누르면 {safe_model_name} 실시간 센서 감시를 시작합니다."
        cause_html = ""
    else:
        stage_class = "monitoring"
        stage_label = "실시간 검사 중"
        stage_detail = f"{safe_model_name} 불량 감시 · 유형 판정 대기"
        cause_html = ""

    def sensor_state(sensor_name: str) -> str:
        if phase == "fault":
            return "sensor-fault" if sensor_name.upper() in suspects else "sensor-healthy"
        if phase == "complete":
            return "sensor-healthy"
        if phase == "idle":
            return "sensor-idle"
        return "sensor-monitoring"

    module_cards: list[str] = []
    for module_number in range(1, 17):
        module = f"M{module_number:02d}"
        module_suspects = sorted(sensor for sensor in suspects if sensor.startswith(module))
        module_class = " affected" if module_suspects else ""
        module_state = "FAULT" if module_suspects else "RUN"
        cells = "".join(
            f'<div class="pack-sensor cell {sensor_state(f"{module}CV{cell:02d}")}" '
            f'title="{module}CV{cell:02d}">C{cell:02d}</div>'
            for cell in range(1, 12)
        )
        temperatures = "".join(
            f'<div class="pack-temp {sensor_state(f"{module}T{sensor:02d}")}" '
            f'title="{module}T{sensor:02d}">T{sensor}</div>'
            for sensor in range(1, 3)
        )
        module_cards.append(
            f'<section class="pack-module{module_class}">'
            f'<div class="pack-module-head"><span>{module}</span>'
            f'<span class="pack-module-state">{module_state}</span></div>'
            f'<div class="pack-cell-grid">{cells}</div>'
            f'<div class="pack-temp-row"><span class="pack-temp-label">TEMP</span>'
            f'{temperatures}</div></section>'
        )

    return f"""
    <style>
      .pack-live-shell {{ --pack-blue:#2563eb; --pack-green:#16a34a; --pack-red:#dc2626; --pack-ink:#e8f1f5;
        background:#08131d; border:1px solid #263948; border-radius:8px; padding:18px; color:var(--pack-ink);
        box-shadow:0 10px 28px rgba(14,31,43,.18); }}
      .pack-live-status {{ display:flex; align-items:center; justify-content:space-between; gap:18px; border:1px solid #2d4556;
        border-left:5px solid var(--pack-blue); border-radius:6px; background:#0e1d28; padding:14px 16px; margin-bottom:14px; }}
      .pack-live-status.fault {{ border-color:#7f1d1d; border-left-color:var(--pack-red); background:#2b1015; }}
      .pack-live-status.complete {{ border-color:#166534; border-left-color:var(--pack-green); background:#0b271d; }}
      .pack-live-title {{ display:flex; align-items:center; gap:10px; font-size:17px; font-weight:800; }}
      .pack-live-light {{ width:12px; height:12px; border-radius:50%; background:var(--pack-blue); box-shadow:0 0 0 5px rgba(37,99,235,.17); }}
      .pack-live-status.monitoring {{ border-color:#166534; border-left-color:var(--pack-green); background:#0b271d; }}
      .pack-live-status.monitoring .pack-live-light {{ background:var(--pack-green); box-shadow:0 0 0 5px rgba(22,163,74,.18); animation:pack-green-pulse 1.35s ease-in-out infinite; }}
      .pack-live-status.fault .pack-live-light {{ background:var(--pack-red); box-shadow:0 0 0 5px rgba(220,38,38,.18); animation:none; }}
      .pack-live-status.complete .pack-live-light {{ background:var(--pack-green); box-shadow:0 0 0 5px rgba(22,163,74,.18); animation:pack-green-pulse 1.6s ease-in-out infinite; }}
      .pack-live-detail {{ color:#9fb3bf; font-size:12px; margin-top:4px; }}
      .pack-live-position {{ color:#bfd0d8; font-size:12px; text-align:right; line-height:1.55; }}
      .pack-stop-reason {{ border:1px solid #991b1b; border-left:5px solid var(--pack-red); border-radius:6px; background:#3a1017;
        padding:14px 16px; margin-bottom:14px; }}
      .pack-stop-label {{ color:#fca5a5; font-size:11px; font-weight:800; letter-spacing:.08em; }}
      .pack-stop-title {{ color:#fff; font-size:21px; font-weight:850; margin-top:4px; }}
      .pack-stop-meta {{ color:#fecaca; font-size:12px; margin-top:6px; overflow-wrap:anywhere; }}
      .pack-overview {{ display:flex; align-items:center; justify-content:space-between; gap:14px; margin:2px 2px 12px; }}
      .pack-overview strong {{ font-size:15px; }}
      .pack-legend {{ display:flex; flex-wrap:wrap; gap:12px; color:#9fb3bf; font-size:11px; }}
      .pack-legend span {{ display:inline-flex; align-items:center; gap:6px; }}
      .pack-legend i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
      .pack-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
      .pack-module {{ min-width:0; border:1px solid #2b4050; border-radius:6px; background:#0d1b25; padding:10px; }}
      .pack-module.affected {{ border-color:#ef4444; box-shadow:0 0 0 1px rgba(239,68,68,.32),0 0 18px rgba(220,38,38,.16); }}
      .pack-module-head {{ display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:12px; font-weight:850; margin-bottom:8px; }}
      .pack-module-state {{ color:#7f96a3; font-size:9px; letter-spacing:.08em; }}
      .pack-module.affected .pack-module-state {{ color:#fca5a5; }}
      .pack-cell-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:4px; }}
      .pack-sensor {{ height:22px; border-radius:3px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:8px; font-weight:750; border:1px solid rgba(255,255,255,.13); }}
      .pack-temp-row {{ display:flex; align-items:center; gap:7px; border-top:1px solid #263946; margin-top:8px; padding-top:7px; }}
      .pack-temp-label {{ color:#758b97; font-size:8px; font-weight:800; margin-right:auto; }}
      .pack-temp {{ width:27px; height:27px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#fff; font-size:8px; font-weight:800; border:1px solid rgba(255,255,255,.16); }}
      .sensor-idle {{ background:var(--pack-blue); }}
      .sensor-monitoring {{ background:var(--pack-green); animation:pack-green-pulse 1.35s ease-in-out infinite; }}
      .sensor-healthy {{ background:var(--pack-green); animation:pack-green-pulse 1.55s ease-in-out infinite; }}
      .sensor-fault {{ background:var(--pack-red); color:#fff; box-shadow:0 0 0 2px rgba(254,202,202,.32),0 0 13px rgba(239,68,68,.92); animation:none; }}
      @keyframes pack-blue-pulse {{ 0%,100%{{opacity:.50;filter:saturate(.85)}} 50%{{opacity:1;filter:saturate(1.35);box-shadow:0 0 9px rgba(96,165,250,.70)}} }}
      @keyframes pack-green-pulse {{ 0%,100%{{opacity:.62}} 50%{{opacity:1;box-shadow:0 0 8px rgba(74,222,128,.55)}} }}
      @media(max-width:1050px) {{ .pack-grid{{grid-template-columns:repeat(2,minmax(0,1fr));}} }}
      @media(max-width:650px) {{ .pack-live-status,.pack-overview{{align-items:flex-start;flex-direction:column;}} .pack-live-position{{text-align:left;}} .pack-grid{{grid-template-columns:1fr;}} }}
    </style>
    <div class="pack-live-shell">
      <div class="pack-live-status {stage_class}">
        <div>
          <div class="pack-live-title"><span class="pack-live-light"></span>{stage_label}</div>
          <div class="pack-live-detail">{stage_detail}</div>
        </div>
        <div class="pack-live-position">Serial {safe_serial}<br>측정된 데이터 {position:,}</div>
      </div>
      {cause_html}
      <div class="pack-overview">
        <strong>16 MODULE · 176 CELL VOLTAGE · 32 TEMPERATURE</strong>
        <div class="pack-legend">
          <span><i style="background:#2563eb"></i>검사 대기</span>
          <span><i style="background:#16a34a"></i>가동·정상 센서</span>
          <span><i style="background:#dc2626"></i>불량 원인 센서</span>
        </div>
      </div>
      <div class="pack-grid">{''.join(module_cards)}</div>
    </div>
    """


def advance_shared_live_playback(
    *,
    total_rows: int,
    row_result: pd.DataFrame | None,
    refresh_seconds: float,
    step_size: int,
) -> int:
    """Advance the shared live cursor once per wall-clock tick across all tabs."""
    position = max(
        1,
        min(int(st.session_state.get("ops_live_position", 1)), int(total_rows)),
    )
    if not st.session_state.get("ops_live_running", False):
        return position

    interval = max(float(refresh_seconds), 0.05)
    now = time.monotonic()
    next_tick = float(st.session_state.get("ops_live_next_tick", 0.0))
    if next_tick <= 0:
        st.session_state["ops_live_next_tick"] = now + interval
        return position
    if now < next_tick:
        return position

    target = min(int(total_rows), position + max(1, int(step_size)))
    st.session_state["ops_live_next_tick"] = now + interval

    st.session_state["ops_live_position"] = target
    if target >= int(total_rows):
        st.session_state["ops_live_running"] = False
    return target


def begin_live_fault_run() -> str:
    """Start a distinct inspection run so a repeated fault becomes a new event."""
    run_id = str(time.time_ns())
    st.session_state["ops_live_run_id"] = run_id
    st.session_state["logged_live_fault_event_ids"] = set()
    return run_id


def persist_completed_live_fault_event(
    result: dict[str, object] | None,
    *,
    record: pd.Series,
    frame: pd.DataFrame,
    position: int,
    total_rows: int,
) -> dict[str, object] | None:
    """Persist one final fault event only after every source row has been replayed."""
    run_id = str(st.session_state.get("ops_live_run_id", "")).strip()
    if not run_id:
        run_id = begin_live_fault_run()
    event = build_completed_fault_event(
        result,
        position=position,
        total_rows=total_rows,
        source_file=str(record["file_name"]),
        source_path=str(record["path"]),
        source_frame=frame,
        mode=str(record["mode"]),
        origin="실시간 파일 완료 판정",
        occurrence_key=run_id,
    )
    if not event:
        return None

    logged_ids = st.session_state.setdefault("logged_live_fault_event_ids", set())
    event_id = str(event.get("event_id", "")).strip()
    if not event_id or event_id in logged_ids:
        return None

    event_row = pd.to_numeric(
        pd.Series([event.get("detected_row")]),
        errors="coerce",
    ).iloc[0]
    time_labels = measurement_time_labels(frame)
    if np.isfinite(event_row) and 1 <= int(event_row) <= len(time_labels):
        event["detected_at"] = str(time_labels.iloc[int(event_row) - 1])

    upsert_fault_event(event)
    logged_ids.add(event_id)
    return event


def measurement_kpi_log(
    frame: pd.DataFrame,
    row_positions: np.ndarray | list[int],
    source_file: str = "",
) -> pd.DataFrame:
    positions = np.asarray(row_positions, dtype=int)
    if positions.size == 0:
        return pd.DataFrame(
            columns=[
                "측정 시각",
                "Serial Num",
                "셀 전압 평균 (V)",
                "셀 전압 편차 (V)",
                "온도 평균 (°C)",
                "온도 범위 (°C)",
                "온도 편차 (°C)",
            ]
        )

    view = frame.iloc[positions]
    kpis = build_sensor_kpis(view)
    range_text = [
        f"{low:.2f} °C ~ {high:.2f} °C" if np.isfinite(low) and np.isfinite(high) else "-"
        for low, high in zip(kpis["temp_min"], kpis["temp_max"])
    ]
    return pd.DataFrame(
        {
            "측정 시각": measurement_time_labels(frame).iloc[positions].to_numpy(),
            "Serial Num": serial_number_labels(frame, source_file).iloc[positions].to_numpy(),
            "셀 전압 평균 (V)": kpis["cv_mean"].round(4).to_numpy(),
            "셀 전압 편차 (V)": kpis["cv_std"].round(5).to_numpy(),
            "온도 평균 (°C)": kpis["temp_mean"].round(2).to_numpy(),
            "온도 범위 (°C)": range_text,
            "온도 편차 (°C)": kpis["temp_std"].round(3).to_numpy(),
        }
    )


def recent_measurement_log(
    frame: pd.DataFrame,
    end_position: int,
    rows: int = 12,
    source_file: str = "",
) -> pd.DataFrame:
    end_position = max(1, min(int(end_position), len(frame)))
    start = max(0, end_position - int(rows))
    positions = np.arange(start, end_position, dtype=int)[::-1]
    return measurement_kpi_log(frame, positions, source_file)


def operational_event_log(
    frame: pd.DataFrame,
    end_position: int,
    policy: dict[str, object],
) -> pd.DataFrame:
    """Build transparent data-quality events without making a model prediction."""
    end_position = max(1, min(int(end_position), len(frame)))
    view = frame.iloc[:end_position]
    cv_cols, temp_cols = detect_sensor_columns(view.columns)
    labels = measurement_time_labels(frame).iloc[:end_position]
    events: list[dict[str, object]] = []

    _, _, timestamp = resolve_time_axis(view)
    if timestamp is not None:
        gaps = timestamp.diff().dt.total_seconds()
        gap_limit = float(policy.get("time_gap_seconds", 30.0))
        for idx in gaps[gaps.gt(gap_limit)].index:
            events.append(
                {
                    "순서": int(view.index.get_loc(idx)) + 1,
                    "측정 시각": labels.loc[idx],
                    "이벤트": "시간 간격 초과",
                    "세부": f"이전 행과 {gaps.loc[idx]:.1f}초 간격",
                    "우선순위": "검토",
                }
            )

    sensor_cols = cv_cols + temp_cols
    if sensor_cols:
        numeric = view[sensor_cols].apply(pd.to_numeric, errors="coerce")
        missing_count = numeric.isna().sum(axis=1)
        for idx in missing_count[missing_count.gt(0)].index:
            events.append(
                {
                    "순서": int(view.index.get_loc(idx)) + 1,
                    "측정 시각": labels.loc[idx],
                    "이벤트": "센서 결측",
                    "세부": f"{int(missing_count.loc[idx])}개 센서 값 누락",
                    "우선순위": "높음",
                }
            )

    if cv_cols:
        voltage = view[cv_cols].apply(pd.to_numeric, errors="coerce")
        bad_voltage = voltage.lt(float(policy.get("min_cell_voltage", 1.5))) | voltage.gt(
            float(policy.get("max_cell_voltage", 5.0))
        )
        bad_count = bad_voltage.sum(axis=1)
        for idx in bad_count[bad_count.gt(0)].index:
            events.append(
                {
                    "순서": int(view.index.get_loc(idx)) + 1,
                    "측정 시각": labels.loc[idx],
                    "이벤트": "전압 물리 범위 이탈",
                    "세부": f"{int(bad_count.loc[idx])}개 셀",
                    "우선순위": "긴급",
                }
            )

    if temp_cols:
        temperature = view[temp_cols].apply(pd.to_numeric, errors="coerce")
        bad_temp = temperature.lt(float(policy.get("min_temperature", -40.0))) | temperature.gt(
            float(policy.get("max_temperature", 100.0))
        )
        bad_count = bad_temp.sum(axis=1)
        for idx in bad_count[bad_count.gt(0)].index:
            events.append(
                {
                    "순서": int(view.index.get_loc(idx)) + 1,
                    "측정 시각": labels.loc[idx],
                    "이벤트": "온도 물리 범위 이탈",
                    "세부": f"{int(bad_count.loc[idx])}개 센서",
                    "우선순위": "긴급",
                }
            )

    if not events:
        return pd.DataFrame(columns=["순서", "측정 시각", "이벤트", "세부", "우선순위"])
    return (
        pd.DataFrame(events)
        .sort_values(["순서", "우선순위"], ascending=[False, True])
        .drop_duplicates(["순서", "이벤트"])
        .head(300)
        .reset_index(drop=True)
    )


def record_label(row: pd.Series) -> str:
    return f"[{row['source_name']}] {row['file_name']} · {display_mode(row['mode'])} · {row['rows']:,}행"


def translated_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "source_name": "데이터 소스",
        "source_role": "역할",
        "file_name": "파일",
        "mode": "충·방전",
        "label_hint": "라벨 힌트",
        "schema": "스키마",
        "rows": "행 수",
        "columns": "열 수",
        "cv_sensors": "전압 센서",
        "temp_sensors": "온도 센서",
        "size_mb": "용량(MB)",
        "readable": "읽기 가능",
        "modified_at": "수정 시각",
        "issue": "이슈",
    }
    shown = [c for c in columns if c in frame.columns]
    translated = frame[shown].copy()
    if "mode" in translated.columns:
        translated["mode"] = translated["mode"].map(display_mode)
    return translated.rename(columns=columns)


def pfmea_risk_label(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"위험도 {int(numeric)}" if np.isfinite(numeric) else "위험도 0"


def pfmea_fault_table_styler(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Color only the PFMEA risk cells using the workbook's red/yellow/white bands."""

    def style_row(row: pd.Series) -> list[str]:
        risk_text = str(row.get("위험도", "위험도 0"))
        if risk_text.endswith("2"):
            style = "background-color:#FECACA;color:#991B1B;font-weight:700;"
        elif risk_text.endswith("1"):
            style = "background-color:#FEF3C7;color:#854D0E;font-weight:700;"
        else:
            style = "background-color:#FFFFFF;color:#172220;font-weight:600;"
        return [style if column in {"위험도", "RPN"} else "" for column in row.index]

    return frame.style.apply(style_row, axis=1)


def translated_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the seven operator-facing review fields in a fixed Korean layout."""
    working = frame.copy()
    if "serial_number" not in working.columns:
        working["serial_number"] = working.get("lot_id", "")
    elif "lot_id" in working.columns:
        working["serial_number"] = working["serial_number"].astype("string").fillna("")
        serial_text = working["serial_number"].str.strip()
        working.loc[serial_text.eq(""), "serial_number"] = working.loc[
            serial_text.eq(""),
            "lot_id",
        ]
    columns = {
        "reviewed_at": "검토 시간",
        "reviewer": "작업자",
        "serial_number": "Serial Num",
        "human_label": "현장 판정",
        "fault_type": "불량 유형",
        "notes": "검토 메모",
        "severity": "조치 우선순위",
    }
    shown = [column for column in columns if column in working.columns]
    translated = working[shown].copy()
    if "reviewed_at" in translated.columns:
        original = translated["reviewed_at"].astype("string")
        parsed = pd.to_datetime(translated["reviewed_at"], errors="coerce")
        translated["reviewed_at"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").where(
            parsed.notna(),
            original,
        )
    return translated.rename(columns=columns)


def is_operational_model(spec: ModelSpec) -> bool:
    """A model is selectable only after explicit deployment approval."""
    try:
        manifest_path = spec.root_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            bool(spec.healthy)
            and bool(manifest.get("enabled", False))
            and str(manifest.get("deployment_status", "candidate")).lower()
            in {"approved", "production"}
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def update_live_file_verdict(
    verdicts: dict[str, str],
    file_key: str,
    *,
    anomaly_detected: bool,
    playback_completed: bool,
) -> None:
    """Store one final OK/NG verdict per CSV without double counting replays."""
    if anomaly_detected:
        verdicts[file_key] = "NG"
    elif playback_completed and file_key not in verdicts:
        verdicts[file_key] = "NORMAL"


def live_file_verdict_counts(verdicts: dict[str, str]) -> tuple[int, int]:
    normal_count = sum(verdict == "NORMAL" for verdict in verdicts.values())
    anomaly_count = sum(verdict == "NG" for verdict in verdicts.values())
    return normal_count, anomaly_count


def anomaly_window_coverage(
    predicted_anomaly: object,
    end_position: int,
    window_size: int,
) -> tuple[np.ndarray, int]:
    """Return the union of anomalous trailing windows and their cumulative count."""
    end_position = max(0, int(end_position))
    window_size = max(1, int(window_size))
    flags = pd.Series(predicted_anomaly).iloc[:end_position].fillna(0).astype(bool).to_numpy()
    window_count = int(flags.sum())
    if not len(flags) or window_count == 0:
        return np.zeros(len(flags), dtype=bool), window_count

    difference = np.zeros(len(flags) + 1, dtype=np.int32)
    for endpoint in np.flatnonzero(flags):
        start = max(0, int(endpoint) - window_size + 1)
        difference[start] += 1
        difference[int(endpoint) + 1] -= 1
    return np.cumsum(difference[:-1]) > 0, window_count


def live_serial_number(frame: pd.DataFrame, file_name: str) -> str:
    serial_col = next(
        (
            col
            for col in frame.columns
            if str(col).replace("_", "").replace(" ", "").casefold() == "serialnumber"
        ),
        None,
    )
    if serial_col is not None:
        values = frame[serial_col].dropna()
        if not values.empty:
            value = values.iloc[0]
            try:
                numeric = float(value)
                if np.isfinite(numeric) and numeric.is_integer():
                    return str(int(numeric))
            except (TypeError, ValueError):
                pass
            text = str(value).strip()
            if text:
                return text
    return Path(file_name).stem


def deployment_state(spec: ModelSpec) -> str:
    if not spec.healthy:
        return "ERROR"
    return "ACTIVE" if is_operational_model(spec) else "CANDIDATE"


def install_model_package(uploaded) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded.getvalue())) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            manifests = [m for m in members if Path(m.filename).name == "manifest.json"]
            if len(manifests) != 1:
                return False, "ZIP 안에 manifest.json이 정확히 하나 있어야 합니다."
            raw_manifest = json.loads(archive.read(manifests[0]).decode("utf-8"))
            model_id = safe_file_name(str(raw_manifest.get("model_id", ""))).replace(".", "_")
            if not model_id:
                return False, "manifest.json에 model_id가 필요합니다."
            target = MODEL_DIR / model_id
            if target.exists():
                return False, f"같은 model_id가 이미 있습니다: {model_id}"
            prefix = Path(manifests[0].filename).parent
            target.mkdir(parents=True, exist_ok=False)
            for member in members:
                relative = Path(member.filename)
                try:
                    relative = relative.relative_to(prefix)
                except ValueError:
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.read(member))
        return True, f"모델 패키지를 등록했습니다: {model_id}"
    except Exception as exc:
        return False, f"모델 패키지 등록 실패: {type(exc).__name__}: {exc}"


settings = load_settings()
settings_payload = json.dumps(settings, ensure_ascii=False, sort_keys=True)
catalog = cached_catalog(settings_payload)
models = discover_models()
model_by_id = {spec.model_id: spec for spec in models}
operational_models = [spec for spec in models if is_operational_model(spec)]
candidate_models = [
    spec
    for spec in models
    if spec.healthy and spec.enabled and not is_operational_model(spec)
]
selectable_models = sorted(
    [spec for spec in models if spec.healthy and spec.enabled],
    key=lambda spec: (
        not is_operational_model(spec),
        spec.name,
        spec.version,
    ),
)


with st.sidebar:
    st.markdown("### BATTERY QA CONTROL")
    st.caption("데이터 검토 및 모델 연동 워크스테이션")
    st.divider()
    if selectable_models:
        def model_option_label(model_id: str) -> str:
            spec = model_by_id[model_id]
            status = "운영" if is_operational_model(spec) else "검증 후보"
            return f"[{status}] {spec.name} · {spec.version}"

        active_model_id = st.selectbox(
            "판정 모델",
            options=[spec.model_id for spec in selectable_models],
            index=0,
            format_func=model_option_label,
            key="active_model_selector_v2",
        )
        active_model = model_by_id[active_model_id]
        if not is_operational_model(active_model):
            st.warning("검증 후보 모델입니다. 운영 승인 모델을 대체하지 않습니다.")
    else:
        active_model = None
        st.info("선택 가능한 모델이 없습니다. 현재는 데이터 점검 모드입니다.")

    readable_catalog = catalog[catalog.get("readable", pd.Series(dtype=bool)).eq(True)].reset_index(drop=True) if not catalog.empty else catalog
    if not readable_catalog.empty:
        default_file_position = next(
            (
                i
                for i, name in enumerate(readable_catalog["file_name"].astype(str))
                if "Test09_NG_dchg" in name
            ),
            next((i for i, role in enumerate(readable_catalog["source_role"]) if role == "test"), 0),
        )
        selected_position = st.selectbox(
            "분석 파일",
            options=list(range(len(readable_catalog))),
            index=default_file_position,
            format_func=lambda i: record_label(readable_catalog.iloc[i]),
        )
        selected_record = readable_catalog.iloc[int(selected_position)]
    else:
        selected_record = None
        st.warning("등록된 CSV가 없습니다.")

    st.divider()
    st.markdown("#### 실시간 재생 설정")
    live_window_size = st.slider(
        "최근 표시 행",
        min_value=60,
        max_value=1200,
        value=240,
        step=20,
        help="실시간 그래프에 표시할 최근 측정 구간입니다.",
    )
    live_step_size = st.slider(
        "한 번에 진행할 행",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
        key="live_step_size_v2",
        help="실시간 측정처럼 기본적으로 한 행씩 진행합니다.",
    )
    live_refresh_seconds = st.slider(
        "자동 재생 간격(초)",
        min_value=0.2,
        max_value=5.0,
        value=0.8,
        step=0.1,
    )
    live_log_rows = st.slider(
        "최근 로그 행",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
    )

    st.divider()
    if st.button("데이터·모델 새로고침", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption(
        f"CSV {len(catalog):,}개 · 운영 모델 {len(operational_models):,}개 · "
        f"검증 후보 {len(candidate_models):,}개"
    )


if active_model is None:
    st.session_state.pop("active_analysis", None)
    st.session_state.pop("batch_result", None)
    st.session_state.pop("batch_path", None)

selected_model_role = (
    "운영 모델"
    if active_model is not None and is_operational_model(active_model)
    else "검증 후보"
    if active_model is not None
    else "판정 모델"
)

st.markdown(
    f"""
    <div class="ops-header">
      <div>
        <div class="ops-kicker">PACK QUALITY OPERATIONS</div>
        <div class="ops-title">배터리팩 품질 관제 대시보드</div>
      </div>
      <div class="ops-meta">
        기준 시각 {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
        {selected_model_role} {active_model.name if active_model else '미등록 · 데이터 점검 모드'}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_live_visual, tab_live, tab_daily, tab_fault, tab_data_status, tab_models, tab_history = st.tabs(
    ["실시간", "실시간 요약", "일별 로그", "불량 로그", "데이터 현황", "모델 관리", "학습 이력"],
    key="main_dashboard_tabs",
    default="실시간",
    on_change="rerun",
)
tab_diagnosis = tab_data_status
tab_review = tab_daily

reference_records = []
if (tab_live.open or tab_daily.open) and not readable_catalog.empty and "source_role" in readable_catalog.columns:
    for _, record in readable_catalog[readable_catalog["source_role"].eq("train")].iterrows():
        reference_records.append(
            {
                "path": str(record["path"]),
                "mode": str(record.get("mode", "UNKNOWN")),
                "modified_at": str(record.get("modified_at", "")),
            }
        )
normal_reference = (
    cached_normal_reference(
        json.dumps(reference_records, ensure_ascii=False, sort_keys=True),
        "sensor_residual_v2",
    )
    if reference_records
    else {}
)

if selected_record is not None:
    ops_model_signature = (
        f"{active_model.model_id}::{active_model.sha256}"
        if active_model is not None
        else "model-not-connected"
    )
    ops_live_signature = (
        f"{selected_record['path']}::{selected_record['modified_at']}::{ops_model_signature}"
    )
    if st.session_state.get("ops_live_signature") != ops_live_signature:
        st.session_state["ops_live_signature"] = ops_live_signature
        st.session_state["ops_live_position"] = 1
        st.session_state["ops_live_running"] = False
        st.session_state["ops_live_next_tick"] = 0.0
        begin_live_fault_run()
    elif not st.session_state.get("ops_live_run_id"):
        begin_live_fault_run()

    playback_config_signature = (
        f"{int(live_step_size)}::{float(live_refresh_seconds):.3f}"
    )
    if st.session_state.get("ops_live_playback_config") != playback_config_signature:
        st.session_state["ops_live_playback_config"] = playback_config_signature
        if st.session_state.get("ops_live_running", False):
            st.session_state["ops_live_next_tick"] = (
                time.monotonic() + float(live_refresh_seconds)
            )

    ops_live_df = load_path(str(selected_record["path"]))
    if not isinstance(ops_live_df.index, pd.RangeIndex) or ops_live_df.index.start != 0:
        ops_live_df = ops_live_df.reset_index(drop=True)
    ops_total_rows = len(ops_live_df)
    ops_live_time_labels = measurement_time_labels(ops_live_df)
    ops_model_key = (
        f"{selected_record['path']}::{selected_record['modified_at']}::"
        f"{active_model.model_id}::{active_model.sha256}"
        if active_model is not None
        else ""
    )

    heartbeat_interval = (
        float(live_refresh_seconds)
        if st.session_state.get("ops_live_running", False)
        else None
    )

    @st.fragment(run_every=heartbeat_interval)
    def run_global_live_heartbeat() -> None:
        cached_model = st.session_state.get("live_model_analysis")
        model_result = (
            cached_model.get("result")
            if isinstance(cached_model, dict) and cached_model.get("key") == ops_model_key
            else None
        )
        row_result = (
            model_result.get("row_result", pd.DataFrame())
            if isinstance(model_result, dict)
            else pd.DataFrame()
        )
        current_position = advance_shared_live_playback(
            total_rows=ops_total_rows,
            row_result=row_result,
            refresh_seconds=float(live_refresh_seconds),
            step_size=int(live_step_size),
        )
        persist_completed_live_fault_event(
            model_result,
            record=selected_record,
            frame=ops_live_df,
            position=current_position,
            total_rows=ops_total_rows,
        )

    run_global_live_heartbeat()


with tab_live:
    st.markdown('<div class="section-label">실시간 배터리팩 운전 모니터링</div>', unsafe_allow_html=True)
    if not tab_live.open:
        pass
    elif selected_record is None:
        st.info("재생할 Train 또는 Test 파일을 선택하세요.")
    else:
        try:
            live_df = ops_live_df
            live_signature = f"{selected_record['path']}::{selected_record['modified_at']}"
            if st.session_state.get("live_signature") != live_signature:
                st.session_state["live_signature"] = live_signature
                st.session_state.pop("live_model_analysis", None)
                st.session_state["live_events"] = operational_event_log(
                    live_df,
                    len(live_df),
                    settings["quality_policy"],
                )

            total_rows = len(live_df)
            with st.container(horizontal=True, gap="small"):
                start_clicked = st.button(
                    "재생",
                    icon=":material/play_arrow:",
                    type="primary",
                    width="content",
                    key="live_start",
                )
                stop_clicked = st.button(
                    "정지",
                    icon=":material/stop:",
                    width="content",
                    key="live_stop",
                )
                step_clicked = st.button(
                    f"{int(live_step_size)}행 진행",
                    icon=":material/skip_next:",
                    width="content",
                    key="live_step",
                )
                reset_clicked = st.button(
                    "초기화",
                    icon=":material/restart_alt:",
                    width="content",
                    key="live_reset",
                )
            if reset_clicked:
                st.session_state["ops_live_position"] = 1
                st.session_state["ops_live_running"] = False
                st.session_state["ops_live_next_tick"] = 0.0
                begin_live_fault_run()
            elif stop_clicked:
                st.session_state["ops_live_running"] = False
                st.session_state["ops_live_next_tick"] = 0.0
            elif step_clicked:
                st.session_state["ops_live_running"] = False
                st.session_state["ops_live_next_tick"] = 0.0
                st.session_state["ops_live_position"] = min(
                    total_rows,
                    int(st.session_state.get("ops_live_position", 1)) + int(live_step_size),
                )
            elif start_clicked:
                if int(st.session_state.get("ops_live_position", 1)) >= total_rows:
                    st.session_state["ops_live_position"] = 1
                    begin_live_fault_run()
                st.session_state["ops_live_running"] = True
                st.session_state["ops_live_next_tick"] = time.monotonic() + float(live_refresh_seconds)
                # Rebuild the global heartbeat once with run_every enabled.
                st.rerun()

            live_model_result = None
            live_time_labels = ops_live_time_labels
            if active_model is not None:
                model_live_key = f"{live_signature}::{active_model.model_id}::{active_model.sha256}"
                cached_live_model = st.session_state.get("live_model_analysis")
                if not cached_live_model or cached_live_model.get("key") != model_live_key:
                    with st.spinner("운영 승인 모델의 실시간 점수를 준비하고 있습니다."):
                        try:
                            cached_live_model = {
                                "key": model_live_key,
                                "result": load_model_score(active_model, selected_record),
                            }
                            st.session_state["live_model_analysis"] = cached_live_model
                        except Exception as exc:
                            st.error(f"실시간 모델 준비 실패: {type(exc).__name__}: {exc}")
                            cached_live_model = None
                if cached_live_model:
                    live_model_result = cached_live_model["result"]

            live_run_every = float(live_refresh_seconds) if st.session_state.get("ops_live_running", False) else None

            @st.fragment(run_every=live_run_every)
            def render_live_panel() -> None:
                position = max(
                    1,
                    min(
                        int(st.session_state.get("ops_live_position", 1)),
                        total_rows,
                    ),
                )
                start = max(0, position - int(live_window_size))
                window_df = live_df.iloc[start:position].copy()
                window_kpis = build_sensor_kpis(window_df)
                measurement_time = live_time_labels.iloc[position - 1]

                current_events = st.session_state.get("live_events", pd.DataFrame())
                seen_events = (
                    current_events[current_events["순서"].le(position)].copy()
                    if isinstance(current_events, pd.DataFrame) and not current_events.empty
                    else pd.DataFrame()
                )
                latest_has_event = bool(not seen_events.empty and int(seen_events["순서"].max()) == position)

                current_flag = False
                current_score = float("nan")
                latest_scored_row: int | None = None
                bad_window_count = 0
                window_anomaly_mask = np.zeros(len(window_df), dtype=bool)
                visible_fault_domains = fault_domain_coverage(
                    live_model_result,
                    total_rows,
                    end_position=position,
                )
                if live_model_result is not None:
                    row_result = live_model_result["row_result"]
                    current_flag, current_score, latest_scored_row = latest_scored_prediction(
                        row_result,
                        position,
                    )
                    model_window_size = int(live_model_result.get("details", {}).get("window_size", 100))
                    anomaly_coverage, bad_window_count = anomaly_window_coverage(
                        row_result["predicted_anomaly"],
                        position,
                        model_window_size,
                    )
                    window_anomaly_mask = anomaly_coverage[start:position]
                    persist_completed_live_fault_event(
                        live_model_result,
                        record=selected_record,
                        frame=live_df,
                        position=position,
                        total_rows=total_rows,
                    )
                    warmup_rows = int(live_model_result.get("details", {}).get("warmup_rows", 0))
                    if position <= warmup_rows:
                        banner_class = "live-banner live-banner-review"
                        banner_text = f"1단계 판정 준비 중 · {position}/{warmup_rows + 1}행"
                    elif current_flag:
                        banner_class = "live-banner live-banner-alert"
                        banner_text = f"{active_model.name} 불량 감지"
                    else:
                        banner_class = "live-banner"
                        banner_text = f"{active_model.name} 정상"
                    score_label = (
                        "불량 확률"
                        if active_model.model_id == "lstm_two_stage_quality_v1"
                        else "판정 점수"
                    )
                    score_note = (
                        f"{score_label} {current_score:.1%}"
                        if np.isfinite(current_score)
                        else f"{score_label} 계산 대기"
                    )
                    evaluated_note = (
                        f"최근 평가 {latest_scored_row:,}행"
                        if latest_scored_row is not None
                        else "평가 윈도우 준비 중"
                    )
                    model_note = (
                        f"{evaluated_note} · {score_note} · "
                        f"{active_model.name} {active_model.version}"
                    )
                elif latest_has_event:
                    banner_class = "live-banner live-banner-alert"
                    banner_text = "입력 데이터 품질 이벤트 감지"
                    model_note = "모델 미연결 · 데이터 품질 기준"
                else:
                    banner_class = "live-banner live-banner-review"
                    banner_text = "센서 데이터 모니터링 중"
                    model_note = "최종 모델 미연결 · 정상/불량 예측 미실행"

                st.markdown(
                    f"<div class='{banner_class}'><span>{banner_text}</span><small>{measurement_time} · {model_note}</small></div>",
                    unsafe_allow_html=True,
                )

                if active_model is not None and live_model_result is not None:
                    counter_scope = f"{active_model.model_id}::{active_model.sha256}"
                    verdicts_by_model = st.session_state.setdefault("live_file_verdicts_by_model", {})
                    file_verdicts = verdicts_by_model.setdefault(counter_scope, {})
                    file_key = str(Path(selected_record["path"]).resolve()).casefold()
                    update_live_file_verdict(
                        file_verdicts,
                        file_key,
                        anomaly_detected=bad_window_count > 0,
                        playback_completed=position >= total_rows,
                    )
                    normal_file_count, anomaly_file_count = live_file_verdict_counts(file_verdicts)

                    total_verdict_count = normal_file_count + anomaly_file_count
                    st.markdown(
                        f"""
                        <div class="product-count-grid">
                          <div class="product-count-card normal">
                            <div class="product-count-head">
                              <span>정상 제품</span>
                              <span class="product-count-state"><span class="product-count-dot"></span>PASS</span>
                            </div>
                            <div class="product-count-body">
                              <span class="product-count-value">{normal_file_count:,}</span><span class="product-count-unit">개</span>
                            </div>
                            <div class="product-count-meta">파일 단위 누적 판정 · 총 {total_verdict_count:,}개</div>
                          </div>
                          <div class="product-count-card anomaly">
                            <div class="product-count-head">
                              <span>불량 제품</span>
                              <span class="product-count-state"><span class="product-count-dot"></span>REVIEW</span>
                            </div>
                            <div class="product-count-body">
                              <span class="product-count-value">{anomaly_file_count:,}</span><span class="product-count-unit">개</span>
                            </div>
                            <div class="product-count-meta">격리·확인 대상 · 총 {total_verdict_count:,}개</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                workspace_key = f"battery-pack-kpi-workspace::{Path(selected_record['file_name']).stem}"
                render_kpi_workspace(
                    draggable_kpi_workspace_html(
                        window_df,
                        window_kpis,
                        storage_key=workspace_key,
                        max_points=int(live_window_size),
                        anomaly_mask=window_anomaly_mask,
                    ),
                    key=workspace_key,
                )

                st.markdown("#### 품질 이벤트")
                if bad_window_count > 0:
                    serial_number = live_serial_number(live_df, selected_record["file_name"])
                    st.error(f"시리얼넘버 {serial_number} · 불량 발생 {bad_window_count:,}건")
                elif seen_events.empty:
                    st.success("현재 위치까지 데이터 품질 이벤트가 없습니다.")
                if not seen_events.empty:
                    st.dataframe(seen_events.head(12), width="stretch", hide_index=True, height=280)
                    st.caption("위 이벤트는 모델 예측이 아니라 결측·시간 간격·물리 범위 점검 결과입니다.")

                voltage_col, temp_col = st.columns(2)
                with voltage_col:
                    st.plotly_chart(
                        sensor_envelope_figure(
                            window_df,
                            "voltage",
                            max_points=int(live_window_size),
                            anomaly_mask=window_anomaly_mask,
                        ),
                        width="stretch",
                        key=f"live_voltage_envelope_{live_signature}",
                    )
                with temp_col:
                    st.plotly_chart(
                        sensor_envelope_figure(
                            window_df,
                            "temperature",
                            max_points=int(live_window_size),
                            anomaly_mask=window_anomaly_mask,
                        ),
                        width="stretch",
                        key=f"live_temp_envelope_{live_signature}",
                    )

                st.markdown("#### 최근 측정 로그")
                recent_start = max(0, position - int(live_log_rows))
                recent_positions = np.arange(recent_start, position, dtype=int)[::-1]
                recent_log = measurement_kpi_log(
                    live_df,
                    recent_positions,
                    source_file=selected_record["file_name"],
                )
                recent_kpis = build_sensor_kpis(live_df.iloc[recent_positions]).reset_index(drop=True)
                st.dataframe(
                    kpi_log_styler(
                        recent_log,
                        recent_kpis,
                        recent_positions,
                        visible_fault_domains,
                        normal_reference,
                        str(selected_record["mode"]),
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if position >= total_rows:
                    st.success("선택 파일 재생이 완료되었습니다.")

            render_live_panel()
        except Exception as exc:
            st.error(f"실시간 모니터링 준비 실패: {type(exc).__name__}: {exc}")


with tab_live_visual:
    st.markdown('<div class="section-label">실시간 배터리팩 센서 맵</div>', unsafe_allow_html=True)
    if not tab_live_visual.open:
        pass
    elif selected_record is None:
        st.info("실시간 검사에 사용할 Train 또는 Test 파일을 선택하세요.")
    elif active_model is None:
        st.warning("운영 승인된 배터리팩 LSTM 모델이 없습니다.")
    else:
        try:
            schematic_df = ops_live_df
            schematic_signature = (
                f"{selected_record['path']}::{selected_record['modified_at']}::"
                f"{active_model.model_id}::{active_model.sha256}"
            )
            if st.session_state.get("schematic_signature") != schematic_signature:
                st.session_state["schematic_signature"] = schematic_signature

            model_live_key = f"{selected_record['path']}::{selected_record['modified_at']}::{active_model.model_id}::{active_model.sha256}"
            cached_schematic_model = st.session_state.get("live_model_analysis")
            if not cached_schematic_model or cached_schematic_model.get("key") != model_live_key:
                with st.spinner("배터리팩 LSTM 모델을 준비하고 있습니다."):
                    cached_schematic_model = {
                        "key": model_live_key,
                        "result": load_model_score(active_model, selected_record),
                    }
                    st.session_state["live_model_analysis"] = cached_schematic_model
            schematic_result = cached_schematic_model["result"]
            schematic_total_rows = len(schematic_df)

            with st.container(horizontal=True, gap="small"):
                schematic_start = st.button(
                    "가동",
                    icon=":material/play_arrow:",
                    type="primary",
                    key="schematic_start",
                )
                schematic_stop = st.button(
                    "정지",
                    icon=":material/stop:",
                    key="schematic_stop",
                )
                schematic_step = st.button(
                    f"{int(live_step_size)}행 진행",
                    icon=":material/skip_next:",
                    key="schematic_step",
                )
                schematic_reset = st.button(
                    "초기화",
                    icon=":material/restart_alt:",
                    key="schematic_reset",
                )

            if schematic_reset:
                st.session_state["ops_live_position"] = 1
                st.session_state["ops_live_running"] = False
                st.session_state["ops_live_next_tick"] = 0.0
                begin_live_fault_run()
            elif schematic_stop:
                st.session_state["ops_live_running"] = False
                st.session_state["ops_live_next_tick"] = 0.0
            elif schematic_step:
                st.session_state["ops_live_running"] = False
                st.session_state["ops_live_next_tick"] = 0.0
                st.session_state["ops_live_position"] = min(
                    schematic_total_rows,
                    int(st.session_state.get("ops_live_position", 1))
                    + int(live_step_size),
                )
            elif schematic_start:
                if int(st.session_state.get("ops_live_position", 1)) >= schematic_total_rows:
                    st.session_state["ops_live_position"] = 1
                    begin_live_fault_run()
                st.session_state["ops_live_running"] = True
                st.session_state["ops_live_next_tick"] = time.monotonic() + float(live_refresh_seconds)
                # Rebuild the global heartbeat once with run_every enabled.
                st.rerun()

            schematic_run_every = (
                float(live_refresh_seconds)
                if st.session_state.get("ops_live_running", False)
                else None
            )

            @st.fragment(run_every=schematic_run_every)
            def render_pack_schematic() -> None:
                row_result = schematic_result.get("row_result", pd.DataFrame())
                position = max(
                    1,
                    min(
                        int(st.session_state.get("ops_live_position", 1)),
                        schematic_total_rows,
                    ),
                )
                persist_completed_live_fault_event(
                    schematic_result,
                    record=selected_record,
                    frame=schematic_df,
                    position=position,
                    total_rows=schematic_total_rows,
                )
                current_flag, _, latest_scored_row = latest_scored_prediction(
                    row_result,
                    position,
                )

                phase = (
                    "monitoring"
                    if st.session_state.get("ops_live_running", False)
                    else "idle"
                )
                fault_type = ""
                confidence = float("nan")
                suspect_sensors: list[str] = []
                event_row: int | None = None

                if current_flag:
                    details = schematic_result.get("details", {})
                    payload_row, payload = latest_fault_payload(
                        schematic_result,
                        latest_scored_row or position,
                    )
                    payload = payload if isinstance(payload, dict) else {}
                    fault_type = str(payload.get("fault_type", details.get("fault_type", "불량 유형 확인 필요")))
                    confidence = pd.to_numeric(
                        pd.Series([payload.get("fault_confidence", details.get("fault_confidence", np.nan))]),
                        errors="coerce",
                    ).iloc[0]
                    raw_suspects = payload.get("suspect_sensors", details.get("suspect_sensors", []))
                    if isinstance(raw_suspects, str):
                        suspect_sensors = [item.strip() for item in raw_suspects.split(",") if item.strip()]
                    elif isinstance(raw_suspects, (list, tuple, set)):
                        suspect_sensors = [str(item).strip() for item in raw_suspects if str(item).strip()]
                    phase = "fault"
                    event_row = payload_row or latest_scored_row
                elif position >= schematic_total_rows:
                    st.session_state["ops_live_running"] = False
                    st.session_state["ops_live_next_tick"] = 0.0
                    phase = "complete"

                serial_number = live_serial_number(schematic_df, selected_record["file_name"])
                st.html(
                    battery_pack_schematic_html(
                        position=position,
                        total_rows=schematic_total_rows,
                        phase=phase,
                        serial_number=serial_number,
                        model_name=active_model.name,
                        fault_type=fault_type,
                        confidence=float(confidence) if np.isfinite(confidence) else float("nan"),
                        suspect_sensors=suspect_sensors,
                        event_row=event_row,
                    )
                )

            render_pack_schematic()
        except Exception as exc:
            st.error(f"실시간 센서 맵 준비 실패: {type(exc).__name__}: {exc}")


with tab_data_status:
    readable_count = int(catalog["readable"].sum()) if not catalog.empty else 0
    chg_count = int(catalog["mode"].eq("CHG").sum()) if not catalog.empty else 0
    dchg_count = int(catalog["mode"].eq("DCHG").sum()) if not catalog.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("저장된 데이터", f"{len(catalog):,}개", delta=f"읽기 가능 {readable_count:,}개")
    c2.metric("충전", f"{chg_count:,}개")
    c3.metric("방전", f"{dchg_count:,}개")
    c4.metric(
        "운영 모델",
        f"{len(operational_models):,}개",
        delta=f"검증 후보 {len(candidate_models):,}개",
    )

    st.markdown('<div class="section-label">데이터 자산 현황</div>', unsafe_allow_html=True)
    if catalog.empty:
        st.info("활성 데이터 소스에서 CSV를 찾지 못했습니다.")
    else:
        asset_catalog = catalog.copy()
        asset_catalog["충·방전"] = asset_catalog["mode"].map({"CHG": "충전", "DCHG": "방전"}).fillna("미분류")
        asset_catalog["OK·NG"] = np.where(
            asset_catalog["source_role"].eq("train"),
            "OK",
            asset_catalog["label_hint"].where(asset_catalog["label_hint"].isin(["OK", "NG"]), "미분류"),
        )
        summary = (
            asset_catalog.groupby(["충·방전", "OK·NG"], as_index=False, sort=False)
            .agg(
                파일_수=("file_name", "count"),
                전압_센서_수=("cv_sensors", "max"),
                온도_센서_수=("temp_sensors", "max"),
            )
            .rename(
                columns={
                    "파일_수": "파일 수",
                    "전압_센서_수": "전압 센서 수",
                    "온도_센서_수": "온도 센서 수",
                }
            )
        )
        summary["_mode_order"] = summary["충·방전"].map({"충전": 0, "방전": 1, "미분류": 2})
        summary["_label_order"] = summary["OK·NG"].map({"OK": 0, "NG": 1, "미분류": 2})
        summary = summary.sort_values(["_mode_order", "_label_order"]).drop(columns=["_mode_order", "_label_order"])
        st.dataframe(summary, width="stretch", hide_index=True)

        recent = catalog.sort_values("modified_at", ascending=False).head(12)
        st.markdown('<div class="section-label">최근 저장 파일</div>', unsafe_allow_html=True)
        st.dataframe(translated_catalog(recent), width="stretch", hide_index=True)

        unreadable = catalog[~catalog["readable"]]
        if not unreadable.empty:
            st.error(f"읽을 수 없는 CSV가 {len(unreadable)}개 있습니다.")
            st.dataframe(translated_catalog(unreadable), width="stretch", hide_index=True)

    reviews = load_reviews()
    if not reviews.empty:
        st.markdown('<div class="section-label">최근 현장 검토</div>', unsafe_allow_html=True)
        st.dataframe(
            translated_reviews(reviews.tail(8).iloc[::-1]),
            width="stretch",
            hide_index=True,
        )


with tab_daily:
    st.markdown('<div class="section-label">일별·파일 측정 로그</div>', unsafe_allow_html=True)
    if not tab_daily.open:
        pass
    elif selected_record is not None:
        try:
            log_df = ops_live_df
            daily_model_result = None
            if active_model is not None:
                daily_signature = f"{selected_record['path']}::{selected_record['modified_at']}"
                daily_model_key = f"{daily_signature}::{active_model.model_id}::{active_model.sha256}"
                cached_daily_model = st.session_state.get("live_model_analysis")
                if not cached_daily_model or cached_daily_model.get("key") != daily_model_key:
                    cached_daily_model = {
                        "key": daily_model_key,
                        "result": load_model_score(active_model, selected_record),
                    }
                    st.session_state["live_model_analysis"] = cached_daily_model
                daily_model_result = cached_daily_model["result"]
            daily_fault_domains = fault_domain_coverage(daily_model_result, len(log_df))
            evaluated_positions = evaluated_row_positions(
                daily_model_result.get("row_result")
                if isinstance(daily_model_result, dict)
                else None,
                len(log_df),
            )

            _, _, log_timestamp = resolve_time_axis(log_df)
            if (
                log_timestamp is not None
                and log_timestamp.notna().any()
                and evaluated_positions.size
            ):
                evaluated_timestamps = log_timestamp.iloc[evaluated_positions]
                available_dates = sorted(evaluated_timestamps.dropna().dt.date.unique())
            else:
                available_dates = []

            daily_date_col, daily_serial_col = st.columns(2)
            if available_dates:
                with daily_date_col:
                    selected_date = st.selectbox(
                        "조회 날짜",
                        options=available_dates,
                        index=len(available_dates) - 1,
                        key=f"daily_date_{selected_record['file_name']}",
                    )
                date_matches = log_timestamp.iloc[evaluated_positions].dt.date.eq(selected_date)
                date_source_positions = evaluated_positions[date_matches.to_numpy()]
                date_text = str(selected_date)
            else:
                date_source_positions = evaluated_positions
                date_text = "판정 행 없음" if not len(date_source_positions) else "시간 정보 없음"
                with daily_date_col:
                    st.selectbox(
                        "조회 날짜",
                        options=[date_text],
                        disabled=True,
                        key=f"daily_date_unavailable_{selected_record['file_name']}",
                    )

            daily_serial_series = serial_number_labels(log_df, selected_record["file_name"])
            serial_options = sorted_serial_numbers(
                daily_serial_series.iloc[date_source_positions].tolist()
                if len(date_source_positions)
                else []
            )
            with daily_serial_col:
                if serial_options:
                    selected_daily_serial = st.selectbox(
                        "Serial Num",
                        options=serial_options,
                        key=f"daily_serial_{selected_record['file_name']}_{date_text}",
                    )
                else:
                    st.selectbox(
                        "Serial Num",
                        options=["판정 행 없음"],
                        disabled=True,
                        key=f"daily_serial_unavailable_{selected_record['file_name']}_{date_text}",
                    )
                    selected_daily_serial = None

            if selected_daily_serial is None:
                day_source_positions = np.asarray([], dtype=int)
            else:
                serial_matches = daily_serial_series.iloc[date_source_positions].eq(
                    selected_daily_serial
                )
                day_source_positions = date_source_positions[serial_matches.to_numpy()]

            day_df = log_df.iloc[day_source_positions].copy()
            if (
                log_timestamp is not None
                and log_timestamp.notna().any()
                and len(day_source_positions)
            ):
                day_times = log_timestamp.iloc[day_source_positions].dropna()
                duration_seconds = (
                    (day_times.max() - day_times.min()).total_seconds()
                    if len(day_times) >= 2
                    else 0.0
                )
            else:
                duration_seconds = np.nan

            d1, d2, d3 = st.columns(3)
            d1.metric("조회 기준", date_text)
            d2.metric("모델 판정 행", f"{len(day_df):,}행")
            d3.metric("측정 구간", f"{duration_seconds / 60:.1f}분" if np.isfinite(duration_seconds) else "-")
            if not len(day_source_positions):
                st.info("현재 파일에서 모델 점수가 계산된 행이 없습니다.")

            daily_export = measurement_kpi_log(
                log_df,
                day_source_positions,
                source_file=selected_record["file_name"],
            )
            display_positions = day_source_positions[-500:][::-1]
            daily_view = measurement_kpi_log(
                log_df,
                display_positions,
                source_file=selected_record["file_name"],
            )
            daily_kpis = build_sensor_kpis(log_df.iloc[display_positions]).reset_index(drop=True)
            log_selection = st.dataframe(
                kpi_log_styler(
                    daily_view,
                    daily_kpis,
                    display_positions,
                    daily_fault_domains,
                    normal_reference,
                    str(selected_record["mode"]),
                ),
                width="stretch",
                hide_index=True,
                height=360,
                on_select="rerun",
                selection_mode="single-row",
                key=(
                    f"daily_log_table_{Path(selected_record['file_name']).stem}_"
                    f"{date_text}_{selected_daily_serial or 'none'}"
                ),
            )
            st.download_button(
                "조회 로그 CSV",
                data=dataframe_csv_bytes(daily_export),
                file_name=(
                    f"{Path(selected_record['file_name']).stem}_{date_text}_"
                    f"{selected_daily_serial or 'none'}_log.csv"
                ),
                mime="text/csv",
                key="daily_log_download",
            )
            st.caption(
                "모델이 실제로 점수와 정상·불량 판정을 생성한 행만 표시합니다. "
                "화면은 최근 500개 판정 행을 보여주며, 행을 클릭하면 아래에 "
                "176개 전압·32개 온도 센서값이 표시됩니다."
            )

            selection = getattr(log_selection, "selection", None)
            selected_rows = list(getattr(selection, "rows", [])) if selection is not None else []
            if selected_rows:
                selected_display_position = int(selected_rows[0])
                selected_source_position = int(display_positions[selected_display_position])
                selected_row = log_df.iloc[selected_source_position]
                selected_time = measurement_time_labels(log_df).iloc[selected_source_position]
                selected_serial = serial_number_labels(log_df, selected_record["file_name"]).iloc[selected_source_position]
                cv_cols, temp_cols = detect_sensor_columns(log_df.columns)
                voltage_matrix = sensor_snapshot_matrix(selected_row, cv_cols, "voltage")
                temperature_matrix = sensor_snapshot_matrix(selected_row, temp_cols, "temperature")

                st.markdown("#### 선택 행 센서 상세")
                st.caption(f"측정 시각 {selected_time} · Serial Num {selected_serial}")
                voltage_detail_col, temp_detail_col = st.columns([2.2, 1.4])
                with voltage_detail_col:
                    st.markdown(f"##### 셀 전압 센서 {len(cv_cols):,}개")
                    if voltage_matrix.empty:
                        st.info("표시할 셀 전압 센서값이 없습니다.")
                    else:
                        st.dataframe(
                            sensor_matrix_styler(
                                voltage_matrix,
                                "voltage",
                                normal_reference,
                                str(selected_record["mode"]),
                            ),
                            width="stretch",
                            height=610,
                        )
                with temp_detail_col:
                    st.markdown(f"##### 온도 센서 {len(temp_cols):,}개")
                    if temperature_matrix.empty:
                        st.info("표시할 온도 센서값이 없습니다.")
                    else:
                        st.dataframe(
                            sensor_matrix_styler(
                                temperature_matrix,
                                "temperature",
                                normal_reference,
                                str(selected_record["mode"]),
                            ),
                            width="stretch",
                            height=610,
                        )
            else:
                st.info("센서 원시값을 확인할 로그 행을 선택하세요.")
        except Exception as exc:
            st.error(f"일별 로그 준비 실패: {type(exc).__name__}: {exc}")
    else:
        st.info("로그를 확인할 파일을 선택하세요.")

    st.divider()
    if selected_record is None:
        st.info("분석할 파일을 선택하세요.")
    elif active_model is None:
        quality_key = f"{selected_record['path']}::{selected_record['modified_at']}"
        if st.button("선택 파일 데이터 품질 점검", type="primary", width="stretch"):
            with st.spinner("입력 스키마와 데이터 품질을 점검하고 있습니다."):
                started = time.perf_counter()
                try:
                    quality_frame = load_path(str(selected_record["path"]))
                    quality_checks, quality_summary = audit_data_quality(
                        quality_frame,
                        settings["quality_policy"],
                    )
                    st.session_state["quality_analysis"] = {
                        "analysis_key": quality_key,
                        "input_df": quality_frame,
                        "quality_checks": quality_checks,
                        "quality_summary": quality_summary,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                    }
                except Exception as exc:
                    st.session_state.pop("quality_analysis", None)
                    st.error(f"품질 점검 실패: {type(exc).__name__}: {exc}")

        quality_result = st.session_state.get("quality_analysis")
        if quality_result and quality_result.get("analysis_key") == quality_key:
            quality = quality_result["quality_summary"]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("데이터 품질", quality["status"])
            q2.metric("미통과 항목", f"{quality['failed_checks']:,}개")
            q3.metric("측정 행", f"{len(quality_result['input_df']):,}행")
            q4.metric("점검 시간", f"{quality_result['elapsed_ms']:.0f} ms")
            voltage_col, temp_col = st.columns(2)
            with voltage_col:
                st.plotly_chart(
                    sensor_envelope_figure(
                        quality_result["input_df"],
                        "voltage",
                        settings["display"]["max_plot_rows"],
                    ),
                    width="stretch",
                    key="quality_voltage_envelope",
                )
            with temp_col:
                st.plotly_chart(
                    sensor_envelope_figure(
                        quality_result["input_df"],
                        "temperature",
                        settings["display"]["max_plot_rows"],
                    ),
                    width="stretch",
                    key="quality_temp_envelope",
                )
            st.dataframe(quality_result["quality_checks"], width="stretch", hide_index=True)
        else:
            st.info("품질 점검을 실행하면 입력 데이터 상태와 센서 범위를 표시합니다.")
    else:
        run_analysis = st.button("선택 파일 판정 실행", type="primary", width="stretch")
        analysis_key = f"{selected_record['path']}::{active_model.model_id}::{active_model.sha256}"
        if run_analysis:
            with st.spinner("센서 구조 점검과 모델 판정을 실행하고 있습니다."):
                started = time.perf_counter()
                try:
                    frame = load_path(str(selected_record["path"]))
                    checks, quality_summary = audit_data_quality(frame, settings["quality_policy"])
                    result = score_dataframe(active_model, frame, selected_record["file_name"])
                    result["quality_checks"] = checks
                    result["quality_summary"] = quality_summary
                    result["input_df"] = frame
                    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
                    result["analysis_key"] = analysis_key
                    st.session_state["active_analysis"] = result
                    event = persist_file_fault_event(
                        result,
                        record=selected_record,
                        frame=frame,
                        spec=active_model,
                    )
                except Exception as exc:
                    st.session_state.pop("active_analysis", None)
                    st.error(f"판정 실패: {type(exc).__name__}: {exc}")

        result = st.session_state.get("active_analysis")
        if result and result.get("analysis_key") == analysis_key:
            summary = result["summary"]
            quality = result["quality_summary"]
            status_label = "불량 검토" if summary["status"] == "NG_REVIEW" else "정상"
            compatibility_badge = status_html("INFO", f"모델 호환 {summary['compatibility']:.0%}")
            st.markdown(
                f"<div class='status-line'>{status_html(summary['status'], status_label)}{status_html(quality['status'], '데이터 품질 ' + quality['status'])}{compatibility_badge}</div>",
                unsafe_allow_html=True,
            )
            if summary["compatibility"] < 0.8:
                st.warning(
                    f"학습 특징 중 {summary['missing_feature_count']}개가 입력에 없어 대체값으로 처리되었습니다. "
                    "이 판정은 참고용이며, 동일 스키마 모델 또는 사용자 정의 어댑터가 필요합니다."
                )
            m1, m2, m3 = st.columns(3)
            m1.metric("최종 판정", status_label)
            m2.metric("이상 행 비율", f"{summary['fire_rate']:.1%}", delta=f"기준 {summary['fire_rate_threshold']:.0%}")
            m3.metric("점수 P95", fmt_float(summary["score_p95"], 4), delta=f"행 기준 {summary['row_threshold']:.4g}")
            m4, m5, m6 = st.columns(3)
            m4.metric("최대 연속 이상", f"{summary['max_consecutive_rows']:,}행")
            m5.metric("데이터 품질", quality["status"], delta=f"미통과 {quality['failed_checks']}개")
            m6.metric("처리 시간", f"{result['elapsed_ms']:.0f} ms")
            st.markdown(
                f"<div class='model-note'><b>판정 근거</b> · {summary['trigger']} · 모델 {summary['model_name']} {summary['model_version']} · 파일 {summary['source_file']}</div>",
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.15, 1])
            with left:
                st.plotly_chart(
                    score_timeline_figure(result["row_result"], summary["row_threshold"]),
                    width="stretch",
                    key="analysis_score_timeline",
                )
            with right:
                st.plotly_chart(feature_timeline_figure(result["features"]), width="stretch", key="analysis_feature_timeline")

            voltage_col, temp_col = st.columns(2)
            with voltage_col:
                st.plotly_chart(
                    sensor_envelope_figure(result["input_df"], "voltage", settings["display"]["max_plot_rows"]),
                    width="stretch",
                    key="analysis_voltage_envelope",
                )
            with temp_col:
                st.plotly_chart(
                    sensor_envelope_figure(result["input_df"], "temperature", settings["display"]["max_plot_rows"]),
                    width="stretch",
                    key="analysis_temp_envelope",
                )

            evidence = result["row_result"].nlargest(30, "score")
            evidence = pd.concat(
                [
                    evidence.reset_index(drop=True),
                    result["features"].iloc[(evidence["row_index"] - 1).astype(int)].reset_index(drop=True)[
                        [c for c in ["cv_range", "cv_resid_maxabs", "temp_range", "temp_resid_maxabs", "temp_pair_gap_max"] if c in result["features"].columns]
                    ],
                ],
                axis=1,
            )
            st.markdown('<div class="section-label">상위 이상 근거 행</div>', unsafe_allow_html=True)
            st.dataframe(evidence, width="stretch", hide_index=True)
            st.download_button(
                "행 단위 판정 CSV",
                data=dataframe_csv_bytes(result["row_result"]),
                file_name=f"{Path(selected_record['file_name']).stem}_{active_model.model_id}_row_scores.csv",
                mime="text/csv",
            )

            with st.expander("데이터 품질 점검 결과"):
                st.dataframe(result["quality_checks"], width="stretch", hide_index=True)
        else:
            st.info("판정을 실행하면 파일 상태와 근거가 표시됩니다.")


with tab_diagnosis:
    st.markdown('<div class="section-label">셀·모듈 상세 진단</div>', unsafe_allow_html=True)
    if selected_record is None:
        st.info("진단할 파일을 선택하세요.")
    else:
        try:
            diagnosis_df = load_path(str(selected_record["path"]))
            cv_cols, temp_cols = detect_sensor_columns(diagnosis_df.columns)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("셀 전압 센서", f"{len(cv_cols):,}개")
            c2.metric("온도 센서", f"{len(temp_cols):,}개")
            c3.metric("측정 행", f"{len(diagnosis_df):,}")
            c4.metric("스키마", selected_record["schema"])

            heat_voltage, heat_temp = st.columns(2)
            with heat_voltage:
                st.plotly_chart(
                    sensor_heatmap_figure(diagnosis_df, "voltage", settings["display"]["max_heatmap_rows"]),
                    width="stretch",
                    key="diagnosis_voltage_heatmap",
                )
            with heat_temp:
                st.plotly_chart(
                    sensor_heatmap_figure(diagnosis_df, "temperature", settings["display"]["max_heatmap_rows"]),
                    width="stretch",
                    key="diagnosis_temp_heatmap",
                )

            voltage_rank = sensor_deviation_ranking(diagnosis_df, "voltage")
            temp_rank = sensor_deviation_ranking(diagnosis_df, "temperature")
            rank_voltage, rank_temp = st.columns(2)
            with rank_voltage:
                st.plotly_chart(ranking_bar_figure(voltage_rank, "voltage"), width="stretch", key="diagnosis_voltage_rank")
                st.dataframe(voltage_rank.head(20), width="stretch", hide_index=True)
            with rank_temp:
                st.plotly_chart(ranking_bar_figure(temp_rank, "temperature"), width="stretch", key="diagnosis_temp_rank")
                st.dataframe(temp_rank.head(20), width="stretch", hide_index=True)

            module_summary = module_temperature_summary(diagnosis_df)
            if not module_summary.empty:
                st.markdown('<div class="section-label">모듈 T01-T02 편차</div>', unsafe_allow_html=True)
                st.dataframe(module_summary, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"상세 진단 실패: {type(exc).__name__}: {exc}")


with tab_data_status:
    st.markdown('<div class="section-label">다중 파일 배치 판정</div>', unsafe_allow_html=True)
    if catalog.empty or active_model is None:
        st.info("운영 승인 모델이 등록되면 여러 파일을 같은 기준으로 일괄 판정할 수 있습니다.")
    else:
        f1, f2, f3 = st.columns([1, 1, 1])
        with f1:
            batch_roles = st.multiselect(
                "데이터 역할",
                options=sorted(catalog["source_role"].unique()),
                default=[role for role in ["test", "external", "inbox"] if role in set(catalog["source_role"])],
            )
        with f2:
            mode_pairs = {
                str(mode): display_mode(mode)
                for mode in sorted(catalog["mode"].dropna().astype(str).unique())
            }
            batch_mode_labels = st.multiselect(
                "충·방전",
                options=list(mode_pairs.values()),
                default=[
                    label
                    for code, label in mode_pairs.items()
                    if code in {"CHG", "DCHG"}
                ],
            )
            batch_modes = [code for code, label in mode_pairs.items() if label in batch_mode_labels]
        with f3:
            batch_limit = st.number_input("최대 파일 수", min_value=1, max_value=500, value=int(settings["display"]["default_batch_limit"]), step=10)

        candidates = catalog[
            catalog["readable"]
            & catalog["source_role"].isin(batch_roles or catalog["source_role"].unique())
            & catalog["mode"].isin(batch_modes or catalog["mode"].unique())
        ].head(int(batch_limit))
        st.caption(f"판정 대상 {len(candidates):,}개")
        if st.button("배치 판정 실행", type="primary"):
            progress = st.progress(0.0, text="배치 판정 준비")
            batch_rows: list[dict[str, object]] = []
            started = time.perf_counter()
            for position, (_, row) in enumerate(candidates.iterrows(), start=1):
                progress.progress(position / max(1, len(candidates)), text=f"{position}/{len(candidates)} · {row['file_name']}")
                try:
                    frame = load_path(str(row["path"]))
                    _, quality_summary = audit_data_quality(frame, settings["quality_policy"])
                    model_result = score_dataframe(active_model, frame, row["file_name"])
                    prediction = model_result["summary"]
                    fault_metadata = extract_fault_metadata(model_result)
                    batch_rows.append(
                        {
                            "source_name": row["source_name"],
                            "source_role": row["source_role"],
                            "file_name": row["file_name"],
                            "path": row["path"],
                            "label_hint": row["label_hint"],
                            "mode": row["mode"],
                            "data_quality": quality_summary["status"],
                            **prediction,
                            **fault_metadata,
                        }
                    )
                except Exception as exc:
                    batch_rows.append(
                        {
                            "source_name": row["source_name"],
                            "source_role": row["source_role"],
                            "file_name": row["file_name"],
                            "path": row["path"],
                            "label_hint": row["label_hint"],
                            "mode": row["mode"],
                            "status": "ERROR",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            progress.empty()
            batch_result = pd.DataFrame(batch_rows)
            csv_path, _ = save_batch_result(
                batch_result,
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "model_id": active_model.model_id,
                    "model_version": active_model.version,
                    "file_count": len(batch_result),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            st.session_state["batch_result"] = batch_result
            st.session_state["batch_path"] = str(csv_path)

        batch_result = st.session_state.get("batch_result")
        if isinstance(batch_result, pd.DataFrame) and not batch_result.empty:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("완료 파일", f"{len(batch_result):,}")
            b2.metric("정상", f"{batch_result['status'].eq('NORMAL').sum():,}")
            b3.metric("불량 검토", f"{batch_result['status'].eq('NG_REVIEW').sum():,}")
            b4.metric("오류", f"{batch_result['status'].eq('ERROR').sum():,}")
            chart_col, table_col = st.columns([0.7, 1.8])
            with chart_col:
                st.plotly_chart(batch_status_figure(batch_result), width="stretch", key="batch_status_chart")
            with table_col:
                st.dataframe(batch_result, width="stretch", hide_index=True)
            st.download_button(
                "배치 결과 CSV",
                data=dataframe_csv_bytes(batch_result),
                file_name=Path(st.session_state.get("batch_path", "batch_result.csv")).name,
                mime="text/csv",
            )


with tab_review:
    st.markdown('<div class="section-label">현장 검토·재라벨링</div>', unsafe_allow_html=True)
    deleted_review_message = st.session_state.pop("deleted_review_message", "")
    if deleted_review_message:
        st.success(deleted_review_message)
    saved_review_message = st.session_state.pop("saved_review_message", "")
    if saved_review_message:
        st.success(saved_review_message)

    review_fault_events = representative_fault_events(
        load_fault_events(),
        model_id=active_model.model_id if active_model is not None else None,
    )
    if "serial_number" in review_fault_events.columns:
        review_serial_options = sorted_serial_numbers(
            review_fault_events["serial_number"].tolist()
        )
    else:
        review_serial_options = []

    if not review_serial_options:
        st.info("현장 검토를 기록할 불량 로그의 Serial Num이 없습니다.")
    else:
        with st.form("operator_review_form", clear_on_submit=True):
            r1, r2, r3 = st.columns(3)
            with r1:
                reviewer = st.text_input("검토자")
                human_label = st.selectbox("현장 판정", ["NORMAL", "NG", "REVIEW"])
            with r2:
                fault_type = st.selectbox(
                    "불량 유형",
                    ["해당 없음", "저용량 불량", "고저항 불량", "용량 불량", "복합 불량", "용접·접촉 불량", "센싱와이어 불량", "온도 센서 불량", "전압 센서 불량", "열 관리 이상", "기타"],
                )
                selected_review_serial = st.selectbox(
                    "Serial Num",
                    options=review_serial_options,
                    index=None,
                    placeholder="불량 로그의 Serial Num 선택",
                )
            with r3:
                severity = st.select_slider("조치 우선순위", options=["낮음", "보통", "높음", "긴급"], value="보통")
                notes = st.text_area("검토 메모", height=96)
            submitted = st.form_submit_button("검토 기록 저장", type="primary")
            if submitted:
                if not reviewer.strip():
                    st.error("검토자 이름을 입력하세요.")
                elif selected_review_serial is None:
                    st.error("검토할 불량 로그의 Serial Num을 선택하세요.")
                else:
                    serial_matches = review_fault_events[
                        review_fault_events["serial_number"]
                        .fillna("")
                        .astype(str)
                        .eq(str(selected_review_serial))
                    ]
                    selected_review_fault = serial_matches.iloc[0]
                    review_id = f"review-{time.time_ns()}"
                    path = append_review(
                        {
                            "review_id": review_id,
                            "reviewer": reviewer.strip(),
                            "source_file": selected_review_fault.get("source_file", ""),
                            "source_path": selected_review_fault.get("source_path", ""),
                            "serial_number": str(selected_review_serial),
                            "model_id": selected_review_fault.get("model_id", ""),
                            "model_version": selected_review_fault.get("model_version", ""),
                            "model_status": selected_review_fault.get("model_status", ""),
                            "human_label": human_label,
                            "fault_type": fault_type,
                            "severity": severity,
                            "fire_rate": selected_review_fault.get("fire_rate", np.nan),
                            "score_p95": selected_review_fault.get("score_p95", np.nan),
                            "max_consecutive_rows": selected_review_fault.get(
                                "max_consecutive_rows",
                                np.nan,
                            ),
                            "notes": notes,
                        }
                    )
                    sync_result = storage_api.apply_human_review_to_fault_events(
                        str(selected_review_serial),
                        human_label,
                        reviewer=reviewer.strip(),
                        notes=notes,
                        review_id=review_id,
                    )
                    if human_label == "NORMAL":
                        sync_text = (
                            f"동일 Serial Num의 불량 로그 "
                            f"{sync_result['deleted']}건을 삭제했습니다."
                        )
                    elif human_label == "NG":
                        sync_text = (
                            f"동일 Serial Num의 불량 로그 "
                            f"{sync_result['updated']}건을 조치 대기로 변경했습니다."
                        )
                    else:
                        sync_text = (
                            f"동일 Serial Num의 불량 로그 "
                            f"{sync_result['updated']}건을 검토 중으로 변경했습니다."
                        )
                    st.session_state["saved_review_message"] = (
                        f"검토 기록을 저장했습니다: {path.name} · {sync_text}"
                    )
                    st.rerun()

    reviews = load_reviews()
    if reviews.empty:
        st.caption("저장된 검토 기록이 없습니다.")
    else:
        v1, v2, v3 = st.columns(3)
        v1.metric("누적 검토", f"{len(reviews):,}건")
        v2.metric("현장 NG", f"{reviews['human_label'].eq('NG').sum():,}건")
        disagreement = reviews[
            reviews["human_label"].isin(["NORMAL", "NG"])
            & reviews["model_status"].isin(["NORMAL", "NG_REVIEW"])
        ].copy()
        if not disagreement.empty:
            disagreement["model_binary"] = np.where(disagreement["model_status"].eq("NG_REVIEW"), "NG", "NORMAL")
            disagreement_count = int((disagreement["model_binary"] != disagreement["human_label"]).sum())
        else:
            disagreement_count = 0
        v3.metric("모델·현장 불일치", f"{disagreement_count:,}건")
        review_table_revision = int(st.session_state.get("review_table_revision", 0))
        review_storage_indices = reviews.index[::-1].tolist()
        review_display = translated_reviews(reviews.loc[review_storage_indices])
        review_selection = st.dataframe(
            review_display,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key=f"review_history_table_{review_table_revision}",
        )
        review_selected_rows_raw = list(
            getattr(getattr(review_selection, "selection", None), "rows", [])
        )
        review_selected_rows = []
        for row_index in review_selected_rows_raw:
            try:
                parsed_index = int(row_index)
            except (TypeError, ValueError):
                continue
            if 0 <= parsed_index < len(review_storage_indices):
                review_selected_rows.append(parsed_index)
        selected_review_indices = [
            int(review_storage_indices[row_index])
            for row_index in review_selected_rows
        ]

        review_download_col, review_confirm_col, review_delete_col = st.columns([1.2, 1.2, 1.5])
        with review_download_col:
            st.download_button(
                "검토 이력 CSV",
                dataframe_csv_bytes(review_display),
                "operator_review_log.csv",
                "text/csv",
                width="stretch",
            )
        with review_confirm_col:
            confirm_review_delete = st.checkbox(
                "삭제 확인",
                value=False,
                disabled=not selected_review_indices,
                key=f"confirm_review_delete_{review_table_revision}",
            )
        with review_delete_col:
            delete_review_requested = st.button(
                f"선택 검토 기록 삭제 ({len(selected_review_indices)}건)",
                disabled=not selected_review_indices or not confirm_review_delete,
                key=f"delete_selected_reviews_{review_table_revision}",
                width="stretch",
            )

        if delete_review_requested:
            delete_result = delete_reviews(selected_review_indices)
            st.session_state["deleted_review_message"] = (
                f"선택한 검토 기록 {delete_result['deleted']}건을 삭제했습니다."
            )
            st.session_state["review_table_revision"] = review_table_revision + 1
            st.rerun()


with tab_fault:
    st.markdown('<div class="section-label">불량 로그 및 조치 관리</div>', unsafe_allow_html=True)
    deleted_fault_message = st.session_state.pop("deleted_fault_message", "")
    if deleted_fault_message:
        st.success(deleted_fault_message)
    current_batch = st.session_state.get("batch_result")
    fault_events = representative_fault_events(
        load_fault_events(current_batch if isinstance(current_batch, pd.DataFrame) else None),
        model_id=active_model.model_id if active_model is not None else None,
    )
    if active_model is not None:
        st.caption(f"{active_model.name} 기준 · 동일 CSV는 유형 신뢰도가 가장 높은 대표 판정 1건만 표시합니다.")

    if fault_events.empty:
        st.info("운영 모델에서 불량으로 판정된 로그가 없습니다.")
        if active_model is None:
            st.markdown(
                f"<div class='status-line'>{status_html('INFO', '이진 판정 모델 미연결')}{status_html('INFO', '유형 분류 모델 미연결')}</div>",
                unsafe_allow_html=True,
            )
    else:
        unresolved = fault_events["fault_type"].eq("유형 분석 대기")
        open_actions = ~fault_events["action_status"].eq("완료")
        fault_events["mode_display"] = fault_events["mode"].map(display_mode)
        risk_levels = pd.to_numeric(fault_events["risk_level"], errors="coerce").fillna(0)
        urgent = risk_levels.ge(2)
        fault_log_revision = int(st.session_state.get("fault_log_table_revision", 0))
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("누적 불량", f"{len(fault_events):,}건")
        f2.metric("유형 분석 대기", f"{unresolved.sum():,}건")
        f3.metric("조치 미완료", f"{open_actions.sum():,}건")
        f4.metric("높음·긴급", f"{urgent.sum():,}건")

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        type_options = ["전체"] + sorted(fault_events["fault_type"].dropna().astype(str).unique().tolist())
        status_options = ["전체"] + sorted(fault_events["action_status"].dropna().astype(str).unique().tolist())
        preferred_mode_order = ["충전", "방전", "충·방전", "미분류"]
        available_modes = set(fault_events["mode_display"].dropna().astype(str))
        mode_options = ["전체"] + [mode for mode in preferred_mode_order if mode in available_modes]
        mode_options += sorted(available_modes - set(mode_options))
        with filter_col1:
            selected_fault_type = st.selectbox(
                "불량 유형",
                type_options,
                key=f"fault_log_type_filter_{fault_log_revision}",
            )
        with filter_col2:
            selected_action_status = st.selectbox(
                "처리 상태",
                status_options,
                key=f"fault_log_status_filter_{fault_log_revision}",
            )
        with filter_col3:
            selected_fault_mode = st.selectbox(
                "충·방전",
                mode_options,
                key=f"fault_log_mode_filter_{fault_log_revision}",
            )

        filtered_faults = fault_events.copy()
        if selected_fault_type != "전체":
            filtered_faults = filtered_faults[filtered_faults["fault_type"].eq(selected_fault_type)]
        if selected_action_status != "전체":
            filtered_faults = filtered_faults[filtered_faults["action_status"].eq(selected_action_status)]
        if selected_fault_mode != "전체":
            filtered_faults = filtered_faults[filtered_faults["mode_display"].eq(selected_fault_mode)]
        filtered_faults = filtered_faults.reset_index(drop=True)

        if filtered_faults.empty:
            st.info("선택한 조건에 해당하는 불량 로그가 없습니다.")
        else:
            fault_table = filtered_faults[
                [
                    "detected_at",
                    "mode_display",
                    "fault_type",
                    "fault_confidence",
                    "risk_level",
                    "rpn",
                    "suspect_sensors",
                    "recommended_action",
                    "action_status",
                    "assignee",
                ]
            ].copy()
            confidence = pd.to_numeric(fault_table["fault_confidence"], errors="coerce")
            fault_table["fault_confidence"] = confidence.map(
                lambda value: f"{value:.1%}" if np.isfinite(value) else "-"
            )
            fault_table["suspect_sensors"] = fault_table["suspect_sensors"].replace("", "-")
            fault_table["risk_level"] = fault_table["risk_level"].map(pfmea_risk_label)
            fault_table["rpn"] = (
                pd.to_numeric(fault_table["rpn"], errors="coerce").fillna(0).astype(int)
            )
            fault_table = fault_table.rename(
                columns={
                    "detected_at": "검출 시각",
                    "mode_display": "충·방전",
                    "fault_type": "불량 유형",
                    "fault_confidence": "유형 신뢰도",
                    "risk_level": "위험도",
                    "rpn": "RPN",
                    "suspect_sensors": "문제 센서",
                    "recommended_action": "권장 조치",
                    "action_status": "처리 상태",
                    "assignee": "담당자",
                }
            )
            fault_selection = st.dataframe(
                pfmea_fault_table_styler(fault_table),
                width="stretch",
                hide_index=True,
                height=330,
                on_select="rerun",
                selection_mode="multi-row",
                key=f"fault_log_table_{fault_log_revision}",
            )

            selection = getattr(fault_selection, "selection", None)
            selected_rows_raw = list(getattr(selection, "rows", [])) if selection is not None else []
            selected_rows = []
            for row_index in selected_rows_raw:
                try:
                    parsed_index = int(row_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= parsed_index < len(filtered_faults):
                    selected_rows.append(parsed_index)
            selected_event_ids = (
                filtered_faults.iloc[selected_rows]["event_id"].astype(str).tolist()
                if selected_rows
                else []
            )
            fault_log_explicitly_selected = bool(selected_rows)

            download_col, confirm_col, delete_col = st.columns([1.2, 1.2, 1.5])
            with download_col:
                st.download_button(
                    "불량 로그 CSV",
                    data=dataframe_csv_bytes(filtered_faults),
                    file_name="model_fault_log.csv",
                    mime="text/csv",
                    key="fault_log_download",
                    width="stretch",
                )
            with confirm_col:
                confirm_fault_delete = st.checkbox(
                    "삭제 확인",
                    value=False,
                    disabled=not selected_event_ids,
                    key=f"confirm_fault_delete_{fault_log_revision}",
                )
            with delete_col:
                delete_fault_requested = st.button(
                    f"선택 로그 삭제 ({len(selected_event_ids)}건)",
                    disabled=not selected_event_ids or not confirm_fault_delete,
                    key=f"delete_selected_fault_logs_{fault_log_revision}",
                    width="stretch",
                )

            if delete_fault_requested:
                delete_result = delete_fault_events(selected_event_ids)
                st.session_state["deleted_fault_message"] = (
                    f"선택한 불량 로그 {delete_result['requested']}건을 삭제했습니다."
                )
                st.session_state["fault_log_table_revision"] = (
                    st.session_state.get("fault_log_table_revision", 0) + 1
                )
                st.rerun()

            selected_fault_index = int(selected_rows[0]) if selected_rows else 0
            selected_fault = filtered_faults.iloc[selected_fault_index]
            event_id = str(selected_fault["event_id"])

            st.divider()
            st.markdown(f"#### {selected_fault['source_file']} 상세 판정")
            st.markdown(
                f"<div class='status-line'>{status_html(selected_fault['model_status'], '불량 검토')}{status_html('REVIEW', selected_fault['fault_type'])}{status_html('INFO', selected_fault['action_status'])}</div>",
                unsafe_allow_html=True,
            )

            confidence_value = pd.to_numeric(pd.Series([selected_fault["fault_confidence"]]), errors="coerce").iloc[0]
            confidence_text = f"{confidence_value:.1%}" if np.isfinite(confidence_value) else "-"
            selected_risk_level = pd.to_numeric(
                pd.Series([selected_fault["risk_level"]]), errors="coerce"
            ).fillna(0).astype(int).iloc[0]
            selected_rpn = pd.to_numeric(
                pd.Series([selected_fault["rpn"]]), errors="coerce"
            ).fillna(0).astype(int).iloc[0]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("불량 유형", str(selected_fault["fault_type"]))
            d2.metric("유형 신뢰도", confidence_text)
            d3.metric("PFMEA 위험도", f"위험도 {selected_risk_level}", delta=f"RPN {selected_rpn}", delta_color="off")
            d4.metric("검출 시작 행", fmt_float(selected_fault["detected_row"], 0))

            evidence_col, probability_col = st.columns(2)
            with evidence_col:
                st.markdown("#### 이진 불량 판정 근거")
                fire_rate = pd.to_numeric(pd.Series([selected_fault["fire_rate"]]), errors="coerce").iloc[0]
                evidence = pd.DataFrame(
                    {
                        "항목": ["이상 행 비율", "점수 P95", "최대 점수", "최대 연속 이상 행"],
                        "값": [
                            f"{fire_rate:.1%}" if np.isfinite(fire_rate) else "-",
                            fmt_float(selected_fault["score_p95"], 4),
                            fmt_float(selected_fault["score_max"], 4),
                            fmt_float(selected_fault["max_consecutive_rows"], 0),
                        ],
                    }
                )
                st.dataframe(evidence, width="stretch", hide_index=True)
            with probability_col:
                st.markdown("#### 불량 유형별 확률")
                probabilities = parse_probabilities(selected_fault["fault_probabilities"])
                if probabilities:
                    probability_frame = pd.DataFrame(
                        {"불량 유형": list(probabilities.keys()), "확률": list(probabilities.values())}
                    ).sort_values("확률", ascending=True)
                    probability_figure = px.bar(
                        probability_frame,
                        x="확률",
                        y="불량 유형",
                        orientation="h",
                        color_discrete_sequence=[TEAL],
                    )
                    probability_figure.update_xaxes(range=[0, 1], tickformat=".0%")
                    probability_figure.update_layout(
                        height=260,
                        margin=dict(l=8, r=8, t=8, b=8),
                        showlegend=False,
                    )
                    st.plotly_chart(
                        probability_figure,
                        width="stretch",
                        key=f"fault_probability_{event_id}",
                    )
                else:
                    st.info("유형 분류 모델 결과가 아직 연결되지 않았습니다.")

            st.markdown("#### 문제 위치 및 센서 후보")
            model_sensors = str(selected_fault["suspect_sensors"]).strip()
            model_modules = str(selected_fault["suspect_modules"]).strip()
            model_cells = str(selected_fault["suspect_cells"]).strip()
            location_summary = pd.DataFrame(
                {
                    "구분": ["모델 지목 센서", "의심 모듈", "의심 셀"],
                    "결과": [model_sensors or "-", model_modules or "-", model_cells or "-"],
                }
            )
            st.dataframe(location_summary, width="stretch", hide_index=True)

            source_path_text = str(selected_fault["source_path"]).strip()
            source_path = Path(source_path_text) if source_path_text and source_path_text.lower() != "nan" else None
            if source_path is None or not source_path.exists():
                matches = catalog[catalog["file_name"].astype(str).eq(str(selected_fault["source_file"]))]
                if not matches.empty:
                    source_path = Path(str(matches.iloc[0]["path"]))

            if source_path is not None and source_path.exists():
                try:
                    fault_frame = load_path(str(source_path))
                    voltage_rank = sensor_deviation_ranking(fault_frame, "voltage").head(8)
                    temp_rank = sensor_deviation_ranking(fault_frame, "temperature").head(8)
                    voltage_candidate_col, temp_candidate_col = st.columns(2)
                    with voltage_candidate_col:
                        st.markdown("##### 셀 전압 편차 후보")
                        if voltage_rank.empty:
                            st.info("전압 센서 후보가 없습니다.")
                        else:
                            st.plotly_chart(
                                ranking_bar_figure(voltage_rank, "voltage"),
                                width="stretch",
                                key=f"fault_voltage_rank_{event_id}",
                            )
                            st.dataframe(voltage_rank, width="stretch", hide_index=True)
                    with temp_candidate_col:
                        st.markdown("##### 온도 편차 후보")
                        if temp_rank.empty:
                            st.info("온도 센서 후보가 없습니다.")
                        else:
                            st.plotly_chart(
                                ranking_bar_figure(temp_rank, "temperature"),
                                width="stretch",
                                key=f"fault_temp_rank_{event_id}",
                            )
                            st.dataframe(temp_rank, width="stretch", hide_index=True)
                    st.caption("편차 순위는 문제 위치 후보이며, 모델이 지목한 센서 및 현장 계측 결과와 함께 확정합니다.")
                except Exception as exc:
                    st.warning(f"센서 후보 계산 실패: {type(exc).__name__}: {exc}")

            st.markdown("#### 권장 조치")
            action_col, disposition_col = st.columns(2)
            with action_col:
                st.warning(f"1차 조치: {selected_fault['recommended_action']}")
                st.caption(f"PFMEA 연계 유형: {selected_fault['pfmea_ng_codes']}")
            with disposition_col:
                st.error(f"처분 기준: {selected_fault['disposition_guide']}")
                st.caption("폐기 여부는 모델이 자동 확정하지 않으며 현장 재계측과 안전 담당자 승인 후 결정합니다.")

            action_status_options = ["신규", "현장 검토 중", "검토 중", "조치 대기", "완료"]
            final_action_options = [
                "미결정",
                "재실험",
                "센서·하네스 점검",
                "용접부 점검",
                "부품 교체 후 재검사",
                "격리",
                "폐기 검토",
                "정상 복귀",
            ]
            current_status = str(selected_fault["action_status"])
            current_action = str(selected_fault["final_action"])
            st.markdown("#### 처리 상태 및 최종 검토")
            with st.form(f"fault_action_form_{event_id}"):
                a1, a2, a3 = st.columns(3)
                with a1:
                    action_status = st.selectbox(
                        "처리 상태",
                        action_status_options,
                        index=action_status_options.index(current_status) if current_status in action_status_options else 0,
                    )
                with a2:
                    final_action = st.selectbox(
                        "최종 조치",
                        final_action_options,
                        index=final_action_options.index(current_action) if current_action in final_action_options else 0,
                    )
                with a3:
                    assignee = st.text_input("담당자", value=str(selected_fault["assignee"]))
                action_notes = st.text_area("조치 메모", value=str(selected_fault["action_notes"]), height=100)
                action_button_col, action_hint_col = st.columns([1.2, 4])
                with action_button_col:
                    action_submitted = st.form_submit_button(
                        "조치 기록 저장",
                        type="primary",
                        disabled=not fault_log_explicitly_selected,
                        width="stretch",
                    )
                with action_hint_col:
                    if not fault_log_explicitly_selected:
                        st.caption("조치 기록을 저장할 불량 로그를 선택해주세요.")
                if action_submitted:
                    if not assignee.strip():
                        st.error("담당자 이름을 입력하세요.")
                    else:
                        append_fault_action(
                            {
                                "event_id": event_id,
                                "source_file": selected_fault["source_file"],
                                "serial_number": selected_fault.get("serial_number", ""),
                                "fault_type": selected_fault["fault_type"],
                                "action_status": action_status,
                                "final_action": final_action,
                                "assignee": assignee.strip(),
                                "action_notes": action_notes.strip(),
                            }
                        )
                        st.success("조치 기록을 저장했습니다.")
                        st.rerun()

            st.markdown("#### 저장된 조치 기록")
            action_message_key = f"deleted_fault_action_message_{event_id}"
            deleted_action_message = st.session_state.pop(action_message_key, "")
            if deleted_action_message:
                st.success(deleted_action_message)
            fault_actions = load_fault_actions()
            if not fault_actions.empty and "event_id" in fault_actions.columns:
                event_action_history = fault_actions[
                    fault_actions["event_id"].fillna("").astype(str).eq(event_id)
                ].copy()
            else:
                event_action_history = pd.DataFrame()

            action_history_columns = {
                "serial_number": "Serial Num",
                "assignee": "담당자",
                "action_status": "처리 상태",
                "final_action": "최종 조치",
                "action_notes": "조치 메모",
            }
            if event_action_history.empty:
                st.info("저장된 조치 기록이 없습니다.")
            else:
                action_table_revision = int(
                    st.session_state.get("fault_action_table_revision", 0)
                )
                if "updated_at" in event_action_history.columns:
                    event_action_history = event_action_history.sort_values(
                        "updated_at",
                        ascending=False,
                    )
                if "serial_number" not in event_action_history.columns:
                    event_action_history["serial_number"] = selected_fault.get(
                        "serial_number",
                        "",
                    )
                else:
                    event_action_history["serial_number"] = (
                        event_action_history["serial_number"].astype("string").fillna("")
                    )
                    serial_text = (
                        event_action_history["serial_number"]
                        .str.strip()
                    )
                    event_action_history.loc[
                        serial_text.eq(""),
                        "serial_number",
                    ] = selected_fault.get("serial_number", "")
                action_storage_indices = event_action_history.index.tolist()
                for column in action_history_columns:
                    if column not in event_action_history.columns:
                        event_action_history[column] = ""
                action_history_display = (
                    event_action_history[list(action_history_columns)]
                    .fillna("")
                    .rename(columns=action_history_columns)
                    .reset_index(drop=True)
                )
                action_history_selection = st.dataframe(
                    action_history_display,
                    width="stretch",
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    key=f"fault_action_history_{event_id}_{action_table_revision}",
                )
                action_selected_rows_raw = list(
                    getattr(getattr(action_history_selection, "selection", None), "rows", [])
                )
                action_selected_rows = []
                for row_index in action_selected_rows_raw:
                    try:
                        parsed_index = int(row_index)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= parsed_index < len(action_storage_indices):
                        action_selected_rows.append(parsed_index)
                selected_action_indices = [
                    int(action_storage_indices[row_index])
                    for row_index in action_selected_rows
                ]

                action_confirm_col, action_delete_col = st.columns([1.2, 1.5])
                with action_confirm_col:
                    confirm_action_delete = st.checkbox(
                        "조치 기록 삭제 확인",
                        value=False,
                        disabled=not selected_action_indices,
                        key=f"confirm_action_delete_{event_id}_{action_table_revision}",
                    )
                with action_delete_col:
                    delete_action_requested = st.button(
                        f"선택 조치 기록 삭제 ({len(selected_action_indices)}건)",
                        disabled=not selected_action_indices or not confirm_action_delete,
                        key=f"delete_selected_actions_{event_id}_{action_table_revision}",
                        width="stretch",
                    )

                if delete_action_requested:
                    delete_result = delete_fault_actions(selected_action_indices)
                    st.session_state[action_message_key] = (
                        f"선택한 조치 기록 {delete_result['deleted']}건을 삭제했습니다."
                    )
                    st.session_state["fault_action_table_revision"] = (
                        action_table_revision + 1
                    )
                    st.rerun()


with tab_models:
    st.markdown('<div class="section-label">모델 레지스트리</div>', unsafe_allow_html=True)
    if operational_models:
        st.success(
            "운영 승인된 2단계 LSTM이 연결되어 있습니다. 1단계는 정상·불량 판정에 사용하고, "
            "2단계는 1단계에서 불량으로 감지된 로그의 유형과 확률을 제공합니다."
        )
    else:
        st.info(
            "운영 승인 모델이 없습니다. 등록 모델은 후보 상태에서 검증한 뒤 운영 승인 상태로 전환합니다."
        )
    inventory = model_inventory(models)
    if not inventory.empty:
        inventory["상태"] = inventory["model_id"].map(
            {spec.model_id: deployment_state(spec) for spec in models}
        )
    if inventory.empty:
        st.warning("등록된 모델이 없습니다.")
    else:
        st.dataframe(inventory, width="stretch", hide_index=True)
        for spec in models:
            registry_state = deployment_state(spec)
            with st.expander(f"{spec.name} · {spec.version} · {registry_state}"):
                left, right = st.columns([1.3, 1])
                with left:
                    st.write(spec.description)
                    st.caption(spec.validation_scope or "검증 범위 미기록")
                    st.code(str(spec.root_path), language=None)
                with right:
                    metrics = spec.metrics or {}
                    metric_frame = pd.DataFrame(
                        [{"지표": key, "값": value} for key, value in metrics.items()]
                    )
                    if not metric_frame.empty:
                        st.dataframe(metric_frame, width="stretch", hide_index=True)
                    st.write(f"지원 모드: {', '.join(spec.supported_modes)}")
                    st.write(f"SHA256: `{spec.sha256[:16] or '-'}`")

    st.markdown('<div class="section-label">신규 모델 패키지 등록</div>', unsafe_allow_html=True)
    model_package = st.file_uploader("모델 ZIP", type=["zip"], key="model_package_upload")
    trusted = st.checkbox("조직 내부에서 생성한 신뢰 가능한 모델 파일입니다.")
    if st.button("모델 패키지 등록", disabled=model_package is None or not trusted):
        ok, message = install_model_package(model_package)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)
    st.caption(
        f"등록 위치: {MODEL_DIR} · 새 템플릿은 후보/비활성 상태가 기본입니다. "
        "최종 검증 후 manifest의 enabled=true, deployment_status=approved가 모두 충족되어야 판정에 사용됩니다."
    )


with tab_history:
    st.markdown('<div class="section-label">모델 학습·검증 기록</div>', unsafe_allow_html=True)
    st.info(
        "이 화면은 모델을 즉시 활성화하는 곳이 아니라, 후보 모델의 학습 조건·검증 범위·성능·배포 승인을 추적하는 기록 화면입니다."
    )
    process_frame = pd.DataFrame(
        [
            {"단계": "1. 후보 등록", "필수 기록": "학습 데이터, 특징 목록, 코드 버전, 모델 해시", "운영 상태": "CANDIDATE"},
            {"단계": "2. 독립 검증", "필수 기록": "파일 단위 분할, OK/NG별 성능, 오탐·미탐 사례", "운영 상태": "검증 중"},
            {"단계": "3. 현장 승인", "필수 기록": "LOT·계절·설비 검증, 처리 시간, 작업자 검토", "운영 상태": "승인 대기"},
            {"단계": "4. 운영 배포", "필수 기록": "임계값·롤백 모델·배포 담당자·배포 일시", "운영 상태": "APPROVED / PRODUCTION"},
        ]
    )
    st.dataframe(process_frame, width="stretch", hide_index=True)

    history_rows = []
    for spec in models:
        metrics = spec.metrics or {}
        manifest_path = spec.root_path / "manifest.json"
        history_rows.append(
            {
                "모델": spec.name,
                "버전": spec.version,
                "상태": deployment_state(spec),
                "유형": spec.model_type,
                "특징 프로필": spec.feature_profile,
                "지원 모드": ", ".join(spec.supported_modes),
                "Accuracy": metrics.get("accuracy", np.nan),
                "Precision": metrics.get("precision", np.nan),
                "Recall": metrics.get("recall", np.nan),
                "F1": metrics.get("f1", np.nan),
                "검증 범위": spec.validation_scope or "미기록",
                "최종 수정": datetime.fromtimestamp(manifest_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if manifest_path.exists() else "-",
                "SHA256": spec.sha256[:12] or "-",
            }
        )
    history_frame = pd.DataFrame(history_rows)
    if history_frame.empty:
        st.warning("등록된 후보 모델 기록이 없습니다.")
    else:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("전체 후보", f"{len(history_frame):,}개")
        h2.metric("운영 승인", f"{history_frame['상태'].eq('ACTIVE').sum():,}개")
        h3.metric("검증 범위 기록", f"{history_frame['검증 범위'].ne('미기록').sum():,}개")
        h4.metric("해시 확인", f"{history_frame['SHA256'].ne('-').sum():,}개")
        st.dataframe(history_frame, width="stretch", hide_index=True, height=360)

    st.markdown("#### 운영 전 필수 검증 항목")
    readiness = pd.DataFrame(
        [
            {"검증 항목": "파일 단위 데이터 분리", "이유": "같은 파일의 행이 학습·평가에 동시에 들어가는 누수 방지"},
            {"검증 항목": "행 단위 + 파일 단위 성능", "이유": "국부 이상 탐지력과 최종 팩 판정을 함께 확인"},
            {"검증 항목": "OK 파일별 FPR", "이유": "정상 제품 과검출로 인한 불필요한 재검사 방지"},
            {"검증 항목": "NG 유형별 Recall", "이유": "용량·접촉·센싱와이어·온도센서 이상 누락 여부 확인"},
            {"검증 항목": "신규 LOT·설비·계절 OOD", "이유": "학습 분포와 다른 현장 조건에서의 일반화 확인"},
            {"검증 항목": "추론 시간·롤백", "이유": "실시간 처리 가능성과 배포 장애 대응 확인"},
        ]
    )
    st.dataframe(readiness, width="stretch", hide_index=True)


with tab_data_status:
    st.divider()
    st.markdown('<div class="section-label">추가 데이터 준비</div>', unsafe_allow_html=True)
    st.caption(
        "현재 카탈로그는 팀 프로젝트 Train/Test 두 경로만 사용합니다. 여기서 저장한 파일은 Inbox에 보관되며, 데이터 소스로 별도 승인하기 전에는 분석에 포함되지 않습니다."
    )
    uploads = st.file_uploader("CSV 또는 CSV ZIP", type=["csv", "zip"], accept_multiple_files=True, key="data_uploads")
    if st.button("Inbox에 저장", disabled=not uploads):
        saved = save_uploads(uploads or [])
        st.success(f"{len(saved)}개 CSV를 검토용 Inbox에 저장했습니다. 현재 분석에는 아직 포함되지 않습니다.")
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div class="section-label">데이터 소스 설정</div>', unsafe_allow_html=True)
    source_frame = pd.DataFrame(settings.get("data_sources", []))
    edited = st.data_editor(
        source_frame,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("소스 이름", required=True),
            "path": st.column_config.TextColumn("폴더 경로", required=True),
            "enabled": st.column_config.CheckboxColumn("활성"),
            "recursive": st.column_config.CheckboxColumn("하위 폴더"),
            "role": st.column_config.SelectboxColumn("역할", options=["train", "test", "external", "inbox"]),
        },
        key="data_source_editor",
    )
    if st.button("데이터 소스 저장"):
        clean_sources = []
        for row in edited.to_dict(orient="records"):
            if str(row.get("name", "")).strip() and str(row.get("path", "")).strip():
                clean_sources.append(
                    {
                        "name": str(row["name"]).strip(),
                        "path": str(row["path"]).strip(),
                        "enabled": bool(row.get("enabled", True)),
                        "recursive": bool(row.get("recursive", False)),
                        "role": str(row.get("role", "external")),
                    }
                )
        settings["data_sources"] = clean_sources
        save_settings(settings)
        st.cache_data.clear()
        st.success("데이터 소스 설정을 저장했습니다.")
        st.rerun()

    st.markdown('<div class="section-label">전체 파일 카탈로그</div>', unsafe_allow_html=True)
    st.dataframe(translated_catalog(catalog), width="stretch", hide_index=True)
    st.download_button("카탈로그 CSV", dataframe_csv_bytes(translated_catalog(catalog)), "battery_data_catalog.csv", "text/csv")


footer_mode = (
    f"운영 모델 {active_model.name} {active_model.version}로 판정을 실행합니다."
    if active_model is not None and is_operational_model(active_model)
    else f"검증 후보 {active_model.name} {active_model.version}의 비교 판정을 실행합니다. 운영 모델을 대체하지 않습니다."
    if active_model is not None
    else "현재는 데이터 점검 모드이며, 운영 승인 모델이 등록된 이후에만 모델 판정을 실행합니다."
)
st.caption(f"작업 폴더: {APP_ROOT} · {footer_mode}")
