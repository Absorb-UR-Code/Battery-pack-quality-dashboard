from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_catalog import detect_sensor_columns, downsample_indices, resolve_time_axis


TEAL = "#0F766E"
RED = "#C43D3D"
AMBER = "#B7791F"
GRAPHITE = "#24302E"
MUTED = "#71807D"
GRID = "#DCE3E1"
BLUE = "#0057B8"
VIOLET = "#6A1B9A"
EMERALD = "#00695C"
OCHRE = "#8A5A00"
CHARCOAL = "#37474F"
ANOMALY_FILL = "rgba(220,38,38,0.16)"


def _normalized_anomaly_mask(anomaly_mask: object, length: int) -> np.ndarray:
    if anomaly_mask is None:
        return np.zeros(length, dtype=bool)
    mask = np.asarray(anomaly_mask, dtype=bool).reshape(-1)
    if len(mask) != length:
        raise ValueError(f"anomaly_mask length mismatch: expected {length}, got {len(mask)}")
    return mask


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _midpoint(left: object, right: object) -> object:
    try:
        return left + (right - left) / 2
    except (TypeError, ValueError):
        return left


def _axis_run_edges(values: list[object], start: int, end: int) -> tuple[object, object]:
    count = len(values)
    if count == 1:
        value = values[0]
        if isinstance(value, pd.Timestamp):
            return value - pd.Timedelta(milliseconds=500), value + pd.Timedelta(milliseconds=500)
        try:
            numeric = float(value)
            return numeric - 0.5, numeric + 0.5
        except (TypeError, ValueError):
            return value, value

    x0 = (
        _midpoint(values[start - 1], values[start])
        if start > 0
        else values[0] - (values[1] - values[0]) / 2
    )
    x1 = (
        _midpoint(values[end], values[end + 1])
        if end < count - 1
        else values[-1] + (values[-1] - values[-2]) / 2
    )
    return x0, x1


def _add_anomaly_window_shading(fig: go.Figure, x: pd.Series, anomaly_mask: object) -> None:
    mask = _normalized_anomaly_mask(anomaly_mask, len(x))
    if not mask.any():
        return
    values = x.reset_index(drop=True).tolist()
    for start, end in _true_runs(mask):
        x0, x1 = _axis_run_edges(values, start, end)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=ANOMALY_FILL,
            line_width=0,
            layer="below",
        )


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=430,
        margin=dict(l=42, r=18, t=58, b=88),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Malgun Gothic, Arial, sans-serif", color=GRAPHITE, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="left", x=0),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    return fig


def sensor_envelope_figure(
    df: pd.DataFrame,
    group: str,
    max_points: int = 3000,
    anomaly_mask: object = None,
) -> go.Figure:
    cv_cols, temp_cols = detect_sensor_columns(df.columns)
    cols = cv_cols if group == "voltage" else temp_cols
    title = "셀 전압 분포" if group == "voltage" else "온도 센서 분포"
    y_title = "전압 (V)" if group == "voltage" else "온도 (°C)"
    if not cols:
        return _layout(go.Figure(), f"{title} - 센서 컬럼 없음", "순서", y_title)
    idx = downsample_indices(len(df), max_points)
    values = df.iloc[idx][cols].apply(pd.to_numeric, errors="coerce")
    x_all, x_label, _ = resolve_time_axis(df)
    x = x_all.iloc[idx]
    stats = pd.DataFrame(
        {
            "min": values.min(axis=1),
            "q25": values.quantile(0.25, axis=1),
            "median": values.median(axis=1),
            "q75": values.quantile(0.75, axis=1),
            "max": values.max(axis=1),
        },
        index=values.index,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=stats["max"], line=dict(color="rgba(0,87,184,0.55)", width=1.3), name="최대"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=stats["min"],
            line=dict(color="rgba(0,105,92,0.55)", width=1.3),
            fill="tonexty",
            fillcolor="rgba(0,87,184,0.07)",
            name="최소-최대",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=stats["q75"], line=dict(color="rgba(106,27,154,0.55)", width=1.2), name="Q75"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=stats["q25"],
            line=dict(color="rgba(0,105,92,0.55)", width=1.2),
            fill="tonexty",
            fillcolor="rgba(106,27,154,0.07)",
            name="Q25-Q75",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=stats["median"], line=dict(color="#102A43", width=2.0), name="중앙값"))
    _add_anomaly_window_shading(fig, x_all.reset_index(drop=True), anomaly_mask)
    return _layout(fig, f"{title} · {len(cols)}개 센서", x_label, y_title)


def score_timeline_figure(row_result: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=row_result["row_index"],
            y=row_result["score"],
            mode="lines",
            line=dict(color=TEAL, width=1.3),
            name="이상 점수",
        )
    )
    anomaly = row_result[row_result["predicted_anomaly"].eq(1)]
    fig.add_trace(
        go.Scatter(
            x=anomaly["row_index"],
            y=anomaly["score"],
            mode="markers",
            marker=dict(color=RED, size=5),
            name="이상 판정 행",
        )
    )
    fig.add_hline(y=threshold, line_color=RED, line_dash="dash", annotation_text=f"임계값 {threshold:.4g}")
    return _layout(fig, "행 단위 이상 점수", "행 순서", "모델 점수")


def feature_timeline_figure(features: pd.DataFrame) -> go.Figure:
    voltage_cols = [c for c in ["cv_range", "cv_resid_maxabs"] if c in features.columns]
    temp_cols = [c for c in ["temp_range", "temp_resid_maxabs", "temp_pair_gap_max"] if c in features.columns]
    if not voltage_cols and not temp_cols:
        return _layout(go.Figure(), "핵심 파생변수 - 사용 가능한 값 없음", "행 순서", "값")
    x = np.arange(1, len(features) + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {
        "cv_range": TEAL,
        "cv_resid_maxabs": RED,
        "temp_range": AMBER,
        "temp_resid_maxabs": "#547A6C",
        "temp_pair_gap_max": "#7A5C61",
    }
    for col in voltage_cols:
        fig.add_trace(
            go.Scatter(x=x, y=features[col], mode="lines", name=col, line=dict(color=colors[col], width=1.2)),
            secondary_y=False,
        )
    for col in temp_cols:
        fig.add_trace(
            go.Scatter(x=x, y=features[col], mode="lines", name=col, line=dict(color=colors[col], width=1.2)),
            secondary_y=True,
        )
    fig = _layout(fig, "핵심 파생변수 추이", "행 순서", "전압 편차 (V)")
    fig.update_yaxes(title_text="전압 편차 (V)", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="온도 편차 (°C)", secondary_y=True, showgrid=False)
    return fig


def operating_signal_figure(df: pd.DataFrame, max_points: int = 1500) -> go.Figure:
    """Display the three operating signals operators use most often."""
    if df.empty:
        return _layout(go.Figure(), "팩 운전 신호 - 데이터 없음", "순서", "값")

    idx = downsample_indices(len(df), max_points)
    view = df.iloc[idx]
    x_all, x_label, _ = resolve_time_axis(df)
    x = x_all.iloc[idx]
    columns_lower = {str(col).lower(): str(col) for col in df.columns}

    def find_col(*names: str) -> str | None:
        return next((columns_lower[name.lower()] for name in names if name.lower() in columns_lower), None)

    current_col = find_col("Current", "PackCurrent", "I")
    voltage_col = find_col("Voltage", "PackVoltage", "V")
    rsoc_col = find_col("RSOCavg", "SOC", "USOCavg")
    soh_col = find_col("SOH", "RSOHavg")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("팩 전류", "팩 전압", "SOC / SOH"),
    )
    if current_col:
        fig.add_trace(
            go.Scatter(x=x, y=pd.to_numeric(view[current_col], errors="coerce"), name="Current", line=dict(color=RED, width=1.4)),
            row=1,
            col=1,
        )
    if voltage_col:
        fig.add_trace(
            go.Scatter(x=x, y=pd.to_numeric(view[voltage_col], errors="coerce"), name="Pack Voltage", line=dict(color=TEAL, width=1.4)),
            row=2,
            col=1,
        )
    if rsoc_col:
        fig.add_trace(
            go.Scatter(x=x, y=pd.to_numeric(view[rsoc_col], errors="coerce"), name="RSOCavg", line=dict(color=AMBER, width=1.4)),
            row=3,
            col=1,
        )
    if soh_col:
        fig.add_trace(
            go.Scatter(x=x, y=pd.to_numeric(view[soh_col], errors="coerce"), name="SOH", line=dict(color="#547A6C", width=1.2, dash="dot")),
            row=3,
            col=1,
        )

    fig.update_layout(
        title={"text": "팩 운전 신호", "x": 0.01, "xanchor": "left"},
        height=590,
        margin=dict(l=50, r=20, t=72, b=78),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Malgun Gothic, Arial, sans-serif", color=GRAPHITE, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="전류 (A)", row=1, col=1, gridcolor=GRID, zeroline=True, zerolinecolor="#AAB7B3")
    fig.update_yaxes(title_text="전압 (V)", row=2, col=1, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text="비율 (%)", row=3, col=1, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(title_text=x_label, row=3, col=1, gridcolor=GRID, zeroline=False)
    return fig


def kpi_timeline_figure(
    df: pd.DataFrame,
    kpi_frame: pd.DataFrame,
    metric: str,
    max_points: int = 1500,
) -> go.Figure:
    """Render one operator-selected sensor KPI on the shared measurement time axis."""
    meta = {
        "cv_mean": ("셀 전압 평균", "전압 (V)", TEAL),
        "cv_std": ("셀 전압 편차", "표준편차 (V)", RED),
        "temp_mean": ("온도 평균", "온도 (°C)", AMBER),
        "temp_range": ("온도 범위", "온도 (°C)", "#547A6C"),
        "temp_std": ("온도 편차", "표준편차 (°C)", "#7A5C61"),
    }
    if df.empty or metric not in meta or metric not in kpi_frame.columns:
        return _layout(go.Figure(), "팩 운전 신호 - 데이터 없음", "순서", "값")

    idx = downsample_indices(len(df), max_points)
    x_all, x_label, _ = resolve_time_axis(df)
    x = x_all.iloc[idx]
    view = kpi_frame.iloc[idx]
    title, y_title, color = meta[metric]
    fig = go.Figure()

    if metric == "temp_range":
        fig.add_trace(
            go.Scatter(
                x=x,
                y=view["temp_max"],
                mode="lines",
                name="최고 온도",
                line=dict(color=RED, width=1.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=view["temp_min"],
                mode="lines",
                name="최저 온도",
                line=dict(color=TEAL, width=1.4),
                fill="tonexty",
                fillcolor="rgba(15,118,110,0.10)",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=view[metric],
                mode="lines",
                name=title,
                line=dict(color=color, width=1.6),
            )
        )

    return _layout(fig, f"팩 운전 신호 · {title}", x_label, y_title)


def draggable_kpi_workspace_html(
    df: pd.DataFrame,
    kpi_frame: pd.DataFrame,
    storage_key: str,
    max_points: int = 1200,
    anomaly_mask: object = None,
) -> str:
    """Build a self-contained drag-and-drop KPI chart workspace for Streamlit."""
    idx = downsample_indices(len(df), max_points) if len(df) else np.array([], dtype=int)
    x_all, _, _ = resolve_time_axis(df)
    x_values = [str(value) for value in x_all.iloc[idx].tolist()] if len(idx) else []
    view = kpi_frame.iloc[idx] if len(idx) else kpi_frame.iloc[0:0]

    def numbers(column: str) -> list[float | None]:
        if column not in view.columns:
            return [None] * len(view)
        values = pd.to_numeric(view[column], errors="coerce")
        return [float(value) if np.isfinite(value) else None for value in values]

    def latest(column: str) -> float:
        if column not in kpi_frame.columns or kpi_frame.empty:
            return np.nan
        value = pd.to_numeric(pd.Series([kpi_frame[column].iloc[-1]]), errors="coerce").iloc[0]
        return float(value) if np.isfinite(value) else np.nan

    def formatted(value: float, digits: int, unit: str) -> str:
        return f"{value:.{digits}f} {unit}" if np.isfinite(value) else "-"

    temp_min = latest("temp_min")
    temp_max = latest("temp_max")
    full_anomaly_mask = _normalized_anomaly_mask(anomaly_mask, len(df))
    payload = {
        "x": x_values,
        "anomaly_mask": full_anomaly_mask[idx].tolist() if len(idx) else [],
        "metrics": {
            "cv_mean": {
                "label": "셀 전압 평균",
                "unit": "V",
                "current": formatted(latest("cv_mean"), 4, "V"),
                "color": BLUE,
                "kind": "line",
                "series": numbers("cv_mean"),
            },
            "cv_std": {
                "label": "셀 전압 편차",
                "unit": "V",
                "current": formatted(latest("cv_std"), 5, "V"),
                "color": VIOLET,
                "kind": "line",
                "series": numbers("cv_std"),
            },
            "temp_mean": {
                "label": "온도 평균",
                "unit": "°C",
                "current": formatted(latest("temp_mean"), 2, "°C"),
                "color": EMERALD,
                "kind": "line",
                "series": numbers("temp_mean"),
            },
            "temp_range": {
                "label": "온도 범위",
                "unit": "°C",
                "current": (
                    f"{temp_min:.2f} °C ~ {temp_max:.2f} °C"
                    if np.isfinite(temp_min) and np.isfinite(temp_max)
                    else "-"
                ),
                "color": OCHRE,
                "kind": "band",
                "low": numbers("temp_min"),
                "high": numbers("temp_max"),
            },
            "temp_std": {
                "label": "온도 편차",
                "unit": "°C",
                "current": formatted(latest("temp_std"), 3, "°C"),
                "color": CHARCOAL,
                "kind": "line",
                "series": numbers("temp_std"),
            },
        },
    }

    template = r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>
  :root { --ink:#172220; --muted:#64736f; --line:#d8e0de; --canvas:#f3f6f5; --teal:#0f766e; --red:#c43d3d; }
  * { box-sizing:border-box; }
  body { margin:0; background:transparent; color:var(--ink); font-family:"Malgun Gothic",Arial,sans-serif; }
  .shell { padding:2px 1px 12px; }
  .head { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin:0 0 10px; }
  .head h3 { margin:0; font-size:18px; letter-spacing:0; }
  .head p { margin:3px 0 0; color:var(--muted); font-size:12px; }
  .drag-note { color:var(--muted); font-size:12px; text-align:right; }
  .palette { display:block; }
  .kpi-grid { display:grid; gap:10px; margin-bottom:10px; }
  .kpi-grid.voltage { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .kpi-grid.temperature { grid-template-columns:repeat(3,minmax(0,1fr)); margin-bottom:18px; }
  .kpi-card { min-height:92px; border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:6px; background:#fff; background:color-mix(in srgb,var(--accent) 6%,#fff); padding:13px 14px; cursor:grab; user-select:none; transition:border-color .15s,box-shadow .15s,transform .15s; }
  .kpi-card:hover { border-color:var(--accent); box-shadow:0 5px 16px rgba(23,34,32,.08); transform:translateY(-1px); }
  .kpi-card:active { cursor:grabbing; }
  .kpi-label { color:var(--muted); font-size:12px; font-weight:700; display:flex; justify-content:space-between; gap:8px; }
  .drag-icon { color:var(--accent); font-size:14px; }
  .kpi-value { margin-top:8px; font-size:21px; line-height:1.2; font-weight:700; white-space:normal; }
  .workspace-head { display:flex; align-items:center; justify-content:space-between; gap:12px; border-bottom:2px solid #20312e; padding-bottom:9px; }
  .workspace-head h3 { margin:0; font-size:18px; }
  .remove-zone { display:none; }
  .remove-zone.drag-over { background:#ffe2e2; border-color:var(--red); }
  .workspace { display:flex; flex-direction:column; gap:12px; padding:13px 0 4px; }
  .workspace-placeholder { min-height:205px; display:flex; align-items:center; justify-content:center; text-align:center; border:1px dashed #aebbb8; border-radius:6px; color:var(--muted); background:#fafcfc; font-size:13px; line-height:1.65; }
  .signal-row { position:relative; min-width:0; border:1px solid transparent; border-radius:6px; background:transparent; transition:border-color .15s,background .15s; }
  .signal-row.empty-signal-row { display:none; min-height:190px; }
  body.layout-dragging .signal-row { border-style:dashed; border-color:#aebbb8; background:#fafcfc; padding:10px; }
  body.layout-dragging .signal-row.empty-signal-row { display:block; }
  .signal-row.drag-over { border-color:var(--teal); background:rgba(15,118,110,.045); }
  .signal-row.drop-denied { border-color:#cf7777; background:#fff5f5; }
  .row-content { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; min-width:0; }
  .row-content.single .chart-panel { grid-column:1/-1; }
  .row-empty { grid-column:1/-1; min-height:168px; display:flex; align-items:center; justify-content:center; text-align:center; color:var(--muted); font-size:12px; line-height:1.6; }
  .drop-guide { display:none; position:absolute; top:10px; bottom:10px; z-index:4; border:2px solid var(--teal); border-radius:6px; background:rgba(15,118,110,.08); pointer-events:none; }
  .signal-row.drag-over .drop-guide { display:block; }
  .signal-row.target-empty .drop-guide { left:10px; right:10px; }
  .signal-row.target-left .drop-guide { left:10px; width:calc(50% - 16px); }
  .signal-row.target-right .drop-guide { right:10px; width:calc(50% - 16px); }
  .signal-row.drop-denied .drop-guide { display:none; }
  .chart-empty { min-height:165px; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:12px; }
  .chart-panel { border:1px solid var(--line); border-radius:6px; background:#fff; padding:12px 12px 9px; cursor:grab; min-width:0; transition:opacity .15s,box-shadow .15s; }
  .chart-panel:hover { box-shadow:0 6px 18px rgba(23,34,32,.08); }
  .chart-panel.dragging { opacity:.45; }
  .chart-head { display:flex; align-items:flex-start; justify-content:space-between; gap:10px; margin-bottom:5px; }
  .chart-title { font-size:14px; font-weight:800; }
  .chart-current { color:var(--muted); font-size:12px; margin-top:2px; }
  .close { width:30px; height:30px; border:0; background:transparent; color:var(--muted); font-size:19px; cursor:pointer; border-radius:4px; }
  .close:hover { background:#f1f4f3; color:var(--red); }
  .chart-svg { width:100%; height:185px; display:block; overflow:visible; }
  .axis-text { fill:#71807d; font-size:10px; }
  .grid-line { stroke:#e3e9e7; stroke-width:1; }
  @media (max-width:720px) {
    .head { align-items:flex-start; flex-direction:column; }
    .drag-note { text-align:left; }
    .kpi-grid.temperature { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .row-content { grid-template-columns:1fr; }
    .row-content .chart-panel { grid-column:1/-1; }
  }
</style>
</head>
<body>
<div class="shell">
  <div class="head">
    <div><h3>실시간 KPI</h3><p>KPI 카드를 원하는 위치에 놓고, 이미 배치된 그래프도 다시 끌어 순서를 바꿀 수 있습니다.</p></div>
    <div class="drag-note">그래프 패널을 작업영역 밖으로 끌면 제거됩니다.</div>
  </div>
  <div id="palette" class="palette">
    <div id="voltageCards" class="kpi-grid voltage"></div>
    <div id="temperatureCards" class="kpi-grid temperature"></div>
  </div>
  <div class="workspace-head">
    <h3>팩 운전 신호</h3>
    <div id="removeZone" class="remove-zone">그래프 제거</div>
  </div>
  <div id="workspace" class="workspace" aria-label="팩 운전 신호 작업영역"></div>
</div>
<script>
let payload = __KPI_DATA__;
const storageKey = __STORAGE_KEY__;
const metricOrder = ["cv_mean","cv_std","temp_mean","temp_range","temp_std"];
const voltageKeys = new Set(["cv_mean","cv_std"]);
const ROW_COUNT = 3;
const MAX_PER_ROW = 2;
const workspace = document.getElementById("workspace");
const palette = document.getElementById("palette");
const removeZone = document.getElementById("removeZone");
let dragging = null;
let droppedInside = false;

function emptyLayout() { return Array.from({length:ROW_COUNT}, () => []); }
function normalizeLayout(saved) {
  const result = emptyLayout();
  const seen = new Set();
  if (!Array.isArray(saved)) return result;
  if (saved.every(Array.isArray)) {
    saved.slice(0,ROW_COUNT).forEach((row,rowIndex) => {
      row.forEach(metric => {
        if (result[rowIndex].length < MAX_PER_ROW && payload.metrics[metric] && !seen.has(metric)) {
          result[rowIndex].push(metric); seen.add(metric);
        }
      });
    });
    return result;
  }
  saved.forEach(metric => {
    if (!payload.metrics[metric] || seen.has(metric)) return;
    const rowIndex = result.findIndex(row => row.length < MAX_PER_ROW);
    if (rowIndex >= 0) { result[rowIndex].push(metric); seen.add(metric); }
  });
  return result;
}
function loadLayout() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    return normalizeLayout(saved);
  } catch (_) { return emptyLayout(); }
}
let layout = loadLayout();
function saveLayout() { try { localStorage.setItem(storageKey, JSON.stringify(layout)); } catch (_) {} }
function activeMetrics() { return layout.flat(); }
function finite(value) { return typeof value === "number" && Number.isFinite(value); }
function fmt(value, unit) {
  if (!finite(value)) return "-";
  const digits = unit === "V" ? 4 : 2;
  return value.toFixed(digits);
}
function shortX(value) {
  const text = String(value ?? "");
  return text.length > 20 ? text.slice(-20) : text;
}
function pathFor(values, sx, sy) {
  let d = ""; let open = false;
  values.forEach((value, index) => {
    if (!finite(value)) { open = false; return; }
    d += `${open ? " L" : "M"}${sx(index).toFixed(2)},${sy(value).toFixed(2)}`;
    open = true;
  });
  return d;
}
function anomalyOverlay(count, sx, left, right, top, plotH) {
  const rawMask = Array.isArray(payload.anomaly_mask) ? payload.anomaly_mask : [];
  const mask = Array.from({length:count}, (_, index) => Boolean(rawMask[index]));
  let rectangles = "";
  let start = null;
  for (let index=0; index<=count; index++) {
    const active = index < count && mask[index];
    if (active && start === null) start = index;
    if (!active && start !== null) {
      const end = index - 1;
      const x0 = start === 0 ? left : (sx(start - 1) + sx(start)) / 2;
      const x1 = end === count - 1 ? right : (sx(end) + sx(end + 1)) / 2;
      rectangles += `<rect x="${x0.toFixed(2)}" y="${top}" width="${Math.max(0,x1-x0).toFixed(2)}" height="${plotH}" fill="rgba(220,38,38,.16)"/>`;
      start = null;
    }
  }
  return rectangles;
}
function chartSvg(item, wide=false) {
  const width = wide ? 1440 : 720, height = 190, left = 54, right = 16, top = 12, bottom = 31;
  const plotW = width - left - right, plotH = height - top - bottom;
  const arrays = item.kind === "band" ? [item.low || [], item.high || []] : [item.series || []];
  const all = arrays.flat().filter(finite);
  if (!all.length) return `<div class="chart-empty">표시 가능한 KPI 값이 없습니다.</div>`;
  let min = Math.min(...all), max = Math.max(...all);
  const spread = max - min;
  const pad = spread > 0 ? spread * .12 : Math.max(Math.abs(max) * .02, .01);
  min -= pad; max += pad;
  const count = Math.max(...arrays.map(values => values.length), 1);
  const sx = index => count <= 1 ? left + plotW / 2 : left + (index / (count - 1)) * plotW;
  const sy = value => top + ((max - value) / (max - min)) * plotH;
  const anomalyBackground = anomalyOverlay(count, sx, left, width-right, top, plotH);
  let grid = "";
  for (let i=0;i<5;i++) {
    const y = top + (i/4)*plotH;
    const value = max - (i/4)*(max-min);
    grid += `<line class="grid-line" x1="${left}" y1="${y}" x2="${width-right}" y2="${y}"/>`;
    grid += `<text class="axis-text" x="${left-8}" y="${y+3}" text-anchor="end">${fmt(value,item.unit)}</text>`;
  }
  let marks = "";
  if (item.kind === "band") {
    const high = item.high || [], low = item.low || [];
    const upper = high.map((v,i) => finite(v) ? `${sx(i)},${sy(v)}` : null).filter(Boolean);
    const lower = low.map((v,i) => finite(v) ? `${sx(i)},${sy(v)}` : null).filter(Boolean).reverse();
    const highPath = pathFor(high,sx,sy), lowPath = pathFor(low,sx,sy);
    if (upper.length && lower.length) marks += `<polygon points="${upper.concat(lower).join(" ")}" fill="${item.color}" fill-opacity=".10"/>`;
    marks += `<path d="${highPath}" fill="none" stroke="rgba(255,255,255,.86)" stroke-width="5.4"/>`;
    marks += `<path d="${lowPath}" fill="none" stroke="rgba(255,255,255,.86)" stroke-width="5.4" stroke-dasharray="7 5"/>`;
    marks += `<path d="${highPath}" fill="none" stroke="${item.color}" stroke-width="2.5"/>`;
    marks += `<path d="${lowPath}" fill="none" stroke="${item.color}" stroke-width="2.5" stroke-dasharray="7 5"/>`;
    if (count === 1 && finite(high[0]) && finite(low[0])) {
      marks += `<line x1="${sx(0)}" y1="${sy(high[0])}" x2="${sx(0)}" y2="${sy(low[0])}" stroke="${item.color}" stroke-width="5" opacity=".42"/>`;
    }
  } else {
    const values = item.series || [];
    const seriesPath = pathFor(values,sx,sy);
    marks += `<path d="${seriesPath}" fill="none" stroke="rgba(255,255,255,.86)" stroke-width="5.2"/>`;
    marks += `<path d="${seriesPath}" fill="none" stroke="${item.color}" stroke-width="2.4"/>`;
    if (count === 1 && finite(values[0])) {
      marks += `<circle cx="${sx(0)}" cy="${sy(values[0])}" r="6" fill="rgba(255,255,255,.9)"/>`;
      marks += `<circle cx="${sx(0)}" cy="${sy(values[0])}" r="4" fill="${item.color}"/>`;
    }
  }
  const firstX = shortX(payload.x[0]);
  const lastX = shortX(payload.x[payload.x.length-1]);
  return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${item.label} 추이">
    ${anomalyBackground}${grid}${marks}
    <text class="axis-text" x="${left}" y="${height-8}" text-anchor="start">${firstX}</text>
    <text class="axis-text" x="${width-right}" y="${height-8}" text-anchor="end">${lastX}</text>
  </svg>`;
}
function removeMetric(metric) {
  layout = layout.map(row => row.filter(key => key !== metric));
  saveLayout(); renderWorkspace();
}
function placeMetric(metric, rowIndex, insertIndex) {
  if (!payload.metrics[metric] || rowIndex < 0 || rowIndex >= ROW_COUNT) return false;
  const previous = layout.map(row => [...row]);
  layout = layout.map(row => row.filter(key => key !== metric));
  const target = layout[rowIndex];
  if (target.length >= MAX_PER_ROW) {
    layout = previous;
    return false;
  }
  const safeIndex = Math.max(0, Math.min(Number(insertIndex) || 0, target.length));
  target.splice(safeIndex, 0, metric);
  saveLayout(); renderWorkspace();
  return true;
}
function addMetric(metric, rowIndex=null, insertIndex=null) {
  if (!payload.metrics[metric] || activeMetrics().includes(metric)) return false;
  const targetRow = Number.isInteger(rowIndex) ? rowIndex : layout.findIndex(row => row.length < MAX_PER_ROW);
  if (targetRow < 0) return false;
  const targetIndex = Number.isInteger(insertIndex) ? insertIndex : layout[targetRow].length;
  return placeMetric(metric, targetRow, targetIndex);
}
function makeCard(metric) {
  const item = payload.metrics[metric];
  const card = document.createElement("div");
  card.className = "kpi-card"; card.draggable = true; card.dataset.metric = metric;
  card.tabIndex = 0; card.setAttribute("role", "button"); card.setAttribute("aria-label", `${item.label} 그래프 추가`);
  card.style.setProperty("--accent", item.color);
  card.innerHTML = `<div class="kpi-label"><span>${item.label}</span><span class="drag-icon">+</span></div><div class="kpi-value">${item.current}</div>`;
  card.addEventListener("dragstart", event => {
    dragging = {type:"kpi", metric}; droppedInside = false;
    document.body.classList.add("layout-dragging");
    if (!activeMetrics().length) renderWorkspace(true);
    event.dataTransfer.effectAllowed = "copy"; event.dataTransfer.setData("text/plain", metric);
  });
  card.addEventListener("dragend", () => {
    document.body.classList.remove("layout-dragging");
    if (!activeMetrics().length) renderWorkspace(false);
    dragging = null; droppedInside = false;
  });
  card.addEventListener("click", () => addMetric(metric));
  card.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); addMetric(metric); }
  });
  return card;
}
function makePanel(metric, wide=false) {
  const item = payload.metrics[metric];
  const panel = document.createElement("section");
  panel.className = "chart-panel"; panel.draggable = true; panel.dataset.metric = metric;
  panel.innerHTML = `<div class="chart-head"><div><div class="chart-title">${item.label}</div><div class="chart-current">현재 ${item.current}</div></div><button class="close" title="그래프 제거" aria-label="${item.label} 그래프 제거">×</button></div>${chartSvg(item,wide)}`;
  panel.querySelector(".close").addEventListener("click", event => { event.stopPropagation(); removeMetric(metric); });
  panel.addEventListener("dragstart", event => {
    dragging = {type:"panel", metric, sourceRow:layout.findIndex(row => row.includes(metric))};
    droppedInside = false; panel.classList.add("dragging"); document.body.classList.add("layout-dragging");
    event.dataTransfer.effectAllowed = "move"; event.dataTransfer.setData("text/plain", metric);
  });
  panel.addEventListener("dragend", () => {
    panel.classList.remove("dragging"); document.body.classList.remove("layout-dragging");
    if (dragging && dragging.type === "panel" && !droppedInside) removeMetric(metric);
    dragging = null; droppedInside = false;
  });
  return panel;
}
function renderCards() {
  const voltage = document.getElementById("voltageCards"), temperature = document.getElementById("temperatureCards");
  voltage.replaceChildren(); temperature.replaceChildren();
  metricOrder.forEach(metric => (voltageKeys.has(metric) ? voltage : temperature).appendChild(makeCard(metric)));
}
function updatePayload(nextPayload) {
  if (!nextPayload || typeof nextPayload !== "object" || !nextPayload.metrics) return;
  payload = nextPayload;
  layout = normalizeLayout(layout);
  renderCards();
  renderWorkspace();
}
function renderWorkspace(showDropRows=false) {
  workspace.replaceChildren();
  if (!activeMetrics().length && !showDropRows) {
    const placeholder = document.createElement("div"); placeholder.className = "workspace-placeholder";
    placeholder.innerHTML = "KPI 카드를 이곳으로 드래그하면 해당 시계열 그래프가 생성됩니다.<br>그래프를 밖으로 끌거나 우측 제거 구역에 놓으면 사라집니다.";
    workspace.appendChild(placeholder); return;
  }
  layout.forEach((metrics,rowIndex) => {
    const row = document.createElement("section");
    row.className = `signal-row ${metrics.length ? "" : "empty-signal-row"}`; row.dataset.rowIndex = String(rowIndex);
    row.setAttribute("aria-label", `그래프 배치 위치 ${rowIndex+1}`);

    const content = document.createElement("div");
    content.className = `row-content ${metrics.length === 1 ? "single" : metrics.length === 0 ? "empty-row" : "double"}`;
    if (!metrics.length) {
      const empty = document.createElement("div"); empty.className = "row-empty";
      empty.textContent = "여기에 놓기";
      content.appendChild(empty);
    } else {
      metrics.forEach(metric => content.appendChild(makePanel(metric,metrics.length === 1)));
    }
    row.appendChild(content);

    const guide = document.createElement("div"); guide.className = "drop-guide";
    row.appendChild(guide);
    bindRowDrop(row,rowIndex);
    workspace.appendChild(row);
  });
}
function clearRowTarget(row) {
  row.classList.remove("drag-over","drop-denied","target-empty","target-left","target-right");
}
function rowInsertIndex(row,rowIndex,event) {
  if (!layout[rowIndex].length) return 0;
  const rect = row.getBoundingClientRect();
  return event.clientX < rect.left + rect.width / 2 ? 0 : layout[rowIndex].length;
}
function canPlaceInRow(metric,rowIndex) {
  return layout[rowIndex].length < MAX_PER_ROW || layout[rowIndex].includes(metric);
}
function updateRowTarget(row,rowIndex,event) {
  clearRowTarget(row); row.classList.add("drag-over");
  if (!dragging || !canPlaceInRow(dragging.metric,rowIndex)) {
    row.classList.add("drop-denied"); return;
  }
  if (!layout[rowIndex].length) row.classList.add("target-empty");
  else row.classList.add(rowInsertIndex(row,rowIndex,event) === 0 ? "target-left" : "target-right");
}
function bindRowDrop(row,rowIndex) {
  row.addEventListener("dragover", event => { event.preventDefault(); updateRowTarget(row,rowIndex,event); });
  row.addEventListener("dragleave", event => { if (!row.contains(event.relatedTarget)) clearRowTarget(row); });
  row.addEventListener("drop", event => {
    event.preventDefault(); event.stopPropagation();
    if (!dragging) { clearRowTarget(row); return; }
    const insertIndex = rowInsertIndex(row,rowIndex,event);
    droppedInside = true;
    if (activeMetrics().includes(dragging.metric)) placeMetric(dragging.metric,rowIndex,insertIndex);
    else addMetric(dragging.metric,rowIndex,insertIndex);
    document.body.classList.remove("layout-dragging");
    clearRowTarget(row);
  });
}
function removalDragOver(event) { event.preventDefault(); removeZone.classList.add("drag-over"); }
function removalDrop(event) {
  event.preventDefault(); event.stopPropagation(); removeZone.classList.remove("drag-over");
  if (dragging && dragging.type === "panel") removeMetric(dragging.metric);
  document.body.classList.remove("layout-dragging");
  droppedInside = false;
}
removeZone.addEventListener("dragover", removalDragOver);
removeZone.addEventListener("dragleave", () => removeZone.classList.remove("drag-over"));
removeZone.addEventListener("drop", removalDrop);
palette.addEventListener("dragover", event => event.preventDefault());
palette.addEventListener("drop", event => {
  event.preventDefault();
  if (dragging && dragging.type === "panel") removeMetric(dragging.metric);
  document.body.classList.remove("layout-dragging");
  droppedInside = false;
});
document.addEventListener("dragover", event => event.preventDefault());
document.addEventListener("drop", event => {
  if (dragging && dragging.type === "panel" && !workspace.contains(event.target)) removeMetric(dragging.metric);
  document.body.classList.remove("layout-dragging");
});
window.addEventListener("message", event => {
  const message = event.data || {};
  if (message.type === "battery-pack:kpi-update") updatePayload(message.payload);
});
window.__kpiWorkspace = {add:addMetric, remove:removeMetric, updatePayload};
renderCards(); renderWorkspace();
</script>
</body>
</html>
"""
    return (
        template.replace("__KPI_DATA__", json.dumps(payload, ensure_ascii=False))
        .replace("__STORAGE_KEY__", json.dumps(storage_key, ensure_ascii=False))
    )


def sensor_heatmap_figure(df: pd.DataFrame, group: str, max_rows: int = 500) -> go.Figure:
    cv_cols, temp_cols = detect_sensor_columns(df.columns)
    cols = cv_cols if group == "voltage" else temp_cols
    title = "셀 전압 히트맵" if group == "voltage" else "온도 센서 히트맵"
    if not cols:
        return _layout(go.Figure(), f"{title} - 센서 컬럼 없음", "행 순서", "센서")
    idx = downsample_indices(len(df), max_rows)
    values = df.iloc[idx][cols].apply(pd.to_numeric, errors="coerce").T
    fig = go.Figure(
        data=go.Heatmap(
            z=values.to_numpy(),
            x=idx + 1,
            y=cols,
            colorscale="Viridis",
            colorbar=dict(title="V" if group == "voltage" else "°C"),
            hovertemplate="행 %{x}<br>센서 %{y}<br>값 %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(height=max(420, min(800, 220 + len(cols) * 5)))
    return _layout(fig, f"{title} · 시간 {len(idx)}점", "행 순서", "센서")


def ranking_bar_figure(ranking: pd.DataFrame, group: str, top_n: int = 15) -> go.Figure:
    if ranking.empty:
        return _layout(go.Figure(), "센서 이탈 순위 - 데이터 없음", "P95 절대 잔차", "센서")
    plot = ranking.head(top_n).sort_values("p95_abs_residual")
    fig = px.bar(
        plot,
        x="p95_abs_residual",
        y="sensor",
        orientation="h",
        color="p95_abs_residual",
        color_continuous_scale=[[0, "#9CC5BD"], [1, RED]],
        hover_data=["mean_abs_residual", "max_abs_residual", "missing_rate", "zero_rate"],
    )
    fig.update_layout(coloraxis_showscale=False)
    unit = "V" if group == "voltage" else "°C"
    return _layout(fig, "센서 이탈 순위", f"P95 절대 잔차 ({unit})", "센서")


def batch_status_figure(batch: pd.DataFrame) -> go.Figure:
    if batch.empty or "status" not in batch.columns:
        return _layout(go.Figure(), "배치 판정 - 결과 없음", "상태", "파일 수")
    counts = batch["status"].value_counts().rename_axis("status").reset_index(name="count")
    colors = {"NORMAL": TEAL, "NG_REVIEW": RED, "ERROR": MUTED}
    fig = px.bar(counts, x="status", y="count", color="status", color_discrete_map=colors, text="count")
    fig.update_traces(textposition="outside")
    return _layout(fig, "배치 판정 현황", "상태", "파일 수")
