from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_settings
from core.data_catalog import audit_data_quality, build_catalog, detect_sensor_columns, read_csv_resilient
from core.diagnostic_display import (
    evaluated_row_positions,
    fault_domain_coverage,
    latest_fault_payload,
    latest_scored_prediction,
    replay_progress_high_water,
    replayed_row_positions,
    reached_fault_payloads,
)
from core.features import build_row_features, build_sensor_kpis, sensor_snapshot_matrix
from core.fault_log import (
    batch_fault_events,
    build_fault_event,
    display_mode,
    extract_fault_metadata,
    parse_probabilities,
    recommendation_for,
    representative_fault_events,
)
from core.kpi_workspace_component import _COMPONENT_DIR
from core.model_registry import discover_models
from core.visuals import draggable_kpi_workspace_html


def main() -> None:
    sparse_rows = pd.DataFrame(
        {
            "row_index": range(1, 151),
            "score": [float("nan")] * 150,
            "predicted_anomaly": [0] * 150,
        }
    )
    sparse_rows.loc[99, ["score", "predicted_anomaly"]] = [0.93, 1]
    sparse_rows.loc[119, ["score", "predicted_anomaly"]] = [0.12, 0]
    sparse_rows.loc[139, ["score", "predicted_anomaly"]] = [0.88, 1]
    sparse_result = {
        "row_result": sparse_rows,
        "details": {
            "window_size": 100,
            "fault_by_row": {
                "100": {
                    "fault_type": "temperature sensor",
                    "suspect_sensors": ["M16T02"],
                },
                "140": {
                    "fault_type": "sensing wire",
                    "suspect_sensors": ["M16CV11"],
                },
            },
        },
    }

    sparse_flag, sparse_score, sparse_row = latest_scored_prediction(sparse_rows, 96)
    assert not sparse_flag and pd.isna(sparse_score) and sparse_row is None
    sparse_flag, sparse_score, sparse_row = latest_scored_prediction(sparse_rows, 101)
    assert sparse_flag and sparse_score == 0.93 and sparse_row == 100
    sparse_flag, sparse_score, sparse_row = latest_scored_prediction(sparse_rows, 121)
    assert not sparse_flag and sparse_score == 0.12 and sparse_row == 120
    sparse_flag, sparse_score, sparse_row = latest_scored_prediction(sparse_rows, 145)
    assert sparse_flag and sparse_score == 0.88 and sparse_row == 140
    assert evaluated_row_positions(sparse_rows, 150).tolist() == [99, 119, 139]
    assert evaluated_row_positions(sparse_rows, 150, end_position=121).tolist() == [99, 119]
    assert replayed_row_positions(4_864, 100).tolist() == list(range(100))
    assert replayed_row_positions(100, 0).size == 0
    assert replay_progress_high_water(100, 1, 4_864) == 100
    assert replay_progress_high_water(100, 130, 4_864) == 130
    assert replay_progress_high_water(4_864, 5_000, 4_864) == 4_864
    assert [row for row, _ in reached_fault_payloads(sparse_result, 101)] == [100]
    assert [row for row, _ in reached_fault_payloads(sparse_result, 145)] == [100, 140]
    payload_row, payload = latest_fault_payload(sparse_result, 145)
    assert payload_row == 140 and payload["suspect_sensors"] == ["M16CV11"]

    sparse_domains = fault_domain_coverage(sparse_result, 150, end_position=145)
    assert sparse_domains.loc[:99, "temperature_fault"].all()
    assert not sparse_domains.loc[100:, "temperature_fault"].any()
    assert sparse_domains.loc[40:139, "voltage_fault"].all()
    assert not sparse_domains.loc[:39, "voltage_fault"].any()
    assert not sparse_domains.loc[140:, "voltage_fault"].any()

    dense_rows = pd.DataFrame(
        {
            "row_index": range(1, 111),
            "score": [float("nan")] * 99 + [0.8] * 11,
            "predicted_anomaly": [0] * 99 + [1] * 11,
        }
    )
    dense_flag, dense_score, dense_row = latest_scored_prediction(dense_rows, 105)
    assert dense_flag and dense_score == 0.8 and dense_row == 105

    representative_row_event = build_fault_event(
        {
            "summary": {
                "status": "NG_REVIEW",
                "model_id": "fault-row-selection",
                "model_version": "1.0",
                "mode": "CHG",
            },
            "details": {
                "window_size": 100,
                "fault_by_row": {
                    "100": {
                        "fault_type": "용접 불량",
                        "fault_confidence": 0.37,
                        "suspect_sensors": ["M07CV02"],
                    },
                    "140": {
                        "fault_type": "용접 불량",
                        "fault_confidence": 0.91,
                        "suspect_sensors": ["M07CV08"],
                    },
                },
            },
        },
        source_file="Test06_NG_chg.csv",
        origin="completed-file",
        occurrence_key="run-1",
    )
    assert representative_row_event is not None
    assert representative_row_event["detected_row"] == 140
    assert representative_row_event["fault_confidence"] == 0.91
    assert representative_row_event["suspect_sensors"] == "M07CV08"

    component_frontend = _COMPONENT_DIR / "index.html"
    assert component_frontend.is_file(), component_frontend
    component_html = component_frontend.read_text(encoding="utf-8")
    assert "streamlit:setFrameHeight" in component_html
    assert "ResizeObserver" in component_html
    assert 'doc.querySelector(".shell")' in component_html
    assert "battery-pack:kpi-update" in component_html
    assert "replaceDocumentWithoutBlank" in component_html
    assert "frame.contentWindow.postMessage" in component_html

    synthetic_result = {
        "summary": {
            "status": "NG_REVIEW",
            "model_id": "fault-smoke",
            "model_version": "1.0",
            "mode": "DCHG",
            "fire_rate": 0.25,
            "score_p95": 0.91,
            "score_max": 0.98,
            "max_consecutive_rows": 18,
        },
        "details": {
            "window_size": 100,
            "fault_type": "temperature sensor",
            "fault_probabilities": {
                "temperature sensor": 0.92,
                "weld": 0.08,
            },
            "suspect_sensors": ["M16T02", "M16CV11"],
        },
    }
    fault_metadata = extract_fault_metadata(synthetic_result)
    assert fault_metadata["fault_type"] == "온도 센서 불량"
    assert fault_metadata["suspect_modules"] == "M16"
    assert fault_metadata["suspect_cells"] == "M16CV11"
    assert parse_probabilities(fault_metadata["fault_probabilities"])["온도 센서 불량"] == 0.92
    fault_event = build_fault_event(
        synthetic_result,
        source_file="Test09_NG_dchg.csv",
        source_frame=pd.DataFrame(
            {
                "SerialNumber": [798] * 120,
                "DATE": ["2021-11-02"] * 120,
                "TIME": [f"08:42:{second:02d}" for second in range(60)] * 2,
                "M01CV01": [4.1 - index * 0.001 for index in range(120)],
                "M01T01": [30.0 + index * 0.01 for index in range(120)],
            }
        ),
        detected_row=120,
        detected_at="2021-11-02 08:42:43",
        origin="smoke",
    )
    assert fault_event is not None
    assert "레시피 자동 로딩 및 이중 확인" in fault_event["recommended_action"]
    assert fault_event["pfmea_ng_codes"] == "NG8, NG9"
    assert fault_event["rpn"] == 96
    assert fault_event["risk_level"] == 1
    assert fault_event["risk_color"] == "노랑"
    assert fault_event["severity"] == "주의"
    assert display_mode(fault_event["mode"]) == "방전"
    assert fault_event["mode_display"] == "방전"
    assert fault_event["risk_label"] == "위험도 1"
    assert fault_event["fault_confidence_percent"] == "92.0%"
    assert fault_event["fire_rate_percent"] == "25.0%"
    assert fault_event["source_row_number"] == 120
    assert fault_event["source_window_start_row"] == 21
    assert fault_event["source_window_end_row"] == 120
    assert fault_event["source_window_row_count"] == 100
    assert fault_event["source_column_count"] == 5
    assert fault_event["raw__SerialNumber"] == 798
    assert abs(fault_event["raw__M01CV01"] - 3.981) < 1e-12
    assert len(json.loads(fault_event["source_window_json"])) == 100
    assert display_mode("CHG") == "충전"
    assert recommendation_for("용량 불량")["risk_level"] == 1
    assert recommendation_for("용량 불량")["risk_color"] == "노랑"
    assert recommendation_for("유형 분석 대기")["risk_level"] == 0
    assert recommendation_for("유형 분석 대기")["risk_color"] == "흰색"
    repeated_fault_event = build_fault_event(
        synthetic_result,
        source_file="Test09_NG_dchg.csv",
        detected_row=120,
        detected_at="2021-11-02 08:42:43",
        origin="smoke",
        occurrence_key="second-live-run",
    )
    assert repeated_fault_event is not None
    assert repeated_fault_event["event_id"] != fault_event["event_id"]
    stable_fault_event_1 = build_fault_event(
        synthetic_result,
        source_file="Test09_NG_dchg.csv",
        detected_row=120,
        detected_at="2021-11-02 08:42:43",
        origin="file-analysis",
        occurrence_key="same-file-revision",
    )
    stable_fault_event_2 = build_fault_event(
        synthetic_result,
        source_file="Test09_NG_dchg.csv",
        detected_row=120,
        detected_at="2026-07-28 16:30:00",
        origin="file-analysis",
        occurrence_key="same-file-revision",
    )
    assert stable_fault_event_1 is not None
    assert stable_fault_event_2 is not None
    assert stable_fault_event_1["event_id"] == stable_fault_event_2["event_id"]
    representative_events = representative_fault_events(
        pd.DataFrame(
            [
                {
                    "event_id": "candidate-a",
                    "source_file": "Test09_NG_dchg.csv",
                    "model_id": "candidate-model",
                    "fault_type": "고저항 불량",
                    "fault_confidence": 0.99,
                    "detected_at": "2026-07-28 15:00:00",
                },
                {
                    "event_id": "production-low",
                    "source_file": "Test09_NG_dchg.csv",
                    "model_id": "production-model",
                    "fault_type": "온도 센서 불량",
                    "fault_confidence": 0.72,
                    "detected_at": "2026-07-28 15:10:00",
                },
                {
                    "event_id": "production-high",
                    "source_file": "Test09_NG_dchg.csv",
                    "model_id": "production-model",
                    "fault_type": "온도 센서 불량",
                    "fault_confidence": 0.92,
                    "detected_at": "2026-07-28 15:20:00",
                },
                {
                    "event_id": "production-other-file",
                    "source_file": "Test08_NG_chg.csv",
                    "model_id": "production-model",
                    "fault_type": "온도 센서 불량",
                    "fault_confidence": 0.88,
                    "detected_at": "2026-07-28 15:30:00",
                },
            ]
        ),
        model_id="production-model",
    )
    assert len(representative_events) == 2
    ng9_representative = representative_events[
        representative_events["source_file"].eq("Test09_NG_dchg.csv")
    ].iloc[0]
    assert ng9_representative["event_id"] == "production-high"
    assert abs(float(ng9_representative["fault_confidence"]) - 0.92) < 1e-12
    batch_events = batch_fault_events(
        pd.DataFrame([{**synthetic_result["summary"], **fault_metadata, "file_name": "Test09_NG_dchg.csv"}]),
        batch_id="smoke-batch",
        detected_at="2021-11-02 08:42:43",
    )
    assert len(batch_events) == 1
    assert batch_events.iloc[0]["fault_type"] == "온도 센서 불량"

    settings = load_settings()
    catalog = build_catalog(settings["data_sources"])
    assert not catalog.empty, "데이터 카탈로그가 비어 있습니다."
    assert catalog["readable"].all(), "읽을 수 없는 CSV가 있습니다."

    models = {model.model_id: model for model in discover_models()}
    active_models = [model for model in models.values() if model.is_active]
    assert len(active_models) == 1, "운영 활성 모델은 정확히 1개여야 합니다."
    assert active_models[0].healthy, "운영 활성 모델 패키지가 정상 상태가 아닙니다."

    for token in ["Test02_OK_dchg", "Test09_NG_dchg"]:
        row = catalog[catalog["file_name"].str.contains(token, case=False, na=False)].iloc[0]
        frame = read_csv_resilient(Path(row["path"]))
        _, quality = audit_data_quality(frame, settings["quality_policy"])
        cv_cols, temp_cols = detect_sensor_columns(frame.columns)
        features = build_row_features(frame, row["file_name"])
        kpis = build_sensor_kpis(frame)
        assert quality["status"] == "PASS", (token, quality)
        assert len(cv_cols) == 176, (token, len(cv_cols))
        assert len(temp_cols) == 32, (token, len(temp_cols))
        assert len(features) == len(frame)
        assert {"cv_range", "temp_range", "temp_pair_gap_max"}.issubset(features.columns)
        assert {"cv_mean", "cv_std", "temp_mean", "temp_min", "temp_max", "temp_range", "temp_std"}.issubset(kpis.columns)
        assert len(sensor_snapshot_matrix(frame.iloc[0], cv_cols, "voltage").stack()) == 176
        assert len(sensor_snapshot_matrix(frame.iloc[0], temp_cols, "temperature").stack()) == 32
        workspace_html = draggable_kpi_workspace_html(frame.head(30), kpis.head(30), f"smoke::{token}")
        assert "draggable = true" in workspace_html
        assert "팩 운전 신호" in workspace_html
        assert "그래프 제거" in workspace_html
        assert "const ROW_COUNT = 3" in workspace_html
        assert "const MAX_PER_ROW = 2" in workspace_html
        assert ".row-content.single .chart-panel" in workspace_html
        assert "이미 배치된 그래프도 다시 끌어 순서를 바꿀 수 있습니다" in workspace_html
        assert "row-meta" not in workspace_html
        assert "workspace-placeholder" in workspace_html
        assert "renderWorkspace(true)" in workspace_html
        assert "let payload =" in workspace_html
        assert "function updatePayload(nextPayload)" in workspace_html
        assert 'message.type === "battery-pack:kpi-update"' in workspace_html
        assert "wide ? 1440 : 720" in workspace_html
        assert "makePanel(metric,metrics.length === 1)" in workspace_html
        assert ".remove-zone { display:none; }" in workspace_html
        assert '<span class="drag-icon">+</span>' in workspace_html
        for color in ["#0057B8", "#6A1B9A", "#00695C", "#8A5A00", "#37474F"]:
            assert color in workspace_html, f"KPI 고유 색상이 누락되었습니다: {color}"
        assert "color-mix(in srgb,var(--accent) 6%,#fff)" in workspace_html
        assert 'stroke="rgba(255,255,255,.86)"' in workspace_html
        assert '<span class="drag-icon">끌기</span>' not in workspace_html
        print(token, "DATA_QA_PASS", f"CV={len(cv_cols)}", f"TEMP={len(temp_cols)}")

    print("ACTIVE_MODELS", len(active_models), active_models[0].model_id, "PASS")


if __name__ == "__main__":
    main()
