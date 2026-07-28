from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.diagnostic_display import evaluated_row_positions
from core.fault_log import build_completed_fault_event
from core.model_registry import discover_models, score_dataframe
from core.n8n_webhook import (
    N8nWebhookSettings,
    send_fault_source_csv_to_n8n,
)
from core import storage


class _CaptureHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body,
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"received":true}')

    def log_message(self, format: str, *args: object) -> None:
        return


class Ng6CompletedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = ROOT / "data" / "demo" / "test" / "Test06_NG_chg.csv"
        if not cls.source_path.exists():
            raise unittest.SkipTest(f"NG6 demo file not found: {cls.source_path}")

        specs = discover_models(ROOT / "models")
        cls.spec = next(
            (
                spec
                for spec in specs
                if spec.model_id == "lstm_two_stage_quality_v1"
            ),
            None,
        )
        if cls.spec is None or not cls.spec.healthy:
            raise unittest.SkipTest("Production LSTM model is unavailable.")

        cls.source_frame = pd.read_csv(cls.source_path)
        cls.result = score_dataframe(
            cls.spec,
            cls.source_frame,
            cls.source_path.name,
        )

    def test_ng6_delete_reset_rerun_recreates_log_and_resends_full_csv(self) -> None:
        total_rows = len(self.source_frame)
        evaluated = evaluated_row_positions(
            self.result["row_result"],
            total_rows,
            end_position=total_rows,
        )
        self.assertGreater(len(evaluated), 0)
        self.assertEqual(int(evaluated[-1]), total_rows - 1)

        before_completion = build_completed_fault_event(
            self.result,
            position=total_rows - 1,
            total_rows=total_rows,
            source_file=self.source_path.name,
            source_path=str(self.source_path),
            source_frame=self.source_frame,
            mode="CHG",
            occurrence_key="ng6-run-before-completion",
        )
        self.assertIsNone(before_completion)

        first_event = build_completed_fault_event(
            self.result,
            position=total_rows,
            total_rows=total_rows,
            source_file=self.source_path.name,
            source_path=str(self.source_path),
            source_frame=self.source_frame,
            mode="CHG",
            occurrence_key="ng6-run-1",
        )
        second_event = build_completed_fault_event(
            self.result,
            position=total_rows,
            total_rows=total_rows,
            source_file=self.source_path.name,
            source_path=str(self.source_path),
            source_frame=self.source_frame,
            mode="CHG",
            occurrence_key="ng6-run-2",
        )
        self.assertIsNotNone(first_event)
        self.assertIsNotNone(second_event)
        self.assertNotEqual(first_event["event_id"], second_event["event_id"])

        for event in (first_event, second_event):
            self.assertEqual(event["fault_type"], "용접·접촉 불량")
            self.assertGreater(float(event["fault_confidence"]), 0.0)
            self.assertGreaterEqual(int(event["risk_level"]), 0)
            self.assertGreater(int(event["rpn"]), 0)
            self.assertTrue(str(event["suspect_sensors"]).strip())
            self.assertTrue(str(event["recommended_action"]).strip())

        _CaptureHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        settings = N8nWebhookSettings(
            enabled=True,
            webhook_url=f"http://127.0.0.1:{server.server_port}/battery-pack-fault",
            auth_header_name="X-Battery-Token",
            auth_token="integration-test-token",
            timeout_seconds=5,
        )

        def send_to_capture(path: str | Path, event: dict[str, object]):
            return send_fault_source_csv_to_n8n(
                path,
                event,
                settings=settings,
            )

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with (
                    patch.object(storage, "FAULT_DIR", Path(temp_dir) / "fault"),
                    patch.object(
                        storage,
                        "send_fault_source_csv_to_n8n",
                        side_effect=send_to_capture,
                    ),
                ):
                    storage.upsert_fault_event(first_event)
                    first_saved = storage.load_fault_event_log()
                    self.assertEqual(len(first_saved), 1)
                    self.assertEqual(
                        first_saved.loc[0, "n8n_delivery_status"],
                        "SENT",
                    )

                    deleted = storage.delete_fault_events(
                        [str(first_event["event_id"])],
                        reason="NG6 재판정 통합 검증",
                    )
                    self.assertEqual(deleted["event_rows_deleted"], 1)
                    self.assertTrue(storage.load_fault_event_log().empty)

                    storage.upsert_fault_event(second_event)
                    second_saved = storage.load_fault_event_log()
                    self.assertEqual(len(second_saved), 1)
                    self.assertEqual(
                        second_saved.loc[0, "event_id"],
                        second_event["event_id"],
                    )
                    self.assertEqual(
                        second_saved.loc[0, "n8n_delivery_status"],
                        "SENT",
                    )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(len(_CaptureHandler.requests), 2)
        source_bytes = self.source_path.read_bytes()
        required_metadata = [
            "용접·접촉 불량",
            str(first_event["risk_label"]),
            str(first_event["rpn"]),
            str(first_event["suspect_sensors"]),
            str(first_event["recommended_action"]),
        ]
        for request in _CaptureHandler.requests:
            self.assertEqual(request["path"], "/battery-pack-fault")
            self.assertEqual(
                request["headers"]["X-Battery-Token"],
                "integration-test-token",
            )
            body = request["body"]
            self.assertIn(source_bytes, body)
            for value in required_metadata:
                self.assertIn(value.encode("utf-8"), body)


if __name__ == "__main__":
    unittest.main()
