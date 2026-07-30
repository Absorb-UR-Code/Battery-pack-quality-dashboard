from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.fault_log import build_completed_fault_event
from core.model_registry import discover_models, score_dataframe
from core.n8n_webhook import N8nWebhookSettings, send_fault_source_csv_to_n8n


class _ArchiveAcknowledgementHandler(BaseHTTPRequestHandler):
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
        self.wfile.write(b'{"ok":true,"archived":true}')

    def log_message(self, format: str, *args: object) -> None:
        return


class Ng8CompletedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = ROOT / "data" / "demo" / "test" / "Test08_NG_chg.csv"
        if not cls.source_path.exists():
            raise unittest.SkipTest(f"NG8 demo file not found: {cls.source_path}")

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

    def test_ng8_is_logged_only_after_completion_and_uploads_full_csv(self) -> None:
        total_rows = len(self.source_frame)
        before_completion = build_completed_fault_event(
            self.result,
            position=total_rows - 1,
            total_rows=total_rows,
            source_file=self.source_path.name,
            source_path=str(self.source_path),
            source_frame=self.source_frame,
            mode="CHG",
            occurrence_key="ng8-before-completion",
        )
        self.assertIsNone(before_completion)

        event = build_completed_fault_event(
            self.result,
            position=total_rows,
            total_rows=total_rows,
            source_file=self.source_path.name,
            source_path=str(self.source_path),
            source_frame=self.source_frame,
            mode="CHG",
            occurrence_key="ng8-completed",
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["fault_type"], "온도 센서 불량")
        self.assertGreater(float(event["fault_confidence"]), 0.0)
        self.assertTrue(str(event["suspect_sensors"]).strip())

        _ArchiveAcknowledgementHandler.requests = []
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _ArchiveAcknowledgementHandler,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = send_fault_source_csv_to_n8n(
                self.source_path,
                event,
                settings=N8nWebhookSettings(
                    enabled=True,
                    webhook_url=(
                        f"http://127.0.0.1:{server.server_port}/battery-pack-fault"
                    ),
                    auth_header_name="X-Battery-Token",
                    auth_token="integration-test-token",
                    timeout_seconds=5,
                ),
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertTrue(result.sent)
        self.assertEqual(len(_ArchiveAcknowledgementHandler.requests), 1)
        request = _ArchiveAcknowledgementHandler.requests[0]
        self.assertEqual(request["path"], "/battery-pack-fault")
        self.assertIn(self.source_path.read_bytes(), request["body"])
        self.assertIn("온도 센서 불량".encode("utf-8"), request["body"])


if __name__ == "__main__":
    unittest.main()
