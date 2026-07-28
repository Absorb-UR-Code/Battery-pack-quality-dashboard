from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.n8n_webhook import (
    N8nDeliveryResult,
    N8nWebhookSettings,
    build_fault_log_multipart,
    send_fault_log_csv_to_n8n,
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


class N8nWebhookTests(unittest.TestCase):
    @staticmethod
    def _write_fault_log(path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def test_multipart_contains_complete_csv_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "model_fault_event_log.csv"
            rows = [
                {"event_id": "fault-001", "serial_number": "1009", "status": "NG"},
                {"event_id": "fault-002", "serial_number": "1017", "status": "NG"},
            ]
            self._write_fault_log(csv_path, rows)
            csv_bytes = csv_path.read_bytes()

            body, content_type, metadata = build_fault_log_multipart(
                csv_path,
                {"event_id": "fault-002", "source_file": "1017_chg.csv"},
                sent_at="2026-07-28T12:00:00",
                boundary="TestBoundary",
            )

        self.assertEqual(
            content_type,
            "multipart/form-data; boundary=TestBoundary",
        )
        self.assertEqual(metadata["csv_row_count"], 2)
        self.assertEqual(metadata["trigger_event_id"], "fault-002")
        self.assertEqual(metadata["csv_size_bytes"], len(csv_bytes))
        self.assertIn(
            b'name="fault_log_csv"; filename="model_fault_event_log.csv"',
            body,
        )
        self.assertIn(csv_bytes, body)
        self.assertIn(b'"csv_row_count":2', body)
        self.assertIn(b'name="event_id"', body)

    def test_successful_post_uploads_csv_with_auth_header(self) -> None:
        _CaptureHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = Path(temp_dir) / "model_fault_event_log.csv"
                self._write_fault_log(
                    csv_path,
                    [{"event_id": "fault-003", "serial_number": "1009"}],
                )
                settings = N8nWebhookSettings(
                    enabled=True,
                    webhook_url=(
                        f"http://127.0.0.1:{server.server_port}/battery-fault"
                    ),
                    auth_header_name="X-Battery-Token",
                    auth_token="test-token",
                    timeout_seconds=2,
                )
                result = send_fault_log_csv_to_n8n(
                    csv_path,
                    {"event_id": "fault-003", "serial_number": "1009"},
                    settings=settings,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result.sent)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(_CaptureHandler.requests), 1)
        captured = _CaptureHandler.requests[0]
        self.assertEqual(captured["path"], "/battery-fault")
        self.assertEqual(captured["headers"]["X-Battery-Token"], "test-token")
        self.assertEqual(captured["headers"]["X-Battery-Event-Id"], "fault-003")
        self.assertTrue(
            captured["headers"]["Content-Type"].startswith("multipart/form-data;")
        )
        self.assertIn(b"fault-003", captured["body"])
        self.assertIn(b"model_fault_event_log.csv", captured["body"])

    def test_invalid_url_fails_without_raising(self) -> None:
        result = send_fault_log_csv_to_n8n(
            "missing.csv",
            {"event_id": "fault-004"},
            settings=N8nWebhookSettings(
                enabled=True,
                webhook_url="not-a-url",
                auth_token="test-token",
            ),
        )
        self.assertFalse(result.sent)
        self.assertFalse(result.attempted)
        self.assertIn("absolute HTTP", result.error)

    def test_upsert_sends_growing_snapshot_and_skips_duplicate_id(self) -> None:
        delivery = N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=True,
            status_code=200,
            response_preview='{"received":true}',
            delivered_at="2026-07-28T12:30:00",
        )
        snapshots: list[tuple[str, int, set[str]]] = []

        def capture_snapshot(path: Path, event: dict[str, object]) -> N8nDeliveryResult:
            frame = pd.read_csv(path, encoding="utf-8-sig")
            snapshots.append(
                (
                    str(event["event_id"]),
                    len(frame),
                    set(frame["event_id"].astype(str)),
                )
            )
            return delivery

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(storage, "FAULT_DIR", Path(temp_dir)),
                patch.object(
                    storage,
                    "send_fault_log_csv_to_n8n",
                    side_effect=capture_snapshot,
                ) as sender,
            ):
                storage.upsert_fault_event(
                    {"event_id": "fault-005", "serial_number": "1009"}
                )
                storage.upsert_fault_event(
                    {"event_id": "fault-006", "serial_number": "1017"}
                )
                storage.upsert_fault_event(
                    {"event_id": "fault-006", "serial_number": "1017"}
                )
                saved = storage.load_fault_event_log()

        self.assertEqual(sender.call_count, 2)
        self.assertEqual(snapshots[0], ("fault-005", 1, {"fault-005"}))
        self.assertEqual(
            snapshots[1],
            ("fault-006", 2, {"fault-005", "fault-006"}),
        )
        self.assertEqual(len(saved), 2)
        self.assertEqual(set(saved["n8n_delivery_status"]), {"SENT"})

    def test_delivery_failure_does_not_prevent_csv_storage(self) -> None:
        delivery = N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=False,
            error="timed out",
            delivered_at="2026-07-28T12:40:00",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(storage, "FAULT_DIR", Path(temp_dir)),
                patch.object(
                    storage,
                    "send_fault_log_csv_to_n8n",
                    return_value=delivery,
                ),
            ):
                path = storage.upsert_fault_event(
                    {"event_id": "fault-007", "serial_number": "1009"}
                )
                saved = storage.load_fault_event_log()

        self.assertTrue(path.name.endswith(".csv"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "n8n_delivery_status"], "FAILED")
        self.assertEqual(saved.loc[0, "n8n_delivery_error"], "timed out")


if __name__ == "__main__":
    unittest.main()
