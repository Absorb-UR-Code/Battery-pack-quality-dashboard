from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.n8n_webhook import (
    N8nDeliveryResult,
    N8nWebhookSettings,
    build_n8n_payload,
    send_fault_event_to_n8n,
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
                "json": json.loads(body.decode("utf-8")),
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"received":true}')

    def log_message(self, format: str, *args: object) -> None:
        return


class N8nWebhookTests(unittest.TestCase):
    def test_payload_excludes_local_path_and_optional_raw_data(self) -> None:
        record = {
            "event_id": "fault-001",
            "source_path": r"C:\private\input.csv",
            "fault_probabilities": '{"temperature": 0.91}',
            "source_window_json": '[{"M01CV01": 4.1}]',
            "raw__M01CV01": 4.1,
        }

        compact = build_n8n_payload(
            record,
            send_raw_window=False,
            sent_at="2026-07-28T12:00:00",
        )
        self.assertNotIn("source_path", compact)
        self.assertNotIn("source_window_json", compact)
        self.assertNotIn("raw__M01CV01", compact)
        self.assertEqual(compact["fault_probabilities"], {"temperature": 0.91})

        full = build_n8n_payload(
            record,
            send_raw_window=True,
            sent_at="2026-07-28T12:00:00",
        )
        self.assertEqual(full["source_window_json"], [{"M01CV01": 4.1}])
        self.assertEqual(full["raw__M01CV01"], 4.1)
        self.assertNotIn("source_path", full)

    def test_successful_post_uses_configured_auth_header(self) -> None:
        _CaptureHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = N8nWebhookSettings(
                enabled=True,
                webhook_url=f"http://127.0.0.1:{server.server_port}/battery-fault",
                auth_header_name="X-Battery-Token",
                auth_token="test-token",
                timeout_seconds=2,
                send_raw_window=False,
            )
            result = send_fault_event_to_n8n(
                {"event_id": "fault-002", "serial_number": "1009"},
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
        self.assertEqual(captured["json"]["event_id"], "fault-002")

    def test_invalid_url_fails_without_raising(self) -> None:
        result = send_fault_event_to_n8n(
            {"event_id": "fault-003"},
            settings=N8nWebhookSettings(enabled=True, webhook_url="not-a-url"),
        )
        self.assertFalse(result.sent)
        self.assertFalse(result.attempted)
        self.assertIn("absolute HTTP", result.error)

    def test_upsert_sends_each_event_id_only_once(self) -> None:
        delivery = N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=True,
            status_code=200,
            response_preview='{"received":true}',
            delivered_at="2026-07-28T12:30:00",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(storage, "FAULT_DIR", Path(temp_dir)),
                patch.object(
                    storage,
                    "send_fault_event_to_n8n",
                    return_value=delivery,
                ) as sender,
            ):
                storage.upsert_fault_event(
                    {"event_id": "fault-004", "serial_number": "1009"}
                )
                storage.upsert_fault_event(
                    {"event_id": "fault-004", "serial_number": "1009"}
                )
                saved = storage.load_fault_event_log()

        self.assertEqual(sender.call_count, 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "n8n_delivery_status"], "SENT")
        self.assertEqual(int(saved.loc[0, "n8n_http_status"]), 200)

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
                    "send_fault_event_to_n8n",
                    return_value=delivery,
                ),
            ):
                path = storage.upsert_fault_event(
                    {"event_id": "fault-005", "serial_number": "1009"}
                )
                saved = storage.load_fault_event_log()

        self.assertTrue(path.name.endswith(".csv"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "n8n_delivery_status"], "FAILED")
        self.assertEqual(saved.loc[0, "n8n_delivery_error"], "timed out")


if __name__ == "__main__":
    unittest.main()
