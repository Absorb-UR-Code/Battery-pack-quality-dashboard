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
    build_fault_source_multipart,
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


class _EmptyAcknowledgementHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class N8nWebhookTests(unittest.TestCase):
    @staticmethod
    def _write_fault_log(path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def test_multipart_contains_complete_fault_source_csv_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "Test09_NG_dchg.csv"
            rows = [
                {"event_id": "fault-001", "serial_number": "1009", "status": "NG"},
                {"event_id": "fault-002", "serial_number": "1017", "status": "NG"},
            ]
            self._write_fault_log(csv_path, rows)
            csv_bytes = csv_path.read_bytes()

            body, content_type, metadata = build_fault_source_multipart(
                csv_path,
                {
                    "event_id": "fault-002",
                    "source_file": "1017_chg.csv",
                    "serial_number": "798",
                    "mode": "충전",
                    "fault_type": "용접·접촉 불량",
                    "fault_confidence": 0.873,
                    "fault_confidence_percent": "87.3%",
                    "risk_level": 2,
                    "risk_label": "위험도 2",
                    "rpn": 168,
                    "suspect_sensors": "M07CV02, M07CV08",
                    "recommended_action": "접촉부 재검사 및 격리",
                    "model_name": "배터리팩 LSTM 모델",
                    "model_version": "1.0.0",
                    "action_status": "현장 검토 중",
                    "owner": "김준영",
                },
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
        self.assertEqual(metadata["schema_version"], "2.1")
        self.assertEqual(metadata["fault_type"], "용접·접촉 불량")
        self.assertEqual(metadata["fault_confidence"], 0.873)
        self.assertEqual(metadata["risk_level"], 2)
        self.assertEqual(metadata["risk_label"], "위험도 2")
        self.assertEqual(metadata["rpn"], 168)
        self.assertEqual(metadata["suspect_sensors"], "M07CV02, M07CV08")
        self.assertEqual(metadata["recommended_action"], "접촉부 재검사 및 격리")
        self.assertEqual(metadata["mode"], "충전")
        self.assertEqual(metadata["model_name"], "배터리팩 LSTM 모델")
        self.assertEqual(metadata["model_version"], "1.0.0")
        self.assertEqual(metadata["action_status"], "현장 검토 중")
        self.assertEqual(metadata["owner"], "김준영")
        self.assertIn(
            b'name="fault_source_csv"; filename="Test09_NG_dchg.csv"',
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
                csv_path = Path(temp_dir) / "Test09_NG_dchg.csv"
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
                result = send_fault_source_csv_to_n8n(
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
        self.assertIn(b"Test09_NG_dchg.csv", captured["body"])

    def test_empty_http_200_is_not_treated_as_archived(self) -> None:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _EmptyAcknowledgementHandler,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = Path(temp_dir) / "Test08_NG_chg.csv"
                self._write_fault_log(
                    csv_path,
                    [{"event_id": "fault-ng8", "serial_number": "798"}],
                )
                result = send_fault_source_csv_to_n8n(
                    csv_path,
                    {"event_id": "fault-ng8", "serial_number": "798"},
                    settings=N8nWebhookSettings(
                        enabled=True,
                        webhook_url=(
                            f"http://127.0.0.1:{server.server_port}/battery-fault"
                        ),
                        auth_header_name="X-Battery-Token",
                        auth_token="test-token",
                        timeout_seconds=2,
                    ),
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertTrue(result.attempted)
        self.assertFalse(result.sent)
        self.assertEqual(result.status_code, 200)
        self.assertIn("without a JSON acknowledgement", result.error)

    def test_invalid_url_fails_without_raising(self) -> None:
        result = send_fault_source_csv_to_n8n(
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

    def test_upsert_sends_each_fault_source_file_and_skips_duplicate_id(self) -> None:
        delivery = N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=True,
            status_code=200,
            response_preview='{"received":true}',
            delivered_at="2026-07-28T12:30:00",
        )
        uploads: list[tuple[str, str, int, set[str]]] = []

        def capture_snapshot(path: Path, event: dict[str, object]) -> N8nDeliveryResult:
            frame = pd.read_csv(path, encoding="utf-8-sig")
            uploads.append(
                (
                    str(event["event_id"]),
                    Path(path).name,
                    len(frame),
                    set(frame.columns),
                )
            )
            return delivery

        with tempfile.TemporaryDirectory() as temp_dir:
            source_1 = Path(temp_dir) / "Test09_NG_dchg.csv"
            source_2 = Path(temp_dir) / "Test08_NG_chg.csv"
            self._write_fault_log(
                source_1,
                [
                    {"order": 1, "M01CV01": 4.01},
                    {"order": 2, "M01CV01": 3.99},
                    {"order": 3, "M01CV01": 3.97},
                ],
            )
            self._write_fault_log(
                source_2,
                [
                    {"order": 1, "M16T02": 35.1},
                    {"order": 2, "M16T02": 39.2},
                ],
            )
            with (
                patch.object(storage, "FAULT_DIR", Path(temp_dir)),
                patch.object(
                    storage,
                    "_atomic_write_csv",
                    wraps=storage._atomic_write_csv,
                ) as writer,
                patch.object(
                    storage,
                    "send_fault_source_csv_to_n8n",
                    side_effect=capture_snapshot,
                ) as sender,
            ):
                storage.upsert_fault_event(
                    {
                        "event_id": "fault-005",
                        "serial_number": "1009",
                        "source_path": str(source_1),
                    }
                )
                storage.upsert_fault_event(
                    {
                        "event_id": "fault-006",
                        "serial_number": "1017",
                        "source_path": str(source_2),
                    }
                )
                storage.upsert_fault_event(
                    {
                        "event_id": "fault-006",
                        "serial_number": "1017",
                        "source_path": str(source_2),
                    }
                )
                saved = storage.load_fault_event_log()

        self.assertEqual(sender.call_count, 2)
        self.assertEqual(
            uploads[0],
            ("fault-005", "Test09_NG_dchg.csv", 3, {"order", "M01CV01"}),
        )
        self.assertEqual(
            uploads[1],
            ("fault-006", "Test08_NG_chg.csv", 2, {"order", "M16T02"}),
        )
        self.assertEqual(len(saved), 2)
        self.assertEqual(set(saved["n8n_delivery_status"]), {"SENT"})
        self.assertEqual(writer.call_count, 4)

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
                    "send_fault_source_csv_to_n8n",
                    return_value=delivery,
                ),
            ):
                path = storage.upsert_fault_event(
                    {
                        "event_id": "fault-007",
                        "serial_number": "1009",
                        "source_path": str(Path(temp_dir) / "missing.csv"),
                    }
                )
                saved = storage.load_fault_event_log()

        self.assertTrue(path.name.endswith(".csv"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "n8n_delivery_status"], "FAILED")
        self.assertEqual(saved.loc[0, "n8n_delivery_error"], "timed out")

    def test_deleted_fault_can_be_recreated_and_uploaded_again(self) -> None:
        delivery = N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=True,
            status_code=200,
            response_preview='{"received":true}',
            delivered_at="2026-07-28T13:00:00",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            fault_dir = Path(temp_dir) / "fault"
            source_path = Path(temp_dir) / "Test06_NG_chg.csv"
            self._write_fault_log(
                source_path,
                [
                    {"order": 1, "M07CV02": 4.08},
                    {"order": 2, "M07CV02": 3.92},
                ],
            )
            record = {
                "event_id": "ng6-repeatable-event",
                "source_file": source_path.name,
                "source_path": str(source_path),
                "fault_type": "용접·접촉 불량",
                "fault_confidence": 0.91,
                "risk_level": 1,
                "risk_label": "위험도 1",
                "rpn": 84,
                "suspect_sensors": "M07CV02",
                "recommended_action": "접촉부 재검사",
            }
            with (
                patch.object(storage, "FAULT_DIR", fault_dir),
                patch.object(
                    storage,
                    "send_fault_source_csv_to_n8n",
                    return_value=delivery,
                ) as sender,
            ):
                storage.upsert_fault_event(record)
                first_saved = storage.load_fault_event_log()
                storage.delete_fault_events([record["event_id"]])
                self.assertIn(
                    record["event_id"],
                    storage.load_deleted_fault_event_ids(),
                )

                storage.upsert_fault_event(record)
                recreated = storage.load_fault_event_log()
                deleted_after_recreate = storage.load_deleted_fault_event_ids()

        self.assertEqual(len(first_saved), 1)
        self.assertEqual(len(recreated), 1)
        self.assertEqual(recreated.loc[0, "event_id"], record["event_id"])
        self.assertNotIn(record["event_id"], deleted_after_recreate)
        self.assertEqual(sender.call_count, 2)


if __name__ == "__main__":
    unittest.main()
