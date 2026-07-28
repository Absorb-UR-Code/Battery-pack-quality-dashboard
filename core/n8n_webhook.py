from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid


LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_PREVIEW = 500
HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
UPLOAD_FIELD_NAME = "fault_source_csv"


@dataclass(frozen=True)
class N8nWebhookSettings:
    enabled: bool = False
    webhook_url: str = ""
    auth_header_name: str = "X-Battery-Token"
    auth_token: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    send_raw_window: bool = True


@dataclass(frozen=True)
class N8nDeliveryResult:
    enabled: bool
    attempted: bool
    sent: bool
    status_code: int | None = None
    error: str = ""
    response_preview: str = ""
    delivered_at: str = ""

    @property
    def status(self) -> str:
        if not self.enabled:
            return "DISABLED"
        if self.sent:
            return "SENT"
        return "FAILED"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _as_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS
    return min(30.0, max(1.0, timeout))


def _streamlit_secret_section() -> dict[str, Any]:
    try:
        import streamlit as st

        section = st.secrets.get("n8n", {})
        return dict(section) if isinstance(section, Mapping) else {}
    except (FileNotFoundError, KeyError, RuntimeError, TypeError):
        return {}


def load_n8n_settings(
    *,
    secret_section: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> N8nWebhookSettings:
    values = dict(secret_section) if secret_section is not None else _streamlit_secret_section()
    env = os.environ if environ is None else environ
    env_overrides = {
        "enabled": env.get("N8N_ENABLED"),
        "webhook_url": env.get("N8N_WEBHOOK_URL"),
        "auth_header_name": env.get("N8N_AUTH_HEADER_NAME"),
        "auth_token": env.get("N8N_AUTH_TOKEN"),
        "timeout_seconds": env.get("N8N_TIMEOUT_SECONDS"),
        "send_raw_window": env.get("N8N_SEND_RAW_WINDOW"),
    }
    for key, value in env_overrides.items():
        if value is not None:
            values[key] = value

    return N8nWebhookSettings(
        enabled=_as_bool(values.get("enabled"), False),
        webhook_url=str(values.get("webhook_url", "")).strip(),
        auth_header_name=str(
            values.get("auth_header_name", "X-Battery-Token")
        ).strip(),
        auth_token=str(values.get("auth_token", "")).strip(),
        timeout_seconds=_as_timeout(values.get("timeout_seconds")),
        send_raw_window=_as_bool(values.get("send_raw_window"), True),
    )


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _validate_settings(settings: N8nWebhookSettings) -> str:
    if not settings.webhook_url:
        return "n8n webhook_url is empty"
    parsed = urlparse(settings.webhook_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "n8n webhook_url must be an absolute HTTP(S) URL"
    if not settings.auth_header_name:
        return "n8n auth_header_name is empty"
    if not HEADER_NAME_PATTERN.fullmatch(settings.auth_header_name):
        return "n8n auth_header_name is invalid"
    if not settings.auth_token:
        return "n8n auth_token is empty"
    return ""


def _csv_row_count(payload: bytes) -> int:
    text = payload.decode("utf-8-sig", errors="replace")
    rows = csv.reader(io.StringIO(text))
    try:
        next(rows)
    except StopIteration:
        return 0
    return sum(1 for _ in rows)


def build_fault_source_metadata(
    csv_path: str | Path,
    trigger_record: Mapping[str, Any],
    *,
    sent_at: str,
    payload: bytes | None = None,
) -> dict[str, Any]:
    path = Path(csv_path)
    csv_payload = payload if payload is not None else path.read_bytes()
    return {
        "schema_version": "2.1",
        "event_type": "battery_pack_fault_source_csv",
        "webhook_sent_at": sent_at,
        "trigger_event_id": str(trigger_record.get("event_id", "")).strip(),
        "trigger_source_file": str(trigger_record.get("source_file", "")).strip(),
        "trigger_serial_number": str(
            trigger_record.get("serial_number", "")
        ).strip(),
        "csv_filename": path.name,
        "csv_row_count": _csv_row_count(csv_payload),
        "csv_size_bytes": len(csv_payload),
        "csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
        "fault_type": str(trigger_record.get("fault_type", "")).strip(),
        "fault_confidence": _json_safe(trigger_record.get("fault_confidence")),
        "fault_confidence_percent": str(
            trigger_record.get("fault_confidence_percent", "")
        ).strip(),
        "risk_level": _json_safe(trigger_record.get("risk_level")),
        "risk_label": str(trigger_record.get("risk_label", "")).strip(),
        "rpn": _json_safe(trigger_record.get("rpn")),
        "suspect_sensors": str(
            trigger_record.get("suspect_sensors", "")
        ).strip(),
        "recommended_action": str(
            trigger_record.get("recommended_action", "")
        ).strip(),
    }


def _multipart_text_part(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n'
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        f"{value}\r\n"
    ).encode("utf-8")


def build_fault_source_multipart(
    csv_path: str | Path,
    trigger_record: Mapping[str, Any],
    *,
    sent_at: str,
    boundary: str | None = None,
) -> tuple[bytes, str, dict[str, Any]]:
    path = Path(csv_path)
    csv_payload = path.read_bytes()
    resolved_boundary = boundary or f"BatteryPackDashboard{uuid.uuid4().hex}"
    metadata = build_fault_source_metadata(
        path,
        trigger_record,
        sent_at=sent_at,
        payload=csv_payload,
    )
    metadata_json = json.dumps(
        _json_safe(metadata),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

    body = bytearray()
    body.extend(
        _multipart_text_part(
            resolved_boundary,
            "metadata",
            metadata_json,
        )
    )
    body.extend(
        _multipart_text_part(
            resolved_boundary,
            "event_id",
            metadata["trigger_event_id"],
        )
    )
    body.extend(
        _multipart_text_part(
            resolved_boundary,
            "row_count",
            str(metadata["csv_row_count"]),
        )
    )
    body.extend(f"--{resolved_boundary}\r\n".encode("ascii"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{UPLOAD_FIELD_NAME}"; '
            f'filename="{path.name}"\r\n'
            "Content-Type: text/csv; charset=utf-8\r\n"
            "\r\n"
        ).encode("utf-8")
    )
    body.extend(csv_payload)
    body.extend(b"\r\n")
    body.extend(f"--{resolved_boundary}--\r\n".encode("ascii"))
    content_type = f"multipart/form-data; boundary={resolved_boundary}"
    return bytes(body), content_type, metadata


def send_fault_source_csv_to_n8n(
    csv_path: str | Path,
    trigger_record: Mapping[str, Any],
    *,
    settings: N8nWebhookSettings | None = None,
) -> N8nDeliveryResult:
    resolved = settings or load_n8n_settings()
    if not resolved.enabled:
        return N8nDeliveryResult(enabled=False, attempted=False, sent=False)

    delivered_at = datetime.now().isoformat(timespec="seconds")
    configuration_error = _validate_settings(resolved)
    if configuration_error:
        LOGGER.warning("n8n fault source CSV delivery skipped: %s", configuration_error)
        return N8nDeliveryResult(
            enabled=True,
            attempted=False,
            sent=False,
            error=configuration_error,
            delivered_at=delivered_at,
        )

    path = Path(csv_path)
    if not path.is_file():
        error = f"fault source CSV does not exist: {path}"
        LOGGER.warning("n8n fault source CSV delivery skipped: %s", error)
        return N8nDeliveryResult(
            enabled=True,
            attempted=False,
            sent=False,
            error=error,
            delivered_at=delivered_at,
        )

    try:
        body, content_type, metadata = build_fault_source_multipart(
            path,
            trigger_record,
            sent_at=delivered_at,
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        LOGGER.warning("n8n fault source CSV payload build failed: %s", exc)
        return N8nDeliveryResult(
            enabled=True,
            attempted=False,
            sent=False,
            error=str(exc),
            delivered_at=delivered_at,
        )

    headers = {
        "Accept": "application/json",
        "Content-Type": content_type,
        "User-Agent": "Battery-Pack-Quality-Dashboard/1.0",
        "X-Battery-Event-Id": metadata["trigger_event_id"],
        "X-Battery-Log-SHA256": metadata["csv_sha256"],
        resolved.auth_header_name: resolved.auth_token,
    }
    request = Request(
        resolved.webhook_url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=resolved.timeout_seconds) as response:
            status_code = int(response.getcode())
            preview = response.read(MAX_RESPONSE_PREVIEW).decode(
                "utf-8",
                errors="replace",
            )
        sent = 200 <= status_code < 300
        if not sent:
            LOGGER.warning("n8n fault source CSV delivery returned HTTP %s", status_code)
        return N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=sent,
            status_code=status_code,
            error="" if sent else f"HTTP {status_code}",
            response_preview=preview,
            delivered_at=delivered_at,
        )
    except HTTPError as exc:
        preview = exc.read(MAX_RESPONSE_PREVIEW).decode("utf-8", errors="replace")
        LOGGER.warning("n8n fault source CSV delivery failed with HTTP %s", exc.code)
        return N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=False,
            status_code=int(exc.code),
            error=f"HTTP {exc.code}",
            response_preview=preview,
            delivered_at=delivered_at,
        )
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        LOGGER.warning("n8n fault source CSV delivery failed: %s", exc)
        return N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=False,
            error=str(exc),
            delivered_at=delivered_at,
        )
