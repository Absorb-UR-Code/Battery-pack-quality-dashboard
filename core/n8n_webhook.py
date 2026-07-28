from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
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


LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_PREVIEW = 500
HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
JSON_TEXT_FIELDS = {
    "fault_probabilities",
    "model_summary_json",
    "model_details_json",
    "source_columns_json",
    "source_row_json",
    "source_window_json",
}
RAW_FIELDS = {
    "source_columns_json",
    "source_row_json",
    "source_window_json",
}


@dataclass(frozen=True)
class N8nWebhookSettings:
    enabled: bool = False
    webhook_url: str = ""
    auth_header_name: str = "X-Battery-Token"
    auth_token: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    send_raw_window: bool = False


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
        send_raw_window=_as_bool(values.get("send_raw_window"), False),
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


def _decode_json_text_fields(event: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(event)
    for field in JSON_TEXT_FIELDS:
        value = decoded.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            decoded[field] = json.loads(value)
        except json.JSONDecodeError:
            pass
    return decoded


def build_n8n_payload(
    record: Mapping[str, Any],
    *,
    send_raw_window: bool,
    sent_at: str | None = None,
) -> dict[str, Any]:
    event = dict(record)
    event.pop("source_path", None)
    if not send_raw_window:
        event = {
            key: value
            for key, value in event.items()
            if key not in RAW_FIELDS and not key.startswith("raw__")
        }
    event = _decode_json_text_fields(event)
    return {
        "schema_version": "1.0",
        "event_type": "battery_pack_fault",
        "webhook_sent_at": sent_at or datetime.now().isoformat(timespec="seconds"),
        **_json_safe(event),
    }


def _validate_settings(settings: N8nWebhookSettings) -> str:
    if not settings.webhook_url:
        return "n8n webhook_url is empty"
    parsed = urlparse(settings.webhook_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "n8n webhook_url must be an absolute HTTP(S) URL"
    if settings.auth_header_name and not HEADER_NAME_PATTERN.fullmatch(
        settings.auth_header_name
    ):
        return "n8n auth_header_name is invalid"
    return ""


def send_fault_event_to_n8n(
    record: Mapping[str, Any],
    *,
    settings: N8nWebhookSettings | None = None,
) -> N8nDeliveryResult:
    resolved = settings or load_n8n_settings()
    if not resolved.enabled:
        return N8nDeliveryResult(enabled=False, attempted=False, sent=False)

    configuration_error = _validate_settings(resolved)
    delivered_at = datetime.now().isoformat(timespec="seconds")
    if configuration_error:
        LOGGER.warning("n8n delivery skipped: %s", configuration_error)
        return N8nDeliveryResult(
            enabled=True,
            attempted=False,
            sent=False,
            error=configuration_error,
            delivered_at=delivered_at,
        )

    payload = build_n8n_payload(
        record,
        send_raw_window=resolved.send_raw_window,
        sent_at=delivered_at,
    )
    body = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "Battery-Pack-Quality-Dashboard/1.0",
    }
    if resolved.auth_header_name and resolved.auth_token:
        headers[resolved.auth_header_name] = resolved.auth_token

    request = Request(
        resolved.webhook_url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=resolved.timeout_seconds) as response:
            status_code = int(response.getcode())
            preview = response.read(MAX_RESPONSE_PREVIEW).decode("utf-8", errors="replace")
        sent = 200 <= status_code < 300
        if not sent:
            LOGGER.warning("n8n delivery returned HTTP %s", status_code)
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
        LOGGER.warning("n8n delivery failed with HTTP %s", exc.code)
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
        LOGGER.warning("n8n delivery failed: %s", exc)
        return N8nDeliveryResult(
            enabled=True,
            attempted=True,
            sent=False,
            error=str(exc),
            delivered_at=delivered_at,
        )
