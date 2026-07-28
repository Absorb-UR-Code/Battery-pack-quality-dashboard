from __future__ import annotations

from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile
from typing import Any

import pandas as pd

from .config import BATCH_DIR, FAULT_DIR, INBOX_DIR, REVIEW_DIR
from .n8n_webhook import send_fault_source_csv_to_n8n


def safe_file_name(name: str) -> str:
    base = Path(str(name)).name
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._ -]+", "_", base).strip(" .")
    return cleaned or "uploaded.csv"


def save_uploads(uploaded_files: list[Any]) -> list[Path]:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for uploaded in uploaded_files:
        name = safe_file_name(uploaded.name)
        payload = uploaded.getvalue()
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in archive.infolist():
                    if member.is_dir() or not member.filename.lower().endswith(".csv"):
                        continue
                    member_name = safe_file_name(member.filename)
                    target = INBOX_DIR / member_name
                    target.write_bytes(archive.read(member))
                    saved.append(target)
        elif name.lower().endswith(".csv"):
            target = INBOX_DIR / name
            target.write_bytes(payload)
            saved.append(target)
    return saved


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def review_log_path() -> Path:
    return REVIEW_DIR / "operator_review_log.csv"


def load_reviews() -> pd.DataFrame:
    path = review_log_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def append_review(record: dict[str, Any]) -> Path:
    record = {
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    existing = load_reviews()
    updated = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    path = review_log_path()
    _atomic_write_csv(updated, path)
    return path


def fault_event_log_path() -> Path:
    return FAULT_DIR / "model_fault_event_log.csv"


def fault_action_log_path() -> Path:
    return FAULT_DIR / "fault_action_log.csv"


def fault_event_delete_log_path() -> Path:
    return FAULT_DIR / "fault_event_delete_log.csv"


def load_fault_event_log() -> pd.DataFrame:
    path = fault_event_log_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def save_fault_event_log(frame: pd.DataFrame) -> Path:
    path = fault_event_log_path()
    _atomic_write_csv(frame, path)
    return path


def upsert_fault_event(record: dict[str, Any]) -> Path:
    event = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    existing = load_fault_event_log()
    event_id = str(event.get("event_id", "")).strip()
    duplicate_event = False
    if event_id and not existing.empty and "event_id" in existing.columns:
        event_mask = existing["event_id"].astype(str).eq(event_id)
        duplicate_event = bool(event_mask.any())
        if duplicate_event:
            return fault_event_log_path()

    updated = pd.concat([existing, pd.DataFrame([event])], ignore_index=True)
    path = fault_event_log_path()
    _atomic_write_csv(updated, path)

    if not duplicate_event:
        delivery = send_fault_source_csv_to_n8n(
            str(event.get("source_path", "")).strip(),
            event,
        )
        if delivery.enabled:
            delivery_fields = {
                "n8n_delivery_status": delivery.status,
                "n8n_delivery_at": delivery.delivered_at,
                "n8n_http_status": delivery.status_code,
                "n8n_delivery_error": delivery.error,
                "n8n_response_preview": delivery.response_preview,
            }
            event.update(delivery_fields)
            updated = pd.concat(
                [existing, pd.DataFrame([event])],
                ignore_index=True,
            )
            _atomic_write_csv(updated, path)

    return path


def load_fault_actions() -> pd.DataFrame:
    path = fault_action_log_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def append_fault_action(record: dict[str, Any]) -> Path:
    action = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    existing = load_fault_actions()
    updated = pd.concat([existing, pd.DataFrame([action])], ignore_index=True)
    path = fault_action_log_path()
    _atomic_write_csv(updated, path)

    event_id = str(action.get("event_id", "")).strip()
    events = load_fault_event_log()
    if event_id and not events.empty and "event_id" in events.columns:
        event_mask = events["event_id"].astype(str).eq(event_id)
        if event_mask.any():
            event_updates = {
                "action_status": action.get("action_status", ""),
                "final_action": action.get("final_action", ""),
                "assignee": action.get("assignee", ""),
                "action_notes": action.get("action_notes", ""),
                "action_updated_at": action["updated_at"],
            }
            for column, value in event_updates.items():
                if column not in events.columns:
                    events[column] = ""
                events.loc[event_mask, column] = value
            save_fault_event_log(events)
    return path


def load_deleted_fault_event_ids() -> set[str]:
    path = fault_event_delete_log_path()
    if not path.exists():
        return set()
    deleted = pd.read_csv(path, encoding="utf-8-sig")
    if "event_id" not in deleted.columns:
        return set()
    return {
        event_id
        for event_id in deleted["event_id"].dropna().astype(str).str.strip()
        if event_id
    }


def delete_fault_events(
    event_ids: list[str] | tuple[str, ...] | set[str],
    *,
    deleted_by: str = "",
    reason: str = "사용자 삭제",
) -> dict[str, Any]:
    ids = {
        str(event_id).strip()
        for event_id in event_ids
        if str(event_id).strip()
    }
    if not ids:
        return {
            "requested": 0,
            "event_rows_deleted": 0,
            "action_rows_deleted": 0,
            "delete_log_path": fault_event_delete_log_path(),
        }

    event_rows_deleted = 0
    event_path = fault_event_log_path()
    events = load_fault_event_log()
    if not events.empty and "event_id" in events.columns:
        remove_mask = events["event_id"].astype(str).isin(ids)
        event_rows_deleted = int(remove_mask.sum())
        _atomic_write_csv(events.loc[~remove_mask].copy(), event_path)

    action_rows_deleted = 0
    action_path = fault_action_log_path()
    actions = load_fault_actions()
    if not actions.empty and "event_id" in actions.columns:
        remove_mask = actions["event_id"].astype(str).isin(ids)
        action_rows_deleted = int(remove_mask.sum())
        _atomic_write_csv(actions.loc[~remove_mask].copy(), action_path)

    delete_path = fault_event_delete_log_path()
    if delete_path.exists():
        deleted = pd.read_csv(delete_path, encoding="utf-8-sig")
    else:
        deleted = pd.DataFrame()
    now = datetime.now().isoformat(timespec="seconds")
    additions = pd.DataFrame(
        [
            {
                "deleted_at": now,
                "event_id": event_id,
                "deleted_by": deleted_by,
                "reason": reason,
            }
            for event_id in sorted(ids)
        ]
    )
    updated = pd.concat([deleted, additions], ignore_index=True)
    updated = updated.drop_duplicates(subset=["event_id"], keep="last")
    _atomic_write_csv(updated, delete_path)

    return {
        "requested": len(ids),
        "event_rows_deleted": event_rows_deleted,
        "action_rows_deleted": action_rows_deleted,
        "delete_log_path": delete_path,
    }


def save_batch_result(frame: pd.DataFrame, metadata: dict[str, Any]) -> tuple[Path, Path]:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = BATCH_DIR / f"batch_result_{stamp}.csv"
    json_path = BATCH_DIR / f"batch_result_{stamp}.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")
