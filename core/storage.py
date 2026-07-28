from __future__ import annotations

from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
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
    payload = dict(record)
    payload["review_id"] = str(payload.get("review_id") or uuid.uuid4().hex)
    record = {
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    existing = load_reviews()
    updated = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    path = review_log_path()
    _atomic_write_csv(updated, path)
    return path


def delete_reviews(row_indices: list[int] | tuple[int, ...] | set[int]) -> dict[str, Any]:
    reviews = load_reviews()
    requested = {
        int(index)
        for index in row_indices
        if isinstance(index, int) or str(index).strip().lstrip("-").isdigit()
    }
    valid = sorted(index for index in requested if 0 <= index < len(reviews))
    path = review_log_path()
    affected_serials: set[str] = set()
    if valid:
        deleted_reviews = reviews.loc[valid].copy()
        affected_serials = {
            serial
            for _, row in deleted_reviews.iterrows()
            if (serial := _review_serial_number(row))
        }
        updated = reviews.drop(index=valid).reset_index(drop=True)
        _atomic_write_csv(updated, path)
        for serial in affected_serials:
            reconcile_human_review_for_serial(serial, reviews=updated)
    return {
        "requested": len(requested),
        "deleted": len(valid),
        "reconciled_serials": len(affected_serials),
        "path": path,
    }


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


def _normalize_serial_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "<na>"}:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def _fault_event_serial_number(row: pd.Series) -> str:
    for column in (
        "serial_number",
        "raw__SerialNumber",
        "raw__Serial_Number",
        "raw__serial_number",
        "raw__serial",
    ):
        serial = _normalize_serial_number(row.get(column))
        if serial:
            return serial

    source_row_json = row.get("source_row_json")
    if isinstance(source_row_json, str) and source_row_json.strip():
        try:
            source_row = json.loads(source_row_json)
        except (json.JSONDecodeError, TypeError):
            source_row = {}
        if isinstance(source_row, dict):
            for key in ("SerialNumber", "Serial_Number", "serial_number", "serial"):
                serial = _normalize_serial_number(source_row.get(key))
                if serial:
                    return serial
    return ""


def fault_event_serial_numbers(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(dtype="string")
    return events.apply(_fault_event_serial_number, axis=1).astype("string")


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

    if event_id:
        restore_deleted_fault_event(event_id)

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
                    events[column] = pd.Series("", index=events.index, dtype="string")
                else:
                    # Empty CSV text columns are inferred as float64 by
                    # read_csv. Convert them before assigning operator text.
                    events[column] = events[column].astype("string").fillna("")
                events.loc[event_mask, column] = "" if value is None else str(value)
            save_fault_event_log(events)
    return path


def delete_fault_actions(
    row_indices: list[int] | tuple[int, ...] | set[int],
) -> dict[str, Any]:
    actions = load_fault_actions()
    requested = {
        int(index)
        for index in row_indices
        if isinstance(index, int) or str(index).strip().lstrip("-").isdigit()
    }
    valid = sorted(index for index in requested if 0 <= index < len(actions))
    path = fault_action_log_path()
    if not valid:
        return {
            "requested": len(requested),
            "deleted": 0,
            "path": path,
        }

    if "event_id" in actions.columns:
        affected_event_ids = {
            str(event_id).strip()
            for event_id in actions.loc[valid, "event_id"].fillna("").astype(str)
            if str(event_id).strip()
        }
    else:
        affected_event_ids = set()

    updated_actions = actions.drop(index=valid).reset_index(drop=True)
    _atomic_write_csv(updated_actions, path)

    events = load_fault_event_log()
    if affected_event_ids and not events.empty and "event_id" in events.columns:
        event_columns = {
            "action_status": "현장 검토 중",
            "final_action": "미결정",
            "assignee": "",
            "action_notes": "",
            "action_updated_at": "",
        }
        for column in event_columns:
            if column not in events.columns:
                events[column] = pd.Series("", index=events.index, dtype="string")
            else:
                events[column] = events[column].astype("string").fillna("")

        for event_id in affected_event_ids:
            event_mask = events["event_id"].fillna("").astype(str).eq(event_id)
            if not event_mask.any():
                continue
            if "event_id" in updated_actions.columns:
                remaining = updated_actions[
                    updated_actions["event_id"].fillna("").astype(str).eq(event_id)
                ].copy()
            else:
                remaining = pd.DataFrame()

            if remaining.empty:
                values = event_columns
            else:
                if "updated_at" in remaining.columns:
                    remaining = remaining.sort_values(
                        "updated_at",
                        kind="stable",
                    )
                latest = remaining.iloc[-1]
                values = {
                    "action_status": latest.get("action_status", "현장 검토 중"),
                    "final_action": latest.get("final_action", "미결정"),
                    "assignee": latest.get("assignee", ""),
                    "action_notes": latest.get("action_notes", ""),
                    "action_updated_at": latest.get("updated_at", ""),
                }

            for column, value in values.items():
                events.loc[event_mask, column] = "" if pd.isna(value) else str(value)
        save_fault_event_log(events)

    return {
        "requested": len(requested),
        "deleted": len(valid),
        "path": path,
    }


def load_deleted_fault_event_ids() -> set[str]:
    path = fault_event_delete_log_path()
    if not path.exists():
        return set()
    deleted = pd.read_csv(path, encoding="utf-8-sig")
    if "event_id" not in deleted.columns:
        return set()
    if "delete_state" not in deleted.columns:
        deleted["delete_state"] = "DELETED"
    latest = deleted.drop_duplicates(subset=["event_id"], keep="last")
    return {
        event_id
        for event_id, state in zip(
            latest["event_id"].fillna("").astype(str).str.strip(),
            latest["delete_state"].fillna("DELETED").astype(str).str.upper(),
        )
        if event_id and state != "RESTORED"
    }


def restore_deleted_fault_event(
    event_id: str,
    *,
    reason: str = "재판정으로 불량 로그 재생성",
) -> bool:
    """Reactivate an event ID that was previously hidden by a user deletion."""
    normalized_id = str(event_id).strip()
    path = fault_event_delete_log_path()
    if not normalized_id or not path.exists():
        return False

    history = pd.read_csv(path, encoding="utf-8-sig")
    if "event_id" not in history.columns:
        return False
    if normalized_id not in set(history["event_id"].fillna("").astype(str).str.strip()):
        return False

    restored = pd.DataFrame(
        [
            {
                "deleted_at": "",
                "restored_at": datetime.now().isoformat(timespec="seconds"),
                "event_id": normalized_id,
                "deleted_by": "",
                "reason": reason,
                "delete_state": "RESTORED",
            }
        ]
    )
    _atomic_write_csv(pd.concat([history, restored], ignore_index=True), path)
    return True


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
                "restored_at": "",
                "event_id": event_id,
                "deleted_by": deleted_by,
                "reason": reason,
                "delete_state": "DELETED",
            }
            for event_id in sorted(ids)
        ]
    )
    updated = pd.concat([deleted, additions], ignore_index=True)
    _atomic_write_csv(updated, delete_path)

    return {
        "requested": len(ids),
        "event_rows_deleted": event_rows_deleted,
        "action_rows_deleted": action_rows_deleted,
        "delete_log_path": delete_path,
    }


def _review_serial_number(row: pd.Series) -> str:
    for column in ("serial_number", "lot_id"):
        serial = _normalize_serial_number(row.get(column))
        if serial:
            return serial
    return ""


def _append_fault_delete_states(
    event_ids: list[str],
    *,
    state: str,
    serial_number: str,
    review_id: str,
    reviewer: str,
    reason: str,
) -> None:
    normalized_ids = sorted(
        {
            str(event_id).strip()
            for event_id in event_ids
            if str(event_id).strip()
        }
    )
    if not normalized_ids:
        return
    path = fault_event_delete_log_path()
    history = (
        pd.read_csv(path, encoding="utf-8-sig")
        if path.exists()
        else pd.DataFrame()
    )
    now = datetime.now().isoformat(timespec="seconds")
    normalized_state = str(state).strip().upper()
    additions = pd.DataFrame(
        [
            {
                "deleted_at": now if normalized_state == "DELETED" else "",
                "restored_at": now if normalized_state == "RESTORED" else "",
                "event_id": event_id,
                "deleted_by": str(reviewer).strip(),
                "reason": reason,
                "delete_state": normalized_state,
                "deletion_origin": "HUMAN_REVIEW",
                "serial_number": serial_number,
                "review_id": review_id,
            }
            for event_id in normalized_ids
        ]
    )
    _atomic_write_csv(pd.concat([history, additions], ignore_index=True), path)


def _restore_human_review_suppressions(serial_number: str) -> int:
    path = fault_event_delete_log_path()
    if not path.exists():
        return 0
    history = pd.read_csv(path, encoding="utf-8-sig")
    required = {"event_id", "deletion_origin", "serial_number"}
    if history.empty or not required.issubset(history.columns):
        return 0
    serials = history["serial_number"].map(_normalize_serial_number)
    review_rows = history[
        history["deletion_origin"].fillna("").astype(str).eq("HUMAN_REVIEW")
        & serials.eq(serial_number)
    ]
    event_ids = sorted(
        {
            str(event_id).strip()
            for event_id in review_rows["event_id"].fillna("").astype(str)
            if str(event_id).strip()
        }
    )
    _append_fault_delete_states(
        event_ids,
        state="RESTORED",
        serial_number=serial_number,
        review_id="RECONCILE",
        reviewer="",
        reason=f"검토 기록 재계산 · Serial Num {serial_number}",
    )
    return len(event_ids)


def _remove_human_review_actions(serial_number: str) -> int:
    actions = load_fault_actions()
    required = {"action_origin", "serial_number"}
    if actions.empty or not required.issubset(actions.columns):
        return 0
    action_serials = actions["serial_number"].map(_normalize_serial_number)
    remove_mask = (
        actions["action_origin"].fillna("").astype(str).eq("HUMAN_REVIEW")
        & action_serials.eq(serial_number)
    )
    indices = [int(index) for index in actions.index[remove_mask]]
    if not indices:
        return 0
    return int(delete_fault_actions(indices)["deleted"])


def _clear_human_review_effects(serial_number: str) -> dict[str, int]:
    return {
        "restored": _restore_human_review_suppressions(serial_number),
        "actions_removed": _remove_human_review_actions(serial_number),
    }


def _apply_human_review_state(
    serial_number: str,
    human_label: str,
    *,
    reviewer: str,
    notes: str,
    review_id: str,
) -> dict[str, Any]:
    serial = _normalize_serial_number(serial_number)
    label = str(human_label).strip().upper()
    empty_result = {
        "serial_number": serial,
        "human_label": label,
        "matched": 0,
        "deleted": 0,
        "updated": 0,
        "restored": 0,
    }
    if not serial or label not in {"NORMAL", "NG", "REVIEW"}:
        return empty_result

    events = load_fault_event_log()
    if events.empty or "event_id" not in events.columns:
        return empty_result
    serials = fault_event_serial_numbers(events)
    matched_events = events.loc[serials.eq(serial)].copy()
    matched_event_ids = (
        matched_events["event_id"].fillna("").astype(str).str.strip()
        if "event_id" in matched_events.columns
        else pd.Series(dtype="string")
    )
    matched_event_ids = matched_event_ids[matched_event_ids.ne("")].tolist()
    if not matched_event_ids:
        return empty_result

    if label == "NORMAL":
        _append_fault_delete_states(
            matched_event_ids,
            state="DELETED",
            serial_number=serial,
            review_id=review_id,
            reviewer=reviewer,
            reason=f"현장 판정 NORMAL · Serial Num {serial}",
        )
        return {
            **empty_result,
            "matched": len(matched_event_ids),
            "deleted": len(matched_event_ids),
        }

    target_status = "조치 대기" if label == "NG" else "검토 중"
    updated_count = 0
    for _, event in matched_events.iterrows():
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            continue
        review_note = f"현장 판정 {label}"
        if str(notes).strip():
            review_note = f"{review_note} · {str(notes).strip()}"
        final_action = str(event.get("final_action", "")).strip()
        if not final_action or final_action.casefold() in {"nan", "none", "<na>"}:
            final_action = "미결정"
        append_fault_action(
            {
                "event_id": event_id,
                "source_file": event.get("source_file", ""),
                "serial_number": serial,
                "fault_type": event.get("fault_type", ""),
                "action_status": target_status,
                "final_action": final_action,
                "assignee": str(reviewer).strip(),
                "action_notes": review_note,
                "action_origin": "HUMAN_REVIEW",
                "review_id": review_id,
            }
        )
        updated_count += 1
    return {
        **empty_result,
        "matched": len(matched_event_ids),
        "updated": updated_count,
    }


def apply_human_review_to_fault_events(
    serial_number: str,
    human_label: str,
    *,
    reviewer: str = "",
    notes: str = "",
    review_id: str = "",
) -> dict[str, Any]:
    """Replace the current field-review effect for one serial number."""
    serial = _normalize_serial_number(serial_number)
    cleared = _clear_human_review_effects(serial) if serial else {
        "restored": 0,
        "actions_removed": 0,
    }
    result = _apply_human_review_state(
        serial,
        human_label,
        reviewer=reviewer,
        notes=notes,
        review_id=str(review_id).strip() or uuid.uuid4().hex,
    )
    result["restored"] = cleared["restored"]
    result["actions_removed"] = cleared["actions_removed"]
    return result


def reconcile_human_review_for_serial(
    serial_number: str,
    *,
    reviews: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rebuild one serial's fault-log state from its latest remaining review."""
    serial = _normalize_serial_number(serial_number)
    cleared = _clear_human_review_effects(serial) if serial else {
        "restored": 0,
        "actions_removed": 0,
    }
    review_frame = load_reviews() if reviews is None else reviews.copy()
    if not serial or review_frame.empty:
        return {
            "serial_number": serial,
            "human_label": "",
            "matched": 0,
            "deleted": 0,
            "updated": 0,
            **cleared,
        }

    review_serials = review_frame.apply(_review_serial_number, axis=1)
    matching = review_frame.loc[review_serials.eq(serial)].copy()
    if matching.empty:
        return {
            "serial_number": serial,
            "human_label": "",
            "matched": 0,
            "deleted": 0,
            "updated": 0,
            **cleared,
        }
    if "reviewed_at" in matching.columns:
        matching = matching.sort_values("reviewed_at", kind="stable")
    latest = matching.iloc[-1]
    result = _apply_human_review_state(
        serial,
        latest.get("human_label", ""),
        reviewer=str(latest.get("reviewer", "")).strip(),
        notes=str(latest.get("notes", "")).strip(),
        review_id=str(latest.get("review_id", "")).strip() or uuid.uuid4().hex,
    )
    result["restored"] = cleared["restored"]
    result["actions_removed"] = cleared["actions_removed"]
    return result


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
