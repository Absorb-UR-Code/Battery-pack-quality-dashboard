from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = APP_ROOT / "config"
DATA_DIR = APP_ROOT / "data"
MODEL_DIR = APP_ROOT / "models"
OUTPUT_DIR = APP_ROOT / "outputs"
INBOX_DIR = DATA_DIR / "inbox"
REVIEW_DIR = OUTPUT_DIR / "review"
BATCH_DIR = OUTPUT_DIR / "batch"
FAULT_DIR = OUTPUT_DIR / "fault"
SETTINGS_PATH = CONFIG_DIR / "settings.json"
LOCAL_SETTINGS_PATH = CONFIG_DIR / "settings.local.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "data_sources": [
        {
            "name": "배포 예제 데이터",
            "path": str(DATA_DIR / "demo"),
            "enabled": True,
            "recursive": True,
            "role": "demo",
        },
        {
            "name": "업로드 Inbox",
            "path": str(INBOX_DIR),
            "enabled": True,
            "recursive": True,
            "role": "inbox",
        },
    ],
    "quality_policy": {
        "time_gap_seconds": 30.0,
        "max_missing_sensor_rate": 0.01,
        "max_duplicate_rate": 0.01,
        "min_cell_voltage": 1.5,
        "max_cell_voltage": 5.0,
        "min_temperature": -40.0,
        "max_temperature": 100.0,
    },
    "display": {
        "max_plot_rows": 3000,
        "max_heatmap_rows": 500,
        "default_batch_limit": 50,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base, ensure_ascii=False))
    merged.update(override)
    merged["quality_policy"] = {
        **base.get("quality_policy", {}),
        **override.get("quality_policy", {}),
    }
    merged["display"] = {
        **base.get("display", {}),
        **override.get("display", {}),
    }
    return merged


def _resolve_data_source_paths(settings: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(settings, ensure_ascii=False))
    for source in resolved.get("data_sources", []):
        raw_path = os.path.expandvars(str(source.get("path", "")).strip())
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = APP_ROOT / path
        source["path"] = str(path.resolve())
    return resolved


def ensure_app_directories() -> None:
    for path in [CONFIG_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR, INBOX_DIR, REVIEW_DIR, BATCH_DIR, FAULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    ensure_app_directories()
    if not SETTINGS_PATH.exists():
        save_settings(DEFAULT_SETTINGS)
    settings = _merge_settings(DEFAULT_SETTINGS, _read_json(SETTINGS_PATH))
    settings = _merge_settings(settings, _read_json(LOCAL_SETTINGS_PATH))
    return _resolve_data_source_paths(settings)


def save_settings(settings: dict[str, Any]) -> None:
    ensure_app_directories()
    target = LOCAL_SETTINGS_PATH if LOCAL_SETTINGS_PATH.exists() else SETTINGS_PATH
    target.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
