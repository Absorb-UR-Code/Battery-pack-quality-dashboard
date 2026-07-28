from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .data_catalog import max_consecutive_true
from .features import build_row_features, prepare_feature_matrix


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    version: str
    model_type: str
    feature_profile: str
    root: str
    artifact: str = ""
    metadata: str = ""
    threshold: float = 0.5
    higher_is_anomaly: bool = True
    supported_modes: tuple[str, ...] = ("CHG", "DCHG", "UNKNOWN")
    description: str = ""
    validation_scope: str = ""
    metrics: dict[str, float] | None = None
    file_policy: dict[str, Any] | None = None
    adapter: str = ""
    enabled: bool = False
    deployment_status: str = "candidate"
    healthy: bool = True
    health_message: str = "정상"
    sha256: str = ""

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def artifact_path(self) -> Path | None:
        return self.root_path / self.artifact if self.artifact else None

    @property
    def metadata_path(self) -> Path | None:
        return self.root_path / self.metadata if self.metadata else None

    @property
    def is_active(self) -> bool:
        return (
            self.healthy
            and self.enabled
            and self.deployment_status.lower() in {"approved", "production"}
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_from_manifest(path: Path) -> ModelSpec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ModelSpec(
            model_id=path.parent.name,
            name=path.parent.name,
            version="unknown",
            model_type="invalid",
            feature_profile="unknown",
            root=str(path.parent),
            healthy=False,
            health_message=f"manifest 오류: {exc}",
        )
    artifact = str(raw.get("artifact", ""))
    artifact_path = path.parent / artifact if artifact else None
    model_type = str(raw.get("model_type", "sklearn_classifier"))
    requires_artifact = model_type not in {"statistical_guard", "custom_no_artifact"}
    healthy = not requires_artifact or bool(artifact_path and artifact_path.exists())
    message = "정상" if healthy else f"모델 파일 없음: {artifact or '(미지정)'}"
    checksum = _sha256(artifact_path) if healthy and artifact_path and artifact_path.is_file() else ""
    return ModelSpec(
        model_id=str(raw.get("model_id", path.parent.name)),
        name=str(raw.get("name", path.parent.name)),
        version=str(raw.get("version", "1.0.0")),
        model_type=model_type,
        feature_profile=str(raw.get("feature_profile", "battery_pack_v1")),
        root=str(path.parent),
        artifact=artifact,
        metadata=str(raw.get("metadata", "")),
        threshold=float(raw.get("threshold", 0.5)),
        higher_is_anomaly=bool(raw.get("higher_is_anomaly", True)),
        supported_modes=tuple(str(x).upper() for x in raw.get("supported_modes", ["CHG", "DCHG", "UNKNOWN"])),
        description=str(raw.get("description", "")),
        validation_scope=str(raw.get("validation_scope", "")),
        metrics=raw.get("metrics", {}),
        file_policy=raw.get("file_policy", {}),
        adapter=str(raw.get("adapter", "")),
        enabled=bool(raw.get("enabled", False)),
        deployment_status=str(raw.get("deployment_status", "candidate")),
        healthy=healthy,
        health_message=message,
        sha256=checksum,
    )


def discover_models(model_dir: Path = MODEL_DIR) -> list[ModelSpec]:
    model_dir.mkdir(parents=True, exist_ok=True)
    manifests = sorted(model_dir.glob("*/manifest.json"))
    return [_spec_from_manifest(path) for path in manifests]


def model_inventory(specs: list[ModelSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        metrics = spec.metrics or {}
        if not spec.healthy:
            registry_status = "ERROR"
        elif spec.is_active:
            registry_status = "ACTIVE"
        elif spec.enabled:
            registry_status = "승인 대기"
        else:
            registry_status = "후보(비활성)"
        rows.append(
            {
                "model_id": spec.model_id,
                "모델": spec.name,
                "버전": spec.version,
                "유형": spec.model_type,
                "특징 프로필": spec.feature_profile,
                "지원 모드": ", ".join(spec.supported_modes),
                "상태": registry_status,
                "배포 상태": spec.deployment_status,
                "F1": metrics.get("f1", np.nan),
                "Recall": metrics.get("recall", np.nan),
                "검증 범위": spec.validation_scope,
                "SHA256": spec.sha256[:12],
                "메시지": spec.health_message,
            }
        )
    return pd.DataFrame(rows)


def _metadata(spec: ModelSpec) -> dict[str, Any]:
    path = spec.metadata_path
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_custom_adapter(spec: ModelSpec):
    adapter_path = spec.root_path / spec.adapter
    if not adapter_path.exists():
        raise FileNotFoundError(f"사용자 정의 어댑터가 없습니다: {adapter_path}")
    return _load_custom_adapter_cached(
        str(adapter_path),
        spec.model_id,
        adapter_path.stat().st_mtime_ns,
    )


@lru_cache(maxsize=24)
def _load_custom_adapter_cached(adapter_text: str, model_id: str, modified_ns: int):
    del modified_ns
    adapter_path = Path(adapter_text)
    module_name = f"battery_model_adapter_{model_id}"
    module_spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"어댑터를 불러올 수 없습니다: {adapter_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise AttributeError("사용자 정의 adapter.py에 predict(df, context) 함수가 필요합니다.")
    return module


def load_model(spec: ModelSpec) -> Any:
    if not spec.healthy:
        raise RuntimeError(spec.health_message)
    if spec.model_type in {"statistical_guard", "custom_no_artifact"}:
        return None
    if spec.model_type == "custom":
        return _load_custom_adapter(spec)
    artifact = spec.artifact_path
    if artifact is None:
        raise FileNotFoundError("모델 artifact가 지정되지 않았습니다.")
    if artifact.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        return _load_joblib_cached(str(artifact), artifact.stat().st_mtime_ns)
    raise ValueError(f"기본 로더가 지원하지 않는 모델 형식입니다: {artifact.suffix}. custom adapter를 사용하세요.")


@lru_cache(maxsize=24)
def _load_joblib_cached(path_text: str, modified_ns: int) -> Any:
    del modified_ns
    return joblib.load(path_text)


def _feature_columns(model: Any, metadata: dict[str, Any]) -> list[str]:
    if metadata.get("feature_columns"):
        return [str(c) for c in metadata["feature_columns"]]
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(c) for c in names]
    configured = metadata.get("features")
    if configured:
        return [str(c) for c in configured]
    raise ValueError("feature_columns가 없습니다. metadata JSON 또는 학습 모델의 feature_names_in_가 필요합니다.")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _statistical_guard(features: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, Any]]:
    components = []
    names = []
    for col, scale in [
        ("cv_range", 0.030),
        ("cv_resid_maxabs", 0.020),
        ("temp_pair_gap_max", 3.0),
        ("temp_range", 5.0),
    ]:
        if col in features.columns:
            components.append(pd.to_numeric(features[col], errors="coerce").fillna(0.0).to_numpy() / scale)
            names.append(col)
    if not components:
        raise ValueError("통계 점검에 필요한 전압 또는 온도 파생변수를 만들 수 없습니다.")
    return np.nanmax(np.vstack(components), axis=0), 1.0, {"feature_columns": names, "missing_features": []}


def _pca_reconstruction_score(bundle: Any, matrix: pd.DataFrame) -> np.ndarray:
    if isinstance(bundle, dict):
        scaler = bundle.get("scaler") or bundle.get("preprocessor")
        pca = bundle.get("pca") or bundle.get("model")
    else:
        scaler = None
        pca = bundle
    values = scaler.transform(matrix) if scaler is not None else matrix.to_numpy(dtype=float)
    latent = pca.transform(values)
    restored = pca.inverse_transform(latent)
    return np.mean((np.asarray(values) - np.asarray(restored)) ** 2, axis=1)


def _aggregate(scores: np.ndarray, predictions: np.ndarray, threshold: float, policy: dict[str, Any]) -> dict[str, Any]:
    series = pd.Series(np.asarray(scores, dtype=float)).replace([np.inf, -np.inf], np.nan)
    flags = np.asarray(predictions, dtype=bool)
    fire_rate = float(flags.mean()) if len(flags) else 0.0
    max_run = max_consecutive_true(flags)
    p95 = float(series.quantile(0.95)) if series.notna().any() else np.nan
    score_max = float(series.max()) if series.notna().any() else np.nan
    fire_limit = float(policy.get("fire_rate_threshold", 0.10))
    min_run = int(policy.get("min_consecutive_rows", 5))
    p95_limit = float(policy.get("score_p95_threshold", threshold))
    fire_trigger = fire_rate >= fire_limit and p95 >= p95_limit
    run_trigger = max_run >= min_run and score_max >= threshold
    file_pred = bool(fire_trigger or run_trigger)
    return {
        "status": "NG_REVIEW" if file_pred else "NORMAL",
        "file_prediction": int(file_pred),
        "fire_rate": fire_rate,
        "score_p95": p95,
        "score_max": score_max,
        "max_consecutive_rows": max_run,
        "row_threshold": float(threshold),
        "fire_rate_threshold": fire_limit,
        "min_consecutive_rows": min_run,
        "trigger": ", ".join(
            name for name, active in [("이상 비율", fire_trigger), ("연속 이상", run_trigger)] if active
        ) or "없음",
    }


def score_dataframe(spec: ModelSpec, df: pd.DataFrame, source_file: str) -> dict[str, Any]:
    features = build_row_features(df, source_file=source_file)
    model = load_model(spec)
    metadata = _metadata(spec)
    details: dict[str, Any] = {}

    if spec.model_type == "statistical_guard":
        scores, threshold, details = _statistical_guard(features)
        predictions = scores >= threshold
    elif spec.model_type == "custom":
        result = model.predict(
            df,
            {
                "spec": spec.to_dict(),
                "metadata": metadata,
                "source_file": source_file,
                "mode": infer_mode_from_features(features),
            },
        )
        scores = np.asarray(result["scores"], dtype=float)
        threshold = float(result.get("threshold", spec.threshold))
        predictions = np.asarray(result.get("predictions", scores >= threshold), dtype=bool)
        details = dict(result.get("details", {}))
    else:
        columns = _feature_columns(model, metadata)
        matrix, missing = prepare_feature_matrix(features, columns, metadata.get("fill_values"))
        details = {
            "feature_columns": columns,
            "missing_features": missing,
            "compatibility": 1.0 - (len(missing) / max(1, len(columns))),
        }
        threshold = float(metadata.get("threshold", metadata.get("iforest_threshold", spec.threshold)))

        if spec.model_type == "sklearn_classifier":
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(matrix))
                classes = list(getattr(model, "classes_", [0, 1]))
                positive_index = classes.index(1) if 1 in classes else min(1, proba.shape[1] - 1)
                scores = proba[:, positive_index]
            elif hasattr(model, "decision_function"):
                scores = _sigmoid(np.asarray(model.decision_function(matrix)))
            else:
                scores = np.asarray(model.predict(matrix), dtype=float)
            predictions = scores >= threshold
        elif spec.model_type in {"sklearn_anomaly", "ocsvm", "isolation_forest"}:
            raw = np.asarray(model.decision_function(matrix), dtype=float)
            scores = -raw if spec.higher_is_anomaly else raw
            predictions = scores >= threshold
        elif spec.model_type == "pca_reconstruction":
            scores = _pca_reconstruction_score(model, matrix)
            predictions = scores >= threshold
        else:
            raise ValueError(f"지원하지 않는 model_type입니다: {spec.model_type}")

    if len(scores) != len(df):
        raise ValueError(f"모델 출력 행 수({len(scores)})와 입력 행 수({len(df)})가 다릅니다.")
    row_result = pd.DataFrame(
        {
            "row_index": np.arange(1, len(df) + 1),
            "score": np.asarray(scores, dtype=float),
            "predicted_anomaly": np.asarray(predictions, dtype=int),
        }
    )
    summary = _aggregate(scores, predictions, threshold, spec.file_policy or {})
    summary_override = details.get("summary_override")
    if isinstance(summary_override, dict):
        allowed_override_fields = {
            "status",
            "file_prediction",
            "trigger",
            "fire_rate",
            "score_p95",
            "score_max",
            "max_consecutive_rows",
        }
        summary.update(
            {
                key: value
                for key, value in summary_override.items()
                if key in allowed_override_fields
            }
        )
    if details.get("insufficient_rows"):
        summary.update(
            {
                "status": "INSUFFICIENT_DATA",
                "file_prediction": 0,
                "trigger": f"판정 보류 · 최소 {details.get('required_rows', 100)}행 필요",
            }
        )
    summary.update(
        {
            "model_id": spec.model_id,
            "model_name": spec.name,
            "model_version": spec.version,
            "source_file": source_file,
            "mode": infer_mode_from_features(features),
            "compatibility": float(details.get("compatibility", 1.0)),
            "missing_feature_count": len(details.get("missing_features", [])),
        }
    )
    return {
        "row_result": row_result,
        "summary": summary,
        "features": features,
        "details": details,
    }


def infer_mode_from_features(features: pd.DataFrame) -> str:
    if "mode" in features.columns and len(features):
        return str(features["mode"].iloc[0])
    return "UNKNOWN"
