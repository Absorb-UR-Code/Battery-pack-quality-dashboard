"""Custom model adapter example.

Use only for trusted local code. Set model_type to "custom" and adapter to
"adapter.py" in manifest.json.
"""

from __future__ import annotations

import numpy as np


def predict(df, context):
    """Return row scores and optional predictions for a pandas DataFrame."""
    scores = np.zeros(len(df), dtype=float)
    threshold = float(context["spec"].get("threshold", 0.5))
    return {
        "scores": scores,
        "threshold": threshold,
        "predictions": scores >= threshold,
        "details": {
            "adapter": "example",
            # Optional fields for the fault log and future type classifier:
            # "fault_type": "온도 센서 불량",
            # "fault_confidence": 0.94,
            # "fault_probabilities": {"온도 센서 불량": 0.94},
            # "suspect_sensors": ["M16T02"],
            # "suspect_modules": ["M16"],
            # "suspect_cells": [],
            # "severity": "높음",
        },
    }
