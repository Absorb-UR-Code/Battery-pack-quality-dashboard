from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).resolve().parents[1] / "components" / "kpi_workspace"
_kpi_workspace = components.declare_component(
    "battery_pack_kpi_workspace",
    path=str(_COMPONENT_DIR),
)


def render_kpi_workspace(html: str, *, key: str) -> None:
    """Render the KPI workspace and let its iframe follow the content height."""
    _kpi_workspace(html=html, key=key, default=None, tab_index=0)
