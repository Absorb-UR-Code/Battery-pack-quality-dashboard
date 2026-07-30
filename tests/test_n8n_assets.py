from __future__ import annotations

import csv
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TABLE_TEMPLATE = (
    ROOT / "n8n_workflows" / "battery_fault_events_table_template.csv"
)


class N8nAssetTests(unittest.TestCase):
    def test_table_template_forces_model_version_to_string(self) -> None:
        with TABLE_TEMPLATE.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 1)
        model_version = rows[0]["model_version"]
        self.assertEqual(model_version, "v1.0.0")
        self.assertFalse(model_version.replace(".", "", 2).isdigit())


if __name__ == "__main__":
    unittest.main()
