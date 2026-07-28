from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import storage


class FaultActionStorageTests(unittest.TestCase):
    def test_complete_action_updates_empty_text_columns_loaded_as_float(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fault_dir = Path(temp_dir) / "fault"
            fault_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "event_id": "ng9-action-test",
                        "source_file": "Test09_NG_dchg.csv",
                        "fault_type": "온도 센서 불량",
                        "action_status": "신규",
                        "final_action": "미결정",
                        "assignee": np.nan,
                        "action_notes": np.nan,
                        "action_updated_at": np.nan,
                    }
                ]
            ).to_csv(
                fault_dir / "model_fault_event_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            with patch.object(storage, "FAULT_DIR", fault_dir):
                storage.append_fault_action(
                    {
                        "event_id": "ng9-action-test",
                        "source_file": "Test09_NG_dchg.csv",
                        "fault_type": "온도 센서 불량",
                        "action_status": "완료",
                        "final_action": "폐기 검토",
                        "assignee": "김준영",
                        "action_notes": "용량 미달",
                    }
                )
                events = storage.load_fault_event_log()
                actions = storage.load_fault_actions()

        self.assertEqual(events.loc[0, "action_status"], "완료")
        self.assertEqual(events.loc[0, "final_action"], "폐기 검토")
        self.assertEqual(events.loc[0, "assignee"], "김준영")
        self.assertEqual(events.loc[0, "action_notes"], "용량 미달")
        self.assertTrue(str(events.loc[0, "action_updated_at"]).strip())
        self.assertEqual(int((~events["action_status"].eq("완료")).sum()), 0)
        self.assertEqual(len(actions), 1)


if __name__ == "__main__":
    unittest.main()
