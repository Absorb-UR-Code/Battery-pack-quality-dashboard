from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import storage


class StoredRecordDeletionTests(unittest.TestCase):
    def test_delete_selected_review_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            review_dir = Path(temp_dir) / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"reviewed_at": "2026-07-28T10:00:00", "reviewer": "A"},
                    {"reviewed_at": "2026-07-28T10:01:00", "reviewer": "B"},
                    {"reviewed_at": "2026-07-28T10:02:00", "reviewer": "C"},
                ]
            ).to_csv(
                review_dir / "operator_review_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            with patch.object(storage, "REVIEW_DIR", review_dir):
                result = storage.delete_reviews([1])
                reviews = storage.load_reviews()

        self.assertEqual(result["deleted"], 1)
        self.assertEqual(reviews["reviewer"].tolist(), ["A", "C"])

    def test_delete_action_restores_previous_then_default_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fault_dir = Path(temp_dir) / "fault"
            fault_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "event_id": "event-1",
                        "source_file": "Test09_NG_dchg.csv",
                        "action_status": "신규",
                        "final_action": "미결정",
                        "assignee": "",
                        "action_notes": "",
                        "action_updated_at": "",
                    }
                ]
            ).to_csv(
                fault_dir / "model_fault_event_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(
                [
                    {
                        "updated_at": "2026-07-28T10:00:00",
                        "event_id": "event-1",
                        "action_status": "검토 중",
                        "final_action": "재실험",
                        "assignee": "담당자 A",
                        "action_notes": "1차 기록",
                    },
                    {
                        "updated_at": "2026-07-28T10:01:00",
                        "event_id": "event-1",
                        "action_status": "완료",
                        "final_action": "정상 복귀",
                        "assignee": "담당자 B",
                        "action_notes": "최종 기록",
                    },
                ]
            ).to_csv(
                fault_dir / "fault_action_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            with patch.object(storage, "FAULT_DIR", fault_dir):
                latest_result = storage.delete_fault_actions([1])
                events_after_latest = storage.load_fault_event_log()
                first_result = storage.delete_fault_actions([0])
                events_after_all = storage.load_fault_event_log()
                remaining_actions = storage.load_fault_actions()

        self.assertEqual(latest_result["deleted"], 1)
        self.assertEqual(events_after_latest.loc[0, "action_status"], "검토 중")
        self.assertEqual(events_after_latest.loc[0, "final_action"], "재실험")
        self.assertEqual(events_after_latest.loc[0, "assignee"], "담당자 A")
        self.assertEqual(first_result["deleted"], 1)
        self.assertEqual(events_after_all.loc[0, "action_status"], "현장 검토 중")
        self.assertEqual(events_after_all.loc[0, "final_action"], "미결정")
        self.assertTrue(
            pd.isna(events_after_all.loc[0, "assignee"])
            or events_after_all.loc[0, "assignee"] == ""
        )
        self.assertTrue(remaining_actions.empty)


if __name__ == "__main__":
    unittest.main()
