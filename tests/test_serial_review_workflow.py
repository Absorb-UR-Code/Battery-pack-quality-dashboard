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
from core.fault_log import build_fault_event


class SerialReviewWorkflowTests(unittest.TestCase):
    def test_new_fault_event_stores_serial_and_field_review_status(self) -> None:
        source = pd.DataFrame(
            {
                "SerialNumber": [798.0],
                "M01CV01": [4.02],
                "M01T01": [31.2],
            }
        )
        event = build_fault_event(
            {
                "summary": {
                    "status": "NG_REVIEW",
                    "model_id": "model-1",
                    "model_name": "test model",
                    "model_version": "1.0",
                },
                "details": {},
            },
            source_file="Test09_NG_dchg.csv",
            source_frame=source,
            detected_row=1,
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["serial_number"], "798")
        self.assertEqual(event["action_status"], "현장 검토 중")

    def test_human_labels_update_or_delete_faults_by_serial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fault_dir = Path(temp_dir) / "fault"
            fault_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "event_id": "event-798-a",
                        "source_file": "Test06_NG_chg.csv",
                        "serial_number": "798",
                        "fault_type": "센싱와이어 불량",
                        "action_status": "현장 검토 중",
                        "final_action": "미결정",
                        "assignee": np.nan,
                        "action_notes": np.nan,
                        "action_updated_at": np.nan,
                    },
                    {
                        "event_id": "event-798-b",
                        "source_file": "Test06_NG_dchg.csv",
                        "serial_number": np.nan,
                        "raw__SerialNumber": 798.0,
                        "fault_type": "센싱와이어 불량",
                        "action_status": "현장 검토 중",
                        "final_action": "미결정",
                        "assignee": np.nan,
                        "action_notes": np.nan,
                        "action_updated_at": np.nan,
                    },
                    {
                        "event_id": "event-999",
                        "source_file": "Test09_NG_dchg.csv",
                        "serial_number": "999",
                        "fault_type": "온도 센서 불량",
                        "action_status": "현장 검토 중",
                        "final_action": "미결정",
                        "assignee": np.nan,
                        "action_notes": np.nan,
                        "action_updated_at": np.nan,
                    },
                ]
            ).to_csv(
                fault_dir / "model_fault_event_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            with patch.object(storage, "FAULT_DIR", fault_dir):
                ng_result = storage.apply_human_review_to_fault_events(
                    "798",
                    "NG",
                    reviewer="작업자 A",
                    notes="현장 불량 확인",
                )
                ng_events = storage.load_fault_event_log()
                ng_actions = storage.load_fault_actions()

                review_result = storage.apply_human_review_to_fault_events(
                    "798",
                    "REVIEW",
                    reviewer="작업자 B",
                )
                review_events = storage.load_fault_event_log()

                normal_result = storage.apply_human_review_to_fault_events(
                    "798",
                    "NORMAL",
                    reviewer="작업자 C",
                )
                final_events = storage.load_fault_event_log()
                final_actions = storage.load_fault_actions()
                deleted_event_ids = storage.load_deleted_fault_event_ids()

        self.assertEqual(ng_result["updated"], 2)
        self.assertTrue(
            ng_events.loc[
                ng_events["event_id"].isin(["event-798-a", "event-798-b"]),
                "action_status",
            ].eq("조치 대기").all()
        )
        self.assertEqual(set(ng_actions["serial_number"].astype(str)), {"798"})
        self.assertEqual(review_result["updated"], 2)
        self.assertTrue(
            review_events.loc[
                review_events["event_id"].isin(["event-798-a", "event-798-b"]),
                "action_status",
            ].eq("검토 중").all()
        )
        self.assertEqual(normal_result["deleted"], 2)
        self.assertEqual(len(final_events), 3)
        self.assertEqual(deleted_event_ids, {"event-798-a", "event-798-b"})
        self.assertTrue(final_actions.empty)

    def test_deleting_normal_review_restores_original_fault_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fault_dir = Path(temp_dir) / "fault"
            review_dir = Path(temp_dir) / "review"
            fault_dir.mkdir(parents=True, exist_ok=True)
            review_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "event_id": "event-798",
                        "source_file": "Test09_NG_dchg.csv",
                        "serial_number": "798",
                        "fault_type": "온도 센서 불량",
                        "action_status": "현장 검토 중",
                        "final_action": "미결정",
                    }
                ]
            ).to_csv(
                fault_dir / "model_fault_event_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            with (
                patch.object(storage, "FAULT_DIR", fault_dir),
                patch.object(storage, "REVIEW_DIR", review_dir),
            ):
                review_id = "review-normal-798"
                storage.append_review(
                    {
                        "review_id": review_id,
                        "reviewer": "작업자 A",
                        "serial_number": "798",
                        "human_label": "NORMAL",
                        "notes": "정상 확인",
                    }
                )
                storage.apply_human_review_to_fault_events(
                    "798",
                    "NORMAL",
                    reviewer="작업자 A",
                    notes="정상 확인",
                    review_id=review_id,
                )
                hidden_ids = storage.load_deleted_fault_event_ids()
                delete_result = storage.delete_reviews([0])
                restored_ids = storage.load_deleted_fault_event_ids()
                restored_events = storage.load_fault_event_log()

        self.assertEqual(hidden_ids, {"event-798"})
        self.assertEqual(delete_result["deleted"], 1)
        self.assertEqual(delete_result["reconciled_serials"], 1)
        self.assertEqual(restored_ids, set())
        self.assertEqual(restored_events["event_id"].tolist(), ["event-798"])

    def test_deleting_ng_review_restores_original_action_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fault_dir = Path(temp_dir) / "fault"
            review_dir = Path(temp_dir) / "review"
            fault_dir.mkdir(parents=True, exist_ok=True)
            review_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "event_id": "event-578",
                        "source_file": "Test06_NG_chg.csv",
                        "serial_number": "578",
                        "fault_type": "센싱와이어 불량",
                        "action_status": "현장 검토 중",
                        "final_action": "미결정",
                    }
                ]
            ).to_csv(
                fault_dir / "model_fault_event_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            with (
                patch.object(storage, "FAULT_DIR", fault_dir),
                patch.object(storage, "REVIEW_DIR", review_dir),
            ):
                review_id = "review-ng-578"
                storage.append_review(
                    {
                        "review_id": review_id,
                        "reviewer": "작업자 B",
                        "serial_number": "578",
                        "human_label": "NG",
                    }
                )
                storage.apply_human_review_to_fault_events(
                    "578",
                    "NG",
                    reviewer="작업자 B",
                    review_id=review_id,
                )
                reviewed_events = storage.load_fault_event_log()
                storage.delete_reviews([0])
                restored_events = storage.load_fault_event_log()
                restored_actions = storage.load_fault_actions()

        self.assertEqual(reviewed_events.loc[0, "action_status"], "조치 대기")
        self.assertEqual(restored_events.loc[0, "action_status"], "현장 검토 중")
        self.assertTrue(restored_actions.empty)

    def test_deleting_latest_review_reapplies_previous_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fault_dir = Path(temp_dir) / "fault"
            review_dir = Path(temp_dir) / "review"
            fault_dir.mkdir(parents=True, exist_ok=True)
            review_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "event_id": "event-578",
                        "source_file": "Test06_NG_chg.csv",
                        "serial_number": "578",
                        "fault_type": "센싱와이어 불량",
                        "action_status": "현장 검토 중",
                        "final_action": "미결정",
                    }
                ]
            ).to_csv(
                fault_dir / "model_fault_event_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            with (
                patch.object(storage, "FAULT_DIR", fault_dir),
                patch.object(storage, "REVIEW_DIR", review_dir),
            ):
                storage.append_review(
                    {
                        "review_id": "review-ng-578",
                        "reviewed_at": "2026-07-28T10:00:00",
                        "reviewer": "작업자 A",
                        "serial_number": "578",
                        "human_label": "NG",
                    }
                )
                storage.apply_human_review_to_fault_events(
                    "578",
                    "NG",
                    reviewer="작업자 A",
                    review_id="review-ng-578",
                )
                storage.append_review(
                    {
                        "review_id": "review-review-578",
                        "reviewed_at": "2026-07-28T11:00:00",
                        "reviewer": "작업자 B",
                        "serial_number": "578",
                        "human_label": "REVIEW",
                    }
                )
                storage.apply_human_review_to_fault_events(
                    "578",
                    "REVIEW",
                    reviewer="작업자 B",
                    review_id="review-review-578",
                )

                before_delete = storage.load_fault_event_log()
                delete_result = storage.delete_reviews([1])
                after_delete = storage.load_fault_event_log()
                remaining_actions = storage.load_fault_actions()

        self.assertEqual(before_delete.loc[0, "action_status"], "검토 중")
        self.assertEqual(delete_result["deleted"], 1)
        self.assertEqual(after_delete.loc[0, "action_status"], "조치 대기")
        self.assertEqual(len(remaining_actions), 1)
        self.assertEqual(remaining_actions.loc[0, "review_id"], "review-ng-578")


if __name__ == "__main__":
    unittest.main()
