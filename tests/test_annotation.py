from __future__ import annotations

from connection_dynamics.annotation import _aggregate_annotations


def test_annotation_aggregation_measures_mutual_vulnerability_and_support() -> None:
    records = [
        {
            "speaker": "a",
            "parent_disclosure_depth": 0,
            "message_disclosure_depth": 2,
            "supportive_response": False,
        },
        {
            "speaker": "b",
            "parent_disclosure_depth": 2,
            "message_disclosure_depth": 2,
            "supportive_response": True,
        },
    ]

    features = _aggregate_annotations(records, expected_messages=2)

    assert features["disclosure_mean_depth"] == 2.0
    assert features["disclosure_balance"] == 1.0
    assert features["disclosure_reciprocity_rate"] == 1.0
    assert features["supportive_response_rate"] == 1.0
    assert features["annotation_coverage"] == 1.0
