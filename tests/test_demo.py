from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from connection_dynamics.demo import (
    TIER_A_ITEMS,
    build_model_features,
    out_of_reference_features,
    reference_band,
    score_percentile,
)

ROOT = Path(__file__).resolve().parents[1]


def _features(**overrides: object) -> dict[str, float]:
    values: dict[str, object] = {
        "replies_a_to_b": 3,
        "replies_b_to_a": 3,
        "median_response_hours_a": 8.0,
        "median_response_hours_b": 8.0,
        "average_words_a": 60.0,
        "average_words_b": 60.0,
        "active_days": 6,
        "recency_days": 2.0,
        "precontact_activity_a": 10,
        "precontact_activity_b": 10,
        "shared_communities": 2,
        "total_communities": 4,
        "shared_topic_tokens": 20,
        "total_topic_tokens": 100,
    }
    values.update(overrides)
    return build_model_features(**values)  # type: ignore[arg-type]


def test_equal_observations_produce_balanced_features() -> None:
    features = _features()

    assert features["exposure_reply_count"] == 6.0
    assert features["reciprocity_balance"] == 1.0
    assert features["latency_symmetry"] == 1.0
    assert features["effort_balance"] == 1.0
    assert features["active_day_fraction"] == 0.2
    assert features["shared_subreddit_jaccard"] == 0.5
    assert features["lexical_topic_overlap"] == 0.2


def test_one_sided_and_zero_interactions_follow_panel_formulas() -> None:
    one_sided = _features(
        replies_a_to_b=4,
        replies_b_to_a=0,
        average_words_a=50.0,
        average_words_b=0.0,
    )
    zero = _features(
        replies_a_to_b=0,
        replies_b_to_a=0,
        average_words_a=0.0,
        average_words_b=0.0,
        precontact_activity_a=0,
        precontact_activity_b=0,
    )

    assert one_sided["reciprocity_balance"] == 0.0
    assert one_sided["effort_balance"] == 0.0
    assert zero["exposure_reply_count"] == 0.0
    assert zero["reciprocity_balance"] == 0.0
    assert zero["effort_balance"] == 1.0
    assert zero["author_activity_balance"] == 1.0


def test_missing_latency_is_preserved_for_xgboost() -> None:
    features = _features(median_response_hours_a=None, median_response_hours_b=None)

    assert math.isnan(features["latency_symmetry"])


def test_invalid_raw_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="intersection"):
        _features(shared_communities=5, total_communities=4)
    with pytest.raises(ValueError, match="Active days"):
        _features(active_days=31)
    with pytest.raises(ValueError, match="greater than zero"):
        _features(median_response_hours_a=0.0)


def test_percentile_band_and_reference_warning_helpers() -> None:
    quantiles = [float(index) for index in range(101)]
    bands = [
        {"lower_percentile": 0, "upper_percentile": 50, "label": "bottom"},
        {"lower_percentile": 50, "upper_percentile": 100, "label": "top"},
    ]
    metadata = {
        "value": {"training_reference": {"p01": 0.1, "p99": 0.9}},
        "missing": {"training_reference": {"p01": 0.1, "p99": 0.9}},
    }

    assert score_percentile(73.5, quantiles) == 73.5
    assert reference_band(73.5, bands)["label"] == "top"
    assert out_of_reference_features(
        {"value": 1.0, "missing": math.nan}, metadata
    ) == ["value"]


def test_questionnaire_covers_every_current_model_feature_once() -> None:
    assert {item["feature"] for item in TIER_A_ITEMS} == set(_features())


def test_public_demo_artifacts_are_aggregate_and_reproducible() -> None:
    reference_path = ROOT / "artifacts" / "demo-reference.json"
    model_path = ROOT / "artifacts" / "combined-model.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {str(key).lower() for key in value} | set().union(
                *(keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    artifact_keys = keys(reference)
    for prohibited in ("author_a", "author_b", "message_text", "raw_predictions"):
        assert prohibited not in artifact_keys
    top_one = reference["rare_tie_lift"]["combined"][0]
    assert top_one["selected_dyads"] == 193
    assert top_one["durable_ties"] == 23
    assert top_one["lift_over_prevalence"] == pytest.approx(11.21, rel=0.01)

    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(model_path)
    smoke = reference["model_smoke_case"]
    order = reference["feature_order"]
    frame = pd.DataFrame([[smoke["features"][name] for name in order]], columns=order)
    assert float(model.predict_proba(frame)[0, 1]) == pytest.approx(smoke["score"], abs=1e-9)
