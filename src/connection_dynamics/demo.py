"""Pure helpers for the local, evidence-first scenario demo."""

from __future__ import annotations

import bisect
import math
from typing import Any

TIER_A_ITEMS = [
    {
        "feature": "exposure_reply_count",
        "item": "We had repeated direct exchanges during the first month.",
    },
    {
        "feature": "active_day_fraction",
        "item": "We interacted on many different days rather than in one short burst.",
    },
    {
        "feature": "recency_days",
        "item": "Our interaction was still active near the end of the first month.",
    },
    {
        "feature": "reciprocity_balance",
        "item": "We contributed a similar share of the replies.",
    },
    {
        "feature": "latency_symmetry",
        "item": "Our usual response times were roughly similar.",
    },
    {
        "feature": "effort_balance",
        "item": "We put similar detail and effort into our messages.",
    },
    {
        "feature": "lexical_topic_overlap",
        "item": "Before meeting, we discussed similar topics.",
    },
    {
        "feature": "shared_subreddit_jaccard",
        "item": "Before meeting, we participated in similar communities.",
    },
    {
        "feature": "author_activity_balance",
        "item": "Before meeting, we were similarly active.",
    },
]

TIER_B_ITEMS = [
    "I felt safe expressing uncertainty or disagreement without expecting ridicule, "
    "rejection, or punishment.",
    "When I shared something personal, this person responded in a way that made me feel "
    "understood, accepted, and cared for.",
    "Personal sharing deepened in both directions rather than remaining one-sided.",
    "We both initiated contact and made time for the interaction.",
    "I felt able to be myself rather than perform.",
    "This person brought out a version of me I want to become.",
    "I felt accepted as I am and encouraged to grow.",
]


def _balance(left: float, right: float) -> float:
    total = left + right
    return 0.0 if total == 0 else 2.0 * min(left, right) / total


def _ratio_similarity(left: float, right: float) -> float:
    maximum = max(left, right)
    return 1.0 if maximum == 0 else min(left, right) / maximum


def _jaccard_from_counts(intersection: float, union: float) -> float:
    if intersection < 0 or union < 0 or intersection > union:
        raise ValueError("Intersection and union must satisfy 0 <= intersection <= union")
    return 0.0 if union == 0 else intersection / union


def build_model_features(
    *,
    replies_a_to_b: int,
    replies_b_to_a: int,
    median_response_hours_a: float | None,
    median_response_hours_b: float | None,
    average_words_a: float,
    average_words_b: float,
    active_days: int,
    recency_days: float,
    precontact_activity_a: int,
    precontact_activity_b: int,
    shared_communities: int,
    total_communities: int,
    shared_topic_tokens: int,
    total_topic_tokens: int,
) -> dict[str, float]:
    """Convert objective observations into the benchmark's nine model features."""

    counts = (
        replies_a_to_b,
        replies_b_to_a,
        active_days,
        precontact_activity_a,
        precontact_activity_b,
        shared_communities,
        total_communities,
        shared_topic_tokens,
        total_topic_tokens,
    )
    if any(value < 0 for value in counts):
        raise ValueError("Counts cannot be negative")
    if not 0 <= active_days <= 30:
        raise ValueError("Active days must be between 0 and 30")
    if not 0 <= recency_days <= 30:
        raise ValueError("Recency must be between 0 and 30 days")
    if average_words_a < 0 or average_words_b < 0:
        raise ValueError("Average word counts cannot be negative")

    if median_response_hours_a is None or median_response_hours_b is None:
        latency_symmetry = math.nan
    else:
        if median_response_hours_a <= 0 or median_response_hours_b <= 0:
            raise ValueError("Known response times must be greater than zero")
        latency_symmetry = math.exp(
            -abs(math.log(median_response_hours_a) - math.log(median_response_hours_b))
        )

    return {
        "shared_subreddit_jaccard": _jaccard_from_counts(
            shared_communities, total_communities
        ),
        "lexical_topic_overlap": _jaccard_from_counts(
            shared_topic_tokens, total_topic_tokens
        ),
        "author_activity_balance": _ratio_similarity(
            precontact_activity_a, precontact_activity_b
        ),
        "exposure_reply_count": float(replies_a_to_b + replies_b_to_a),
        "reciprocity_balance": _balance(replies_a_to_b, replies_b_to_a),
        "latency_symmetry": latency_symmetry,
        "effort_balance": _ratio_similarity(average_words_a, average_words_b),
        "active_day_fraction": active_days / 30.0,
        "recency_days": float(recency_days),
    }


def score_percentile(score: float, quantiles: list[float]) -> float:
    """Interpolate a score against 101 ascending held-out score quantiles."""

    if len(quantiles) != 101:
        raise ValueError("Expected exactly 101 score quantiles")
    if score <= quantiles[0]:
        return 0.0
    if score >= quantiles[-1]:
        return 100.0
    upper = bisect.bisect_left(quantiles, score)
    lower = upper - 1
    lower_score = quantiles[lower]
    upper_score = quantiles[upper]
    if upper_score == lower_score:
        return float(upper)
    fraction = (score - lower_score) / (upper_score - lower_score)
    return float(lower + fraction)


def reference_band(percentile: float, bands: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the non-overlapping held-out percentile band for a score percentile."""

    for band in bands:
        lower = float(band["lower_percentile"])
        upper = float(band["upper_percentile"])
        lower_match = percentile >= lower if lower == 0 else percentile > lower
        if lower_match and percentile <= upper:
            return band
    raise ValueError(f"No reference band covers percentile {percentile}")


def out_of_reference_features(
    values: dict[str, float], metadata: dict[str, dict[str, Any]]
) -> list[str]:
    """List features outside their training-cohort 1st-99th percentile range."""

    outside: list[str] = []
    for feature, value in values.items():
        if math.isnan(value):
            continue
        reference = metadata[feature]["training_reference"]
        if value < float(reference["p01"]) or value > float(reference["p99"]):
            outside.append(feature)
    return outside
