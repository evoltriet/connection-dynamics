"""Local Streamlit scenario explorer for the published durable-tie benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

from connection_dynamics.demo import (
    TIER_A_ITEMS,
    TIER_B_ITEMS,
    build_model_features,
    out_of_reference_features,
    reference_band,
    score_percentile,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "combined-model.json"
REFERENCE_PATH = ROOT / "artifacts" / "demo-reference.json"


@st.cache_resource
def load_model(path: Path) -> XGBClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    model = XGBClassifier()
    model.load_model(path)
    return model


@st.cache_resource
def load_explainer(path: Path) -> shap.TreeExplainer:
    return shap.TreeExplainer(load_model(path))


@st.cache_data
def load_reference(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing demo reference: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def local_contribution_chart(
    values: np.ndarray, feature_order: list[str], metadata: dict
) -> plt.Figure:
    order = np.argsort(np.abs(values))[-6:]
    labels = [metadata[feature_order[index]]["label"] for index in order]
    contributions = values[order]
    colors = ["#B45309" if value < 0 else "#0F766E" for value in contributions]
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    axis.barh(labels, contributions, color=colors)
    axis.axvline(0, color="#64748B", linewidth=1)
    axis.set_xlabel("Contribution to this scenario's model score")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return figure


def main() -> None:
    st.set_page_config(page_title="Rare durable-tie scenario explorer", layout="wide")
    st.title("Rare durable-tie scenario explorer")
    st.caption(
        "A local research demo of one Reddit cohort. It ranks interaction scenarios; it does "
        "not assess friendship, compatibility, or a person's relationship potential."
    )

    try:
        reference = load_reference(REFERENCE_PATH)
        model = load_model(MODEL_PATH)
    except FileNotFoundError as error:
        st.error(str(error))
        st.code(
            "python -m pip install -e \".[demo]\"\n"
            "# Then reproduce the benchmark artifacts using the README command."
        )
        st.stop()

    st.subheader("Objective first-month observations")
    first, second = st.columns(2)
    with first:
        replies_a = st.number_input("Replies from person A to B", min_value=0, value=1)
        replies_b = st.number_input("Replies from person B to A", min_value=0, value=1)
        response_data_known = st.checkbox("Response-time data available", value=True)
        response_a = st.number_input(
            "Typical response time for A (hours)",
            min_value=0.01,
            value=8.0,
            disabled=not response_data_known,
        )
        response_b = st.number_input(
            "Typical response time for B (hours)",
            min_value=0.01,
            value=32.0,
            disabled=not response_data_known,
        )
        words_a = st.number_input("Average words per reply from A", min_value=0.0, value=20.0)
        words_b = st.number_input("Average words per reply from B", min_value=0.0, value=20.0)
    with second:
        active_days = st.number_input(
            "Distinct interaction days in the first month", min_value=0, max_value=30, value=1
        )
        recency = st.number_input(
            "Days since the last interaction at day 30",
            min_value=0.0,
            max_value=30.0,
            value=30.0,
        )
        activity_a = st.number_input("Pre-contact posts by A", min_value=0, value=2)
        activity_b = st.number_input("Pre-contact posts by B", min_value=0, value=10)
        shared_communities = st.number_input("Shared communities", min_value=0, value=1)
        total_communities = st.number_input(
            "Total distinct communities", min_value=int(shared_communities), value=1
        )
        shared_tokens = st.number_input("Shared topic tokens", min_value=0, value=10)
        total_tokens = st.number_input(
            "Total distinct topic tokens", min_value=int(shared_tokens), value=100
        )

    features = build_model_features(
        replies_a_to_b=int(replies_a),
        replies_b_to_a=int(replies_b),
        median_response_hours_a=float(response_a) if response_data_known else None,
        median_response_hours_b=float(response_b) if response_data_known else None,
        average_words_a=float(words_a),
        average_words_b=float(words_b),
        active_days=int(active_days),
        recency_days=float(recency),
        precontact_activity_a=int(activity_a),
        precontact_activity_b=int(activity_b),
        shared_communities=int(shared_communities),
        total_communities=int(total_communities),
        shared_topic_tokens=int(shared_tokens),
        total_topic_tokens=int(total_tokens),
    )
    feature_order = reference["feature_order"]
    frame = pd.DataFrame([[features[name] for name in feature_order]], columns=feature_order)
    score = float(model.predict_proba(frame)[0, 1])
    percentile = score_percentile(score, reference["score_quantiles"])
    band = reference_band(percentile, reference["reference_bands"])
    outside = out_of_reference_features(features, reference["feature_metadata"])
    if outside:
        labels = [reference["feature_metadata"][name]["label"] for name in outside]
        st.warning(
            "Outside the training cohort's 1st-99th percentile range: " + ", ".join(labels)
        )

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Model-score percentile", f"{percentile:.0f}th")
    metric_two.metric("Reference band", band["label"])
    metric_three.metric("Observed cohort rate", f"{band['observed_rate']:.1%}")
    interval = band["wilson_95"]
    st.caption(
        f"In this held-out band, {band['durable_ties']} of {band['dyads']} dyads met the "
        f"online-interaction outcome (95% interval {interval[0]:.1%}-{interval[1]:.1%}). "
        "This cohort rate is not a personal probability."
    )

    explainer = load_explainer(MODEL_PATH)
    shap_values = np.asarray(explainer.shap_values(frame))[0]
    st.subheader("Why this scenario received its model score")
    figure = local_contribution_chart(
        shap_values, feature_order, reference["feature_metadata"]
    )
    st.pyplot(figure, clear_figure=True)
    st.caption(
        "Positive bars push this model's score higher; negative bars push it lower. "
        "SHAP explains model behavior and does not establish causation."
    )

    st.subheader("Questionnaire translation")
    st.write(
        "Use a seven-point agreement scale plus **Not enough information**. These prompts are "
        "for research design only; the demo does not collect or score responses."
    )
    with st.expander("Tier A - behavioral proxies represented in the current model"):
        for index, item in enumerate(TIER_A_ITEMS, start=1):
            label = reference["feature_metadata"][item["feature"]]["label"]
            st.markdown(f"{index}. {item['item']}  \n   *Computational proxy: {label}*")
        st.caption(
            "Self-reports are not interchangeable with the logged behaviors used to train "
            "the model."
        )
    with st.expander("Tier B - exploratory thesis constructs, not current model inputs"):
        for index, item in enumerate(TIER_B_ITEMS, start=1):
            st.markdown(f"{index}. {item}")
        st.caption(
            "These are candidate items, not validated measures. They require prospective month-one "
            "collection and days 90-180 outcome validation before predictive use."
        )


if __name__ == "__main__":
    main()
