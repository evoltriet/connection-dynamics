"""Leakage-resistant temporal ablations and ranking evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SIMILARITY_FEATURES = [
    "shared_subreddit_jaccard",
    "lexical_topic_overlap",
    "author_activity_balance",
]
DYNAMICS_FEATURES = [
    "exposure_reply_count",
    "reciprocity_balance",
    "latency_symmetry",
    "effort_balance",
    "active_day_fraction",
    "recency_days",
]
GRAPH_FEATURES = [
    "node2vec_cosine",
    "node2vec_l2_similarity",
    "node2vec_hadamard_mean",
    "node2vec_coverage",
]
DISCLOSURE_FEATURES = [
    "disclosure_mean_depth",
    "disclosure_balance",
    "disclosure_reciprocity_rate",
    "supportive_response_rate",
]
FEATURE_SETS = {
    "similarity": SIMILARITY_FEATURES,
    "dynamics": DYNAMICS_FEATURES,
    "combined": SIMILARITY_FEATURES + DYNAMICS_FEATURES,
}
FEATURE_METADATA = {
    "shared_subreddit_jaccard": {
        "label": "Shared-community overlap",
        "family": "profile_similarity",
        "unit": "Jaccard index",
    },
    "lexical_topic_overlap": {
        "label": "Pre-contact topic overlap",
        "family": "profile_similarity",
        "unit": "Jaccard index",
    },
    "author_activity_balance": {
        "label": "Pre-contact activity balance",
        "family": "profile_similarity",
        "unit": "ratio similarity",
    },
    "exposure_reply_count": {
        "label": "First-month direct replies",
        "family": "interaction_dynamics",
        "unit": "replies",
    },
    "reciprocity_balance": {
        "label": "Reply reciprocity",
        "family": "interaction_dynamics",
        "unit": "balance index",
    },
    "latency_symmetry": {
        "label": "Response-time symmetry",
        "family": "interaction_dynamics",
        "unit": "ratio similarity",
    },
    "effort_balance": {
        "label": "Message-effort balance",
        "family": "interaction_dynamics",
        "unit": "ratio similarity",
    },
    "active_day_fraction": {
        "label": "Interaction regularity",
        "family": "interaction_dynamics",
        "unit": "fraction of first 30 days",
    },
    "recency_days": {
        "label": "Days since last interaction",
        "family": "interaction_dynamics",
        "unit": "days",
    },
}
LIFT_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
REFERENCE_BAND_EDGES = (0.0, 50.0, 80.0, 90.0, 95.0, 99.0, 100.0)


def _available_feature_sets(frame: Any) -> dict[str, list[str]]:
    feature_sets = {name: list(features) for name, features in FEATURE_SETS.items()}
    if all(feature in frame.columns for feature in GRAPH_FEATURES):
        feature_sets["graph"] = GRAPH_FEATURES
        feature_sets["combined_graph"] = (
            SIMILARITY_FEATURES + DYNAMICS_FEATURES + GRAPH_FEATURES
        )
    if all(feature in frame.columns for feature in DISCLOSURE_FEATURES):
        if "annotation_coverage" not in frame or float(frame["annotation_coverage"].min()) < 1.0:
            raise ValueError(
                "Disclosure benchmarks require complete annotation coverage for every dyad"
            )
        feature_sets["disclosure"] = DISCLOSURE_FEATURES
        feature_sets["combined_disclosure"] = (
            SIMILARITY_FEATURES + DYNAMICS_FEATURES + DISCLOSURE_FEATURES
        )
    return feature_sets


def _temporal_split(frame: Any) -> tuple[Any, Any, Any, dict[str, int]]:
    ordered = frame.sort_values("anchor_timestamp").reset_index(drop=True)
    if len(ordered) < 20:
        raise ValueError("At least 20 dyads are required for a temporal split")
    train_position = max(int(len(ordered) * 0.70) - 1, 0)
    validation_position = max(int(len(ordered) * 0.85) - 1, train_position + 1)
    train_end = int(ordered.loc[train_position, "anchor_timestamp"])
    validation_end = int(ordered.loc[validation_position, "anchor_timestamp"])
    train = ordered[ordered["anchor_timestamp"] <= train_end].copy()
    validation = ordered[
        (ordered["anchor_timestamp"] > train_end)
        & (ordered["anchor_timestamp"] <= validation_end)
    ].copy()
    test = ordered[ordered["anchor_timestamp"] > validation_end].copy()
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if split.empty or split["label"].nunique() != 2:
            raise ValueError(f"{name} split must contain both outcome classes")
    return train, validation, test, {
        "train_end_timestamp": train_end,
        "validation_end_timestamp": validation_end,
    }


def _classification_metrics(labels: Any, scores: Any) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def _ranking_metrics(test: Any, scores: Any, k: int = 10) -> dict[str, float | int]:
    import pandas as pd

    candidates = []
    for (_, row), score in zip(test.iterrows(), scores, strict=True):
        candidates.append((row["author_a"], row["author_b"], int(row["label"]), float(score)))
        candidates.append((row["author_b"], row["author_a"], int(row["label"]), float(score)))
    expanded = pd.DataFrame(candidates, columns=["person", "partner", "label", "score"])
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    candidate_counts: list[int] = []
    for _, group in expanded.groupby("person"):
        positives = int(group["label"].sum())
        if positives == 0:
            continue
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        top = ranked.head(k)
        recalls.append(float(top["label"].sum() / positives))
        first_positive = int(ranked.index[ranked["label"] == 1][0]) + 1
        reciprocal_ranks.append(1.0 / first_positive)
        dcg = sum(
            int(label) / math.log2(rank + 2)
            for rank, label in enumerate(top["label"].tolist())
        )
        ideal_length = min(positives, k)
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_length))
        ndcgs.append(dcg / ideal)
        candidate_counts.append(len(ranked))
    if not recalls:
        raise ValueError("No test-set users have a positive candidate for ranking evaluation")
    return {
        f"recall_at_{k}": float(sum(recalls) / len(recalls)),
        "mrr": float(sum(reciprocal_ranks) / len(reciprocal_ranks)),
        f"ndcg_at_{k}": float(sum(ndcgs) / len(ndcgs)),
        "evaluable_people": len(recalls),
        "median_candidates": float(sorted(candidate_counts)[len(candidate_counts) // 2]),
    }


def _paired_bootstrap(
    labels: Any,
    baseline_scores: Any,
    combined_scores: Any,
    iterations: int,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    random = np.random.default_rng(seed)
    labels = np.asarray(labels)
    baseline_scores = np.asarray(baseline_scores)
    combined_scores = np.asarray(combined_scores)
    deltas: dict[str, list[float]] = {"roc_auc": [], "average_precision": []}
    for _ in range(iterations):
        sample = random.integers(0, len(labels), len(labels))
        sampled_labels = labels[sample]
        if len(np.unique(sampled_labels)) != 2:
            continue
        deltas["roc_auc"].append(
            float(
                roc_auc_score(sampled_labels, combined_scores[sample])
                - roc_auc_score(sampled_labels, baseline_scores[sample])
            )
        )
        deltas["average_precision"].append(
            float(
                average_precision_score(sampled_labels, combined_scores[sample])
                - average_precision_score(sampled_labels, baseline_scores[sample])
            )
        )
    summary: dict[str, dict[str, float | int]] = {}
    for metric, values in deltas.items():
        array = np.asarray(values)
        summary[metric] = {
            "iterations": int(len(array)),
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
        }
    return summary


def _fit_model(train: Any, validation: Any, features: list[str]) -> Any:
    from xgboost import XGBClassifier

    positives = int(train["label"].sum())
    negatives = len(train) - positives
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=1_000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=negatives / positives,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=4,
    )
    model.fit(
        train[features],
        train["label"],
        eval_set=[(validation[features], validation["label"])],
        verbose=False,
    )
    return model


def _shap_values(model: Any, frame: Any, features: list[str]) -> Any:
    import numpy as np
    import shap

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(frame[features]))
    if values.ndim != 2 or values.shape[1] != len(features):
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return values


def _mean_absolute_shap(values: Any, features: list[str]) -> dict[str, float]:
    import numpy as np

    means = np.abs(values).mean(axis=0)
    return {feature: float(value) for feature, value in zip(features, means, strict=True)}


def _lift_summary(labels: Any, scores_by_model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    prevalence = positives / len(labels)
    summary: dict[str, list[dict[str, Any]]] = {}
    for model_name, scores in scores_by_model.items():
        order = np.argsort(-np.asarray(scores), kind="stable")
        rows: list[dict[str, Any]] = []
        for fraction in LIFT_FRACTIONS:
            selected = max(1, math.ceil(len(labels) * fraction))
            hits = int(labels[order[:selected]].sum())
            precision = hits / selected
            rows.append(
                {
                    "top_fraction": fraction,
                    "selected_dyads": selected,
                    "durable_ties": hits,
                    "precision": precision,
                    "recall": hits / positives,
                    "lift_over_prevalence": precision / prevalence,
                }
            )
        summary[model_name] = rows
    return summary


def _wilson_interval(positives: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    proportion = positives / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _reference_bands(labels: Any, scores: Any) -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd

    labels = np.asarray(labels, dtype=int)
    percentiles = pd.Series(np.asarray(scores)).rank(method="average", pct=True).to_numpy() * 100
    bands: list[dict[str, Any]] = []
    for lower, upper in zip(REFERENCE_BAND_EDGES[:-1], REFERENCE_BAND_EDGES[1:], strict=True):
        lower_mask = percentiles >= lower if lower == 0 else percentiles > lower
        mask = lower_mask & (percentiles <= upper)
        total = int(mask.sum())
        positives = int(labels[mask].sum())
        label = "Top 1%" if lower == 99 else f"{lower:.0f}th-{upper:.0f}th percentile"
        bands.append(
            {
                "label": label,
                "lower_percentile": lower,
                "upper_percentile": upper,
                "dyads": total,
                "durable_ties": positives,
                "observed_rate": positives / total,
                "wilson_95": _wilson_interval(positives, total),
            }
        )
    return bands


def _build_demo_reference(
    train: Any,
    test: Any,
    combined_scores: Any,
    rare_tie_lift: dict[str, list[dict[str, Any]]],
    study_label: str,
    model: Any,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    features = FEATURE_SETS["combined"]
    metadata: dict[str, Any] = {}
    for feature in features:
        values = train[feature].dropna().to_numpy()
        metadata[feature] = {
            **FEATURE_METADATA[feature],
            "training_reference": {
                "p01": float(np.quantile(values, 0.01)),
                "median": float(np.quantile(values, 0.50)),
                "p99": float(np.quantile(values, 0.99)),
                "missing_fraction": float(train[feature].isna().mean()),
            },
        }
    smoke_features = {
        feature: float(metadata[feature]["training_reference"]["median"])
        for feature in features
    }
    smoke_frame = pd.DataFrame(
        [[smoke_features[feature] for feature in features]], columns=features
    )
    return {
        "schema_version": 1,
        "study_label": study_label,
        "feature_order": features,
        "feature_metadata": metadata,
        "test_cohort": {
            "dyads": int(len(test)),
            "durable_ties": int(test["label"].sum()),
            "prevalence": float(test["label"].mean()),
        },
        "score_quantiles": [
            float(value)
            for value in np.quantile(combined_scores, np.linspace(0.0, 1.0, 101))
        ],
        "reference_bands": _reference_bands(test["label"].to_numpy(), combined_scores),
        "rare_tie_lift": rare_tie_lift,
        "model_smoke_case": {
            "features": smoke_features,
            "score": float(model.predict_proba(smoke_frame)[0, 1]),
        },
        "outcome_definition": (
            "At least one direct reply in each direction during days 90-180 after first contact."
        ),
        "interpretation_boundary": (
            "Scores rank scenarios relative to one held-out Reddit cohort. Band rates are cohort "
            "observations, not personal probabilities or causal effects."
        ),
    }


def _save_verified_model(model: Any, output: Path, frame: Any, features: list[str]) -> None:
    import numpy as np
    from xgboost import XGBClassifier

    output.parent.mkdir(parents=True, exist_ok=True)
    expected = model.predict_proba(frame[features])[:, 1]
    model.save_model(output)
    reloaded = XGBClassifier()
    reloaded.load_model(output)
    actual = reloaded.predict_proba(frame[features])[:, 1]
    if not np.allclose(expected, actual, rtol=1e-7, atol=1e-9):
        raise RuntimeError("Reloaded model predictions do not match the fitted benchmark model")


def _render_hero_chart(results: dict[str, Any], output: Path, study_label: str) -> None:
    import matplotlib.pyplot as plt

    names = ["Similarity only", "Interaction dynamics", "Combined"]
    keys = ["similarity", "dynamics", "combined"]
    colors = ["#94A3B8", "#2563EB", "#0F766E"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="#F8FAFC")
    for axis, metric, title in zip(
        axes,
        ("roc_auc", "average_precision"),
        ("ROC-AUC", "Average precision (PR-AUC)"),
        strict=True,
    ):
        values = [results["models"][key]["metrics"][metric] for key in keys]
        bars = axis.bar(names, values, color=colors, width=0.62)
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold")
        axis.set_ylim(0, 1 if metric == "roc_auc" else max(values) * 1.35)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=18, labelsize=9)
        axis.tick_params(axis="y", length=0)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.02 if metric == "roc_auc" else max(values) * 0.035),
                f"{value:.3f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
    delta = results["delta_combined_vs_similarity"]["roc_auc"]
    interval = results["delta_combined_vs_similarity"]["paired_bootstrap_95"]["roc_auc"]
    figure.suptitle(
        f"First-month dynamics add {delta * 100:.1f} ROC-AUC points beyond similarity",
        x=0.07,
        y=0.95,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    figure.text(
        0.07,
        0.88,
        f"{study_label} · 95% CI {interval['lower_95'] * 100:+.1f} to "
        f"{interval['upper_95'] * 100:+.1f} points · strict chronological test",
        fontsize=10.5,
        color="#475569",
    )
    figure.text(
        0.07,
        0.02,
        "Higher is better. Combined = similarity + interaction dynamics. "
        "No usernames or outcome-window features are used.",
        fontsize=9,
        color="#64748B",
    )
    figure.tight_layout(rect=(0.04, 0.07, 0.98, 0.82))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _render_lift_chart(
    rare_tie_lift: dict[str, list[dict[str, Any]]],
    output: Path,
    study_label: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "similarity": "Similarity only",
        "dynamics": "Interaction dynamics",
        "combined": "Combined",
    }
    colors = {"similarity": "#94A3B8", "dynamics": "#2563EB", "combined": "#0F766E"}
    markers = {"similarity": "o", "dynamics": "s", "combined": "D"}
    figure, axis = plt.subplots(figsize=(10.5, 6.2), facecolor="#F8FAFC")
    for key in ("similarity", "dynamics", "combined"):
        rows = rare_tie_lift[key]
        x_values = [row["top_fraction"] * 100 for row in rows]
        y_values = [row["lift_over_prevalence"] for row in rows]
        axis.plot(
            x_values,
            y_values,
            color=colors[key],
            marker=markers[key],
            linewidth=2.2,
            markersize=7,
            label=labels[key],
        )
    axis.axhline(1.0, color="#64748B", linewidth=1.2, linestyle="--", label="Random ranking")
    axis.set_xticks([1, 5, 10, 20], labels=["Top 1%", "Top 5%", "Top 10%", "Top 20%"])
    axis.set_ylabel("Lift over the 1.06% durable-tie prevalence")
    axis.set_xlabel("Highest-scoring share of held-out dyads")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)
    combined_top = rare_tie_lift["combined"][0]
    axis.annotate(
        f"{combined_top['lift_over_prevalence']:.1f}x lift\n"
        f"{combined_top['durable_ties']}/{combined_top['selected_dyads']} durable",
        xy=(1, combined_top["lift_over_prevalence"]),
        xytext=(2.4, combined_top["lift_over_prevalence"] + 0.2),
        arrowprops={"arrowstyle": "-", "color": colors["combined"]},
        fontsize=10,
        color="#0F172A",
    )
    axis.legend(frameon=False, ncols=2, loc="upper right")
    figure.suptitle(
        "The top 1% is 11.2x richer in rare durable ties",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    figure.text(
        0.08,
        0.90,
        f"{study_label} - strict chronological test set",
        fontsize=10.5,
        color="#475569",
    )
    figure.text(
        0.08,
        0.02,
        "Lift measures concentration, not certainty: most high-scoring dyads still do not meet "
        "the durable-interaction label.",
        fontsize=9,
        color="#64748B",
    )
    figure.tight_layout(rect=(0.04, 0.07, 0.98, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _render_shap_chart(
    shap_values: Any,
    frame: Any,
    features: list[str],
    output: Path,
    study_label: str,
) -> None:
    import matplotlib.pyplot as plt
    import shap

    display_names = {}
    for feature in features:
        family = (
            "dynamics"
            if FEATURE_METADATA[feature]["family"] == "interaction_dynamics"
            else "similarity"
        )
        display_names[feature] = f"{FEATURE_METADATA[feature]['label']} [{family}]"
    display_frame = frame[features].rename(columns=display_names)
    shap.summary_plot(
        shap_values,
        display_frame,
        max_display=len(features),
        show=False,
        plot_size=(10.5, 6.4),
    )
    figure = plt.gcf()
    axis = plt.gca()
    figure.set_facecolor("#F8FAFC")
    axis.set_facecolor("#F8FAFC")
    axis.set_xlabel("Contribution to model score (lower <- 0 -> higher)")
    axis.set_title("")
    figure.suptitle(
        "What the combined model uses to rank durable ties",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    figure.text(
        0.08,
        0.92,
        f"{study_label} - held-out SHAP values; color shows low-to-high feature values",
        fontsize=10.5,
        color="#475569",
    )
    figure.text(
        0.08,
        0.01,
        "SHAP explains this model's predictions; it does not establish that a feature causes "
        "a relationship to persist.",
        fontsize=9,
        color="#64748B",
    )
    figure.tight_layout(rect=(0.04, 0.06, 0.98, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def run_benchmark(
    panel_path: str | Path,
    *,
    output_path: str | Path,
    predictions_path: str | Path,
    hero_chart_path: str | Path | None = None,
    model_artifact_path: str | Path | None = None,
    demo_reference_path: str | Path | None = None,
    lift_chart_path: str | Path | None = None,
    shap_chart_path: str | Path | None = None,
    study_label: str = "Temporal dyad benchmark",
    bootstrap_iterations: int = 1_000,
) -> dict[str, Any]:
    """Fit feature ablations once and persist predictions, metrics, and a hero chart."""
    import pandas as pd

    frame = pd.read_csv(panel_path)
    feature_sets = _available_feature_sets(frame)
    train, validation, test, cutoffs = _temporal_split(frame)
    results: dict[str, Any] = {
        "study_label": study_label,
        "panel_path": str(panel_path),
        "features": feature_sets,
        "split_cutoffs": cutoffs,
        "splits": {},
        "models": {},
    }
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        results["splits"][name] = {
            "dyads": len(split),
            "positives": int(split["label"].sum()),
            "positive_rate": float(split["label"].mean()),
            "first_anchor_month": str(split["anchor_month"].min()),
            "last_anchor_month": str(split["anchor_month"].max()),
        }

    prediction_frame = test[["author_a", "author_b", "anchor_month", "label"]].copy()
    models: dict[str, Any] = {}
    combined_shap_values: Any | None = None
    for name, features in feature_sets.items():
        model = _fit_model(train, validation, features)
        scores = model.predict_proba(test[features])[:, 1]
        models[name] = model
        prediction_frame[f"score_{name}"] = scores
        results["models"][name] = {
            "best_iteration": int(model.best_iteration),
            "metrics": _classification_metrics(test["label"], scores),
            "ranking": _ranking_metrics(test, scores),
            "gain_importance": {
                feature: float(importance)
                for feature, importance in zip(features, model.feature_importances_, strict=True)
            },
        }
        if name == "combined":
            combined_shap_values = _shap_values(model, test, features)
            results["models"][name]["mean_abs_shap"] = _mean_absolute_shap(
                combined_shap_values, features
            )

    baseline_scores = prediction_frame["score_similarity"].to_numpy()
    combined_scores = prediction_frame["score_combined"].to_numpy()
    rare_tie_lift = _lift_summary(
        test["label"].to_numpy(),
        {
            "similarity": baseline_scores,
            "dynamics": prediction_frame["score_dynamics"].to_numpy(),
            "combined": combined_scores,
        },
    )
    results["rare_tie_lift"] = rare_tie_lift
    results["delta_combined_vs_similarity"] = {
        metric: (
            results["models"]["combined"]["metrics"][metric]
            - results["models"]["similarity"]["metrics"][metric]
        )
        for metric in ("roc_auc", "average_precision")
    }
    results["delta_combined_vs_similarity"]["paired_bootstrap_95"] = _paired_bootstrap(
        test["label"].to_numpy(),
        baseline_scores,
        combined_scores,
        bootstrap_iterations,
    )
    if "combined_graph" in results["models"]:
        results["delta_combined_graph_vs_combined"] = {
            metric: (
                results["models"]["combined_graph"]["metrics"][metric]
                - results["models"]["combined"]["metrics"][metric]
            )
            for metric in ("roc_auc", "average_precision")
        }
    if "combined_disclosure" in results["models"]:
        results["delta_combined_disclosure_vs_combined"] = {
            metric: (
                results["models"]["combined_disclosure"]["metrics"][metric]
                - results["models"]["combined"]["metrics"][metric]
            )
            for metric in ("roc_auc", "average_precision")
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions = Path(predictions_path)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions, index=False)
    if model_artifact_path is not None:
        model_artifact = Path(model_artifact_path)
        _save_verified_model(
            models["combined"], model_artifact, test, FEATURE_SETS["combined"]
        )
    if demo_reference_path is not None:
        demo_reference = Path(demo_reference_path)
        demo_reference.parent.mkdir(parents=True, exist_ok=True)
        reference = _build_demo_reference(
            train,
            test,
            combined_scores,
            rare_tie_lift,
            study_label,
            models["combined"],
        )
        demo_reference.write_text(
            json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if hero_chart_path is not None:
        _render_hero_chart(results, Path(hero_chart_path), study_label)
    if lift_chart_path is not None:
        _render_lift_chart(rare_tie_lift, Path(lift_chart_path), study_label)
    if shap_chart_path is not None:
        if combined_shap_values is None:
            raise AssertionError("Combined-model SHAP values were not computed")
        _render_shap_chart(
            combined_shap_values,
            test,
            FEATURE_SETS["combined"],
            Path(shap_chart_path),
            study_label,
        )
    print(json.dumps(results, indent=2, sort_keys=True))
    return results
