# Leakage audit

1. A dyad's timestamp origin is its first observed direct reply.
2. Profile-similarity predictors read only `[day -30, day 0)` activity.
3. Interaction-dynamics predictors read only `[day 0, day 30)` replies.
4. Labels read only `[day 90, day 180)` direct replies.
5. The 60-day gap is ignored and never used as a predictor.
6. Dyads without complete 180-day corpus follow-up are excluded.
7. Negative labels require both authors to remain active in the outcome window.
8. One-directional outcome interaction is censored rather than coded as durable mutual interaction.
9. Model fitting, early stopping, and class weighting use training/validation data only.
10. Annual node2vec snapshots contain only dyads anchored before January 1 of the target year.
11. Author IDs, comment IDs, dyad IDs, and target-derived fields are prohibited predictors.
12. Earliest 70% of dyads train the model, the next 15% validate, and the last 15% test.
13. Paired bootstrap samples the same test rows for both ablations.
14. Annotation manifests contain no label or outcome fields; annotation completeness is required.

The small Cornell feasibility run exposed and corrected a schema mismatch and a baseline-window flaw.
The larger ApplyingToCollege test was evaluated after those definitions were frozen. Later reruns only
changed pseudonyms, aggregate SHAP output, and chart layout; the features, cutoffs, model settings, and
scores remained unchanged.
