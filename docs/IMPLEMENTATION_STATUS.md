# Implementation status

Last updated: 2026-07-18

## Complete

- Real ConvoKit parser with corpus-namespaced reply IDs
- Pre-contact similarity and first-month dynamics windows
- Active-negative censoring and strict 90–180-day reciprocal outcome
- Keyed HMAC pseudonyms and private artifact boundaries
- Chronological XGBoost ablations, paired bootstrap, ranking metrics, and held-out SHAP
- Public aggregate result and hero chart for 128,604 ApplyingToCollege dyads
- Prior-year node2vec snapshots with an honestly negative ablation
- Outcome-blinded annotation manifest for 216,456 messages
- Eight passing tests and clean Ruff output

## Credential-gated

The disclosure/support annotator is ready, but neither `OPENAI_API_KEY` nor a local compatible endpoint
was available. Run annotation only after configuring one of those providers, then complete the human
agreement audit before adding disclosure results to the README.

## Future work

- Run the ChangeMyView corpus with a tested scale-out runtime.
- Complete and validate disclosure annotation.
- Test temporal graph models only after historical graph coverage is adequate.
- Replicate across communities before making broad claims about human connection.
