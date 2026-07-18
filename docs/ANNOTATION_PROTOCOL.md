# Behavioral annotation protocol

Status: **prepared, not yet used in the reported model**.

The outcome-blinded ApplyingToCollege manifest contains 216,456 first-month reply records covering all
128,604 eligible dyads. Counts reconcile exactly with `exposure_reply_count`. The manifest contains no
outcomes or structured author fields and is stored only under `artifacts/private/`.

## Disclosure depth

- `0`: no personal disclosure.
- `1`: low-stakes personal fact, preference, or routine.
- `2`: personal experience, emotion, difficulty, uncertainty, or meaningful aspiration.
- `3`: sensitive fear, pain, shame, identity, or other high-stakes vulnerability.

## Supportive response

Evaluated only when the parent message has disclosure depth 2 or 3.

- `1`: acknowledges, validates, empathizes, encourages, helps, or responds with care.
- `0`: dismisses, attacks, ignores, exploits, or redirects away from the disclosure.

## Derived dyad features

- Mean disclosure depth
- Directional disclosure-depth balance
- Disclosure reciprocity after a vulnerable parent message
- Supportive-response rate after a vulnerable parent message

Missing qualifying disclosures remain missing, not zero. The model command refuses a confirmatory
disclosure ablation unless annotation coverage is 100% for every dyad.

## Provider and validation

The CLI uses structured outputs with a versioned schema and supports the OpenAI API or an
OpenAI-compatible base URL. Outputs record provider, model, rubric version, and confidence. Annotation
is resumable and does not write source text into the scored output.

Before disclosure features are described as confirmatory, two reviewers should independently label a
blinded, stratified 400-message audit sample. Report weighted agreement for depth, agreement/F1 for
support, and performance by disclosure class. No claim that disclosure reciprocity is the strongest
signal is valid until that audit and the held-out model are complete.
