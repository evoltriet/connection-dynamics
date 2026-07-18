# Candidate post-meeting questionnaire

This instrument translates the benchmark's logged interaction features into reflection prompts and
separately records the relationship constructs the broader thesis aims to test. Use a seven-point
agreement scale (`1 = strongly disagree`, `7 = strongly agree`) plus `Not enough information`.

Do not calculate a total score. Tier A self-reports have not been validated as substitutes for the
logged behavior used by the current model. Tier B items are candidate research questions, not
validated measures and not current model inputs.

## Tier A: behavioral proxies represented in the model

| Candidate item | Computational feature | Logged operationalization |
|---|---|---|
| We had repeated direct exchanges during the first month. | `exposure_reply_count` | Direct replies during days 0-30 |
| We interacted on many different days rather than in one short burst. | `active_day_fraction` | Distinct interaction days divided by 30 |
| Our interaction was still active near the end of the first month. | `recency_days` | Days between the last exposure reply and day 30 |
| We contributed a similar share of the replies. | `reciprocity_balance` | Balance of replies in the two directions |
| Our usual response times were roughly similar. | `latency_symmetry` | Similarity of the two median response latencies |
| We put similar detail and effort into our messages. | `effort_balance` | Similarity of mean reply token counts |
| Before meeting, we discussed similar topics. | `lexical_topic_overlap` | Pre-contact token-set Jaccard overlap |
| Before meeting, we participated in similar communities. | `shared_subreddit_jaccard` | Pre-contact community-set Jaccard overlap |
| Before meeting, we were similarly active. | `author_activity_balance` | Similarity of pre-contact activity counts |

The wording is intentionally understandable to a respondent, while the model uses objective logged
proxies. Agreement with an item must not be converted directly into the corresponding numeric model
feature.

## Tier B: exploratory thesis constructs

1. I felt safe expressing uncertainty or disagreement without expecting ridicule, rejection, or
   punishment.
2. When I shared something personal, this person responded in a way that made me feel understood,
   accepted, and cared for.
3. Personal sharing deepened in both directions rather than remaining one-sided.
4. We both initiated contact and made time for the interaction.
5. I felt able to be myself rather than perform.
6. This person brought out a version of me I want to become.
7. I felt accepted as I am and encouraged to grow.

These prompts represent emotional safety, perceived responsiveness, mutual vulnerability, reciprocal
investment, authenticity, self-expansion, and acceptance-plus-growth. They are hypotheses to test,
not properties demonstrated by the present Reddit benchmark.

## Validation path

Collect questionnaire responses during days 0-30 without exposing the outcome label. Measure the
same reciprocal days 90-180 outcome used by the benchmark, preregister item coding and exclusions,
and evaluate reliability before testing whether Tier B adds held-out predictive value beyond the
logged similarity and dynamics features. Human agreement auditing remains required before combining
questionnaire results with message-level disclosure annotations.
