# Data card

## Source

The benchmark consumes ConvoKit's by-subreddit archives, derived from the historical Pushshift Reddit
corpus and documented as spanning each included community's inception through October 2018.

The headline archive is `ApplyingToCollege.corpus.zip` (155.3 MB compressed; 672.0 MB utterance
JSONL). The dependency-light panel build completed in 228 seconds on the development workstation.
Cornell is an 11.2 MB feasibility corpus. ChangeMyView is 1.27 GB compressed and is not part of the
reported result.

## Cohort flow

For ApplyingToCollege:

- 495,792 observed dyads
- 112,201 censored for incomplete follow-up
- 251,636 censored because at least one author was inactive in the outcome window
- 3,351 censored for one-directional outcome interaction
- 128,604 eligible dyads
- 2,369 durable reciprocal ties (1.84%)

## Privacy

The ingestion layer needs public Reddit usernames only to link messages. Derived panels replace them
with keyed HMAC pseudonyms before writing. The HMAC key is supplied through an environment variable
and is never committed. Raw archives, row-level panels, predictions, raw text, and annotation records
are gitignored. Public artifacts are limited to aggregate metrics and charts, an XGBoost tree model
containing no feature rows or identifiers, and a demo reference containing aggregate quantiles,
outcome rates, confidence intervals, and lift statistics.

The private annotation manifest contains public comment text and may contain links or self-identifying
statements inside that text. It contains no structured username fields and no outcomes, but it must
still be treated as sensitive research data and must not be published.

## Known limitations

- Corpus history is not guaranteed complete, and some reply targets are missing.
- Deleted authors and comments cannot form valid dyads.
- Continued Reddit interaction is an imperfect proxy for a durable relationship.
- Requiring active negatives changes the estimand to people who remain observable on the platform.
- The headline is one youth/education-oriented subreddit and may not generalize to other communities.
- A direct reply can reflect disagreement or conflict; persistence is not necessarily healthy connection.
