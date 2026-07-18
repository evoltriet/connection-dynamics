"""Construct temporal dyad labels and exposure-only behavioral features."""

from __future__ import annotations

import bisect
import csv
import hashlib
import hmac
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .corpus import iter_utterances

DAY = 86_400
PROFILE_LOOKBACK = 30 * DAY
EXPOSURE_END = 30 * DAY
OUTCOME_START = 90 * DAY
OUTCOME_END = 180 * DAY
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
INVALID_AUTHORS = {"[deleted]", "[removed]", "automoderator"}


@dataclass(frozen=True, slots=True)
class ReplyEvent:
    corpus: str
    utterance_id: str
    parent_id: str
    author: str
    parent_author: str
    timestamp: int
    parent_timestamp: int
    subreddit: str
    text: str


@dataclass(frozen=True, slots=True)
class DyadRow:
    author_a: str
    author_b: str
    anchor_timestamp: int
    anchor_month: str
    subreddit: str
    label: int
    exposure_reply_count: int
    replies_a_to_b: int
    replies_b_to_a: int
    reciprocity_balance: float
    latency_symmetry: float | None
    effort_balance: float
    active_day_fraction: float
    recency_days: float
    author_activity_balance: float
    shared_subreddit_jaccard: float
    lexical_topic_overlap: float


def _valid_author(author: str | None) -> bool:
    return bool(author and author.lower() not in INVALID_AUTHORS)


def _pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _balance(left: float, right: float) -> float:
    total = left + right
    return 0.0 if total == 0 else 2.0 * min(left, right) / total


def _ratio_similarity(left: float, right: float) -> float:
    maximum = max(left, right)
    return 1.0 if maximum == 0 else min(left, right) / maximum


def _set_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def _anchor_month(timestamp: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m")


def _count_between(sorted_times: list[int], start: int, end: int) -> int:
    return bisect.bisect_left(sorted_times, end) - bisect.bisect_left(sorted_times, start)


def _activity_between(
    activity: list[tuple[int, str, str]], start: int, end: int
) -> list[tuple[int, str, str]]:
    left = bisect.bisect_left(activity, (start, "", ""))
    right = bisect.bisect_left(activity, (end, "", ""))
    return activity[left:right]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) >= 3}


def collect_events(
    corpus_paths: Iterable[str | Path],
) -> tuple[
    dict[tuple[str, str], list[ReplyEvent]],
    dict[str, list[int]],
    dict[str, list[tuple[int, str, str]]],
    int,
]:
    paths = [Path(path) for path in corpus_paths]
    utterance_index: dict[tuple[Path, str], tuple[str | None, int | None]] = {}
    corpus_end = 0

    for path in paths:
        for utterance in iter_utterances(path):
            utterance_index[(path, utterance.utterance_id)] = (
                utterance.author,
                utterance.timestamp,
            )
            if utterance.timestamp is not None:
                corpus_end = max(corpus_end, utterance.timestamp)

    replies: dict[tuple[str, str], list[ReplyEvent]] = defaultdict(list)
    activity_times: dict[str, list[int]] = defaultdict(list)
    activity: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    for path in paths:
        for utterance in iter_utterances(path):
            if not _valid_author(utterance.author) or utterance.timestamp is None:
                continue
            author = str(utterance.author)
            subreddit = str(utterance.subreddit or path.stem.split(".")[0])
            activity_times[author].append(utterance.timestamp)
            activity[author].append((utterance.timestamp, subreddit, utterance.text))
            reply_key = (path, str(utterance.reply_to))
            if utterance.reply_to is None or reply_key not in utterance_index:
                continue
            parent_author, parent_timestamp = utterance_index[reply_key]
            if (
                not _valid_author(parent_author)
                or parent_author == author
                or parent_timestamp is None
                or utterance.timestamp < parent_timestamp
            ):
                continue
            pair = _pair(author, str(parent_author))
            replies[pair].append(
                ReplyEvent(
                    corpus=str(path),
                    utterance_id=utterance.utterance_id,
                    parent_id=str(utterance.reply_to),
                    author=author,
                    parent_author=str(parent_author),
                    timestamp=utterance.timestamp,
                    parent_timestamp=parent_timestamp,
                    subreddit=subreddit,
                    text=utterance.text,
                )
            )

    for values in replies.values():
        values.sort(key=lambda event: event.timestamp)
    for values in activity_times.values():
        values.sort()
    for values in activity.values():
        values.sort(key=lambda item: item[0])
    return replies, activity_times, activity, corpus_end


def build_panel(
    corpus_paths: Iterable[str | Path],
    *,
    exposure_sink: dict[tuple[str, str], list[ReplyEvent]] | None = None,
) -> tuple[list[DyadRow], dict[str, int]]:
    return _build_panel(corpus_paths, exposure_sink=exposure_sink)


def _build_panel(
    corpus_paths: Iterable[str | Path],
    *,
    exposure_sink: dict[tuple[str, str], list[ReplyEvent]] | None = None,
) -> tuple[list[DyadRow], dict[str, int]]:
    replies, activity_times, activity, corpus_end = collect_events(corpus_paths)
    rows: list[DyadRow] = []
    counters: dict[str, int] = defaultdict(int)

    for (author_a, author_b), events in replies.items():
        counters["observed_dyads"] += 1
        anchor = events[0].timestamp
        if anchor + OUTCOME_END > corpus_end:
            counters["censored_incomplete_followup"] += 1
            continue

        exposure = [event for event in events if anchor <= event.timestamp < anchor + EXPOSURE_END]
        outcome = [
            event
            for event in events
            if anchor + OUTCOME_START <= event.timestamp < anchor + OUTCOME_END
        ]
        future_a = any(event.author == author_a for event in outcome)
        future_b = any(event.author == author_b for event in outcome)

        if future_a and future_b:
            label = 1
        elif outcome:
            counters["censored_one_direction_outcome"] += 1
            continue
        else:
            a_active = _count_between(
                activity_times[author_a], anchor + OUTCOME_START, anchor + OUTCOME_END
            )
            b_active = _count_between(
                activity_times[author_b], anchor + OUTCOME_START, anchor + OUTCOME_END
            )
            if not (a_active and b_active):
                counters["censored_inactive_author"] += 1
                continue
            label = 0

        a_to_b = [event for event in exposure if event.author == author_a]
        b_to_a = [event for event in exposure if event.author == author_b]
        latency_a = [event.timestamp - event.parent_timestamp for event in a_to_b]
        latency_b = [event.timestamp - event.parent_timestamp for event in b_to_a]
        if latency_a and latency_b:
            median_a = max(statistics.median(latency_a), 1.0)
            median_b = max(statistics.median(latency_b), 1.0)
            latency_symmetry = math.exp(-abs(math.log(median_a) - math.log(median_b)))
        else:
            latency_symmetry = None

        effort_a = statistics.mean([len(_tokens(event.text)) for event in a_to_b]) if a_to_b else 0
        effort_b = statistics.mean([len(_tokens(event.text)) for event in b_to_a]) if b_to_a else 0
        active_days = {int((event.timestamp - anchor) // DAY) for event in exposure}
        last_exposure = max(event.timestamp for event in exposure)

        # Keep the comparison honest: profile similarity is measured before
        # first contact, while relationship dynamics are measured after it.
        profile_window: dict[str, list[tuple[int, str, str]]] = {}
        for author in (author_a, author_b):
            profile_window[author] = _activity_between(
                activity[author], anchor - PROFILE_LOOKBACK, anchor
            )
        encounter_subreddit = events[0].subreddit
        subreddits_a = {encounter_subreddit} | {item[1] for item in profile_window[author_a]}
        subreddits_b = {encounter_subreddit} | {item[1] for item in profile_window[author_b]}
        tokens_a = set().union(*(_tokens(item[2]) for item in profile_window[author_a]))
        tokens_b = set().union(*(_tokens(item[2]) for item in profile_window[author_b]))

        rows.append(
            DyadRow(
                author_a=author_a,
                author_b=author_b,
                anchor_timestamp=anchor,
                anchor_month=_anchor_month(anchor),
                subreddit=events[0].subreddit,
                label=label,
                exposure_reply_count=len(exposure),
                replies_a_to_b=len(a_to_b),
                replies_b_to_a=len(b_to_a),
                reciprocity_balance=_balance(len(a_to_b), len(b_to_a)),
                latency_symmetry=latency_symmetry,
                effort_balance=_ratio_similarity(effort_a, effort_b),
                active_day_fraction=len(active_days) / 30.0,
                recency_days=(anchor + EXPOSURE_END - last_exposure) / DAY,
                author_activity_balance=_ratio_similarity(
                    len(profile_window[author_a]), len(profile_window[author_b])
                ),
                shared_subreddit_jaccard=_set_jaccard(subreddits_a, subreddits_b),
                lexical_topic_overlap=_set_jaccard(tokens_a, tokens_b),
            )
        )
        if exposure_sink is not None:
            exposure_sink[(author_a, author_b)] = exposure
        counters[f"label_{label}"] += 1

    counters["eligible_dyads"] = len(rows)
    return rows, dict(counters)


def pseudonymize_author(author: str, hash_key: str) -> str:
    digest = hmac.new(hash_key.encode(), author.encode(), hashlib.sha256).hexdigest()
    return f"user_{digest[:16]}"


def pseudonymize_dyad(author_a: str, author_b: str, hash_key: str) -> str:
    left, right = _pair(author_a, author_b)
    digest = hmac.new(hash_key.encode(), f"{left}\0{right}".encode(), hashlib.sha256).hexdigest()
    return f"dyad_{digest[:20]}"


def write_panel(rows: Iterable[DyadRow], output: str | Path, hash_key: str) -> None:
    records = [asdict(row) for row in rows]
    if not records:
        raise ValueError("No eligible dyads were produced")
    if not hash_key:
        raise ValueError("A non-empty hash key is required before writing dyad identifiers")
    for record in records:
        author_a = str(record["author_a"])
        author_b = str(record["author_b"])
        record["author_a"] = pseudonymize_author(author_a, hash_key)
        record["author_b"] = pseudonymize_author(author_b, hash_key)
        record["dyad_id"] = pseudonymize_dyad(author_a, author_b, hash_key)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
