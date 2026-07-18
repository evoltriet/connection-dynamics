from __future__ import annotations

import json
from pathlib import Path

from connection_dynamics.corpus import iter_utterances
from connection_dynamics.panel import DAY, build_panel, collect_events, write_panel


def _row(
    utterance_id: str,
    author: str,
    timestamp: int,
    reply_to: str | None = None,
    text: str = "message",
) -> dict:
    return {
        "id": utterance_id,
        "speaker": author,
        "conversation_id": "conversation",
        "reply_to": reply_to,
        "timestamp": timestamp,
        "text": text,
        "meta": {"subreddit": "Synthetic"},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_mutual_outcome_is_positive_and_uses_exposure_only(tmp_path: Path) -> None:
    corpus = tmp_path / "utterances.jsonl"
    rows = [
        _row("root", "alice", 0, text="I like hiking and jazz"),
        _row("first", "bob", 10, "root", "Me too, hiking is great"),
        _row("exposure", "alice", 29 * DAY, "first", "Want a hiking recommendation?"),
        _row("gap", "bob", 60 * DAY, "exposure", "This must not be a feature"),
        _row("future_a", "alice", 90 * DAY + 10, "gap", "Checking in"),
        _row("future_b", "bob", 100 * DAY + 10, "future_a", "Good to hear from you"),
        _row("end", "carol", 200 * DAY),
    ]
    _write(corpus, rows)

    panel, summary = build_panel([corpus])

    target = next(row for row in panel if {row.author_a, row.author_b} == {"alice", "bob"})
    assert target.label == 1
    assert target.exposure_reply_count == 2
    assert target.lexical_topic_overlap == 0.0
    assert target.shared_subreddit_jaccard == 1.0
    assert summary["label_1"] >= 1


def test_inactive_negative_is_censored(tmp_path: Path) -> None:
    corpus = tmp_path / "utterances.jsonl"
    rows = [
        _row("root", "alice", 0),
        _row("first", "bob", 1, "root"),
        _row("only_alice_active", "alice", 100 * DAY),
        _row("end", "carol", 200 * DAY),
    ]
    _write(corpus, rows)

    panel, summary = build_panel([corpus])

    assert not any({row.author_a, row.author_b} == {"alice", "bob"} for row in panel)
    assert summary["censored_inactive_author"] == 1


def test_active_noninteracting_pair_is_negative(tmp_path: Path) -> None:
    corpus = tmp_path / "utterances.jsonl"
    rows = [
        _row("root", "alice", 0),
        _row("first", "bob", 1, "root"),
        _row("alice_active", "alice", 100 * DAY),
        _row("bob_active", "bob", 110 * DAY),
        _row("end", "carol", 200 * DAY),
    ]
    _write(corpus, rows)

    panel, summary = build_panel([corpus])

    target = next(row for row in panel if {row.author_a, row.author_b} == {"alice", "bob"})
    assert target.label == 0
    assert summary["label_0"] >= 1


def test_reads_downloadable_convokit_subreddit_schema(tmp_path: Path) -> None:
    corpus = tmp_path / "utterances.jsonl"
    rows = [
        {
            "id": "comment",
            "user": "alice",
            "root": "post",
            "reply_to": "post",
            "timestamp": 123,
            "text": "hello",
            "meta": {"subreddit": "Cornell"},
        }
    ]
    _write(corpus, rows)

    utterance = next(iter_utterances(corpus))

    assert utterance.author == "alice"
    assert utterance.conversation_id == "post"
    assert utterance.reply_to == "post"


def test_written_panel_uses_stable_pseudonyms(tmp_path: Path) -> None:
    corpus = tmp_path / "utterances.jsonl"
    rows = [
        _row("root", "alice", 0),
        _row("first", "bob", 1, "root"),
        _row("alice_active", "alice", 100 * DAY),
        _row("bob_active", "bob", 110 * DAY),
        _row("end", "carol", 200 * DAY),
    ]
    _write(corpus, rows)
    panel, _ = build_panel([corpus])
    output = tmp_path / "panel.csv"

    write_panel(panel, output, "test-secret")

    content = output.read_text(encoding="utf-8")
    assert "alice" not in content
    assert "bob" not in content
    assert "user_" in content
    assert "dyad_" in content


def test_reply_ids_are_namespaced_by_corpus(tmp_path: Path) -> None:
    first_corpus = tmp_path / "first.jsonl"
    second_corpus = tmp_path / "second.jsonl"
    _write(
        first_corpus,
        [
            _row("root", "alice", 0),
            _row("reply", "bob", 1, "root"),
        ],
    )
    _write(
        second_corpus,
        [
            _row("root", "carol", 0),
            _row("reply", "dave", 1, "root"),
        ],
    )

    replies, _, _, _ = collect_events([first_corpus, second_corpus])

    assert set(replies) == {("alice", "bob"), ("carol", "dave")}
