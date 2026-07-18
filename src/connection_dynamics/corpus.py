"""Dependency-light readers for ConvoKit utterance archives."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True, slots=True)
class Utterance:
    utterance_id: str
    author: str | None
    conversation_id: str | None
    reply_to: str | None
    timestamp: int | None
    text: str
    subreddit: str | None


@contextmanager
def _open_jsonl(path: Path) -> Iterator[TextIO]:
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        candidates = [name for name in archive.namelist() if name.endswith("utterances.jsonl")]
        if len(candidates) != 1:
            archive.close()
            raise ValueError(f"Expected one utterances.jsonl in {path}, found {candidates}")
        raw = archive.open(candidates[0], "r")
        import io

        text = io.TextIOWrapper(raw, encoding="utf-8")
        try:
            yield text
        finally:
            text.close()
            archive.close()
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield handle


def iter_utterances(path: str | Path) -> Iterator[Utterance]:
    source = Path(path)
    with _open_jsonl(source) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {source} line {line_number}") from exc

            # ConvoKit's downloadable subreddit corpora use ``user``/``root``;
            # newer exports and synthetic fixtures commonly use
            # ``speaker``/``conversation_id``. Accept both without requiring
            # the full ConvoKit dependency.
            speaker = row.get("speaker", row.get("user"))
            if isinstance(speaker, dict):
                author = speaker.get("id")
            else:
                author = speaker
            metadata = row.get("meta") or {}
            yield Utterance(
                utterance_id=str(row["id"]),
                author=None if author is None else str(author),
                conversation_id=(
                    None
                    if row.get("conversation_id", row.get("root")) is None
                    else str(row.get("conversation_id", row.get("root")))
                ),
                reply_to=None if row.get("reply_to") is None else str(row["reply_to"]),
                timestamp=None if row.get("timestamp") is None else int(row["timestamp"]),
                text=str(row.get("text") or ""),
                subreddit=metadata.get("subreddit"),
            )
