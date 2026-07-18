"""Leakage-safe node2vec features from a fixed training-period graph snapshot."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Adjacency = dict[str, dict[str, float]]


def _build_adjacency(train: Any) -> Adjacency:
    adjacency: Adjacency = {}
    for row in train.itertuples(index=False):
        left = str(row.author_a)
        right = str(row.author_b)
        weight = math.log1p(float(row.exposure_reply_count))
        adjacency.setdefault(left, {})[right] = weight
        adjacency.setdefault(right, {})[left] = weight
    return adjacency


def _biased_walk(
    adjacency: Adjacency,
    start: str,
    *,
    length: int,
    return_parameter: float,
    inout_parameter: float,
    random_state: random.Random,
) -> list[str]:
    walk = [start]
    while len(walk) < length:
        current = walk[-1]
        candidates = list(adjacency.get(current, {}))
        if not candidates:
            break
        if len(walk) == 1:
            weights = [adjacency[current][candidate] for candidate in candidates]
        else:
            previous = walk[-2]
            previous_neighbors = adjacency.get(previous, {})
            weights = []
            for candidate in candidates:
                bias = 1.0
                if candidate == previous:
                    bias = 1.0 / return_parameter
                elif candidate not in previous_neighbors:
                    bias = 1.0 / inout_parameter
                weights.append(adjacency[current][candidate] * bias)
        walk.append(random_state.choices(candidates, weights=weights, k=1)[0])
    return walk


class _WalkCorpus:
    def __init__(
        self,
        adjacency: Adjacency,
        *,
        walk_length: int,
        walks_per_node: int,
        return_parameter: float,
        inout_parameter: float,
        seed: int,
    ) -> None:
        self.adjacency = adjacency
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.return_parameter = return_parameter
        self.inout_parameter = inout_parameter
        self.seed = seed

    def __iter__(self) -> Iterator[list[str]]:
        random_state = random.Random(self.seed)
        nodes = sorted(self.adjacency)
        for _ in range(self.walks_per_node):
            random_state.shuffle(nodes)
            for node in nodes:
                yield _biased_walk(
                    self.adjacency,
                    node,
                    length=self.walk_length,
                    return_parameter=self.return_parameter,
                    inout_parameter=self.inout_parameter,
                    random_state=random_state,
                )


def _edge_features(model: Any, left: str, right: str) -> tuple[float, float, float, int]:
    import numpy as np

    if left not in model.wv or right not in model.wv:
        return math.nan, math.nan, math.nan, 0
    left_vector = model.wv[left]
    right_vector = model.wv[right]
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    cosine = 0.0 if denominator == 0 else float(np.dot(left_vector, right_vector) / denominator)
    l2_similarity = float(math.exp(-float(np.linalg.norm(left_vector - right_vector))))
    hadamard_mean = float(np.mean(left_vector * right_vector))
    return cosine, l2_similarity, hadamard_mean, 1


def _fit_snapshot(
    adjacency: Adjacency,
    *,
    dimensions: int,
    walk_length: int,
    walks_per_node: int,
    return_parameter: float,
    inout_parameter: float,
    seed: int,
) -> Any:
    from gensim.models import Word2Vec

    corpus = _WalkCorpus(
        adjacency,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        return_parameter=return_parameter,
        inout_parameter=inout_parameter,
        seed=seed,
    )
    return Word2Vec(
        sentences=corpus,
        vector_size=dimensions,
        window=5,
        min_count=1,
        sg=1,
        negative=5,
        sample=0,
        workers=1,
        epochs=5,
        seed=seed,
    )


def run_node2vec_enrichment(
    panel_path: str | Path,
    *,
    output_path: str | Path,
    metadata_path: str | Path,
    dimensions: int = 16,
    walk_length: int = 12,
    walks_per_node: int = 3,
    return_parameter: float = 1.0,
    inout_parameter: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Add annual node2vec snapshots that always predate each target dyad."""
    import pandas as pd

    frame = pd.read_csv(panel_path)
    frame["_anchor_year"] = pd.to_datetime(frame["anchor_timestamp"], unit="s", utc=True).dt.year
    missing_features = (math.nan, math.nan, math.nan, 0)
    features: list[tuple[float, float, float, int]] = [missing_features] * len(frame)
    snapshots: list[dict[str, Any]] = []
    for year in sorted(int(value) for value in frame["_anchor_year"].unique()):
        cutoff = int(datetime(year, 1, 1, tzinfo=UTC).timestamp())
        history = frame[frame["anchor_timestamp"] < cutoff]
        targets = frame.index[frame["_anchor_year"] == year]
        if history.empty:
            snapshots.append(
                {
                    "year": year,
                    "cutoff": cutoff,
                    "historical_edges": 0,
                    "historical_nodes": 0,
                    "target_dyads": len(targets),
                    "coverage": 0.0,
                }
            )
            continue
        adjacency = _build_adjacency(history)
        model = _fit_snapshot(
            adjacency,
            dimensions=dimensions,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            return_parameter=return_parameter,
            inout_parameter=inout_parameter,
            seed=seed + year,
        )
        covered = 0
        for index in targets:
            row = frame.loc[index]
            edge_features = _edge_features(model, str(row["author_a"]), str(row["author_b"]))
            features[int(index)] = edge_features
            covered += edge_features[-1]
        snapshots.append(
            {
                "year": year,
                "cutoff": cutoff,
                "historical_edges": len(history),
                "historical_nodes": len(adjacency),
                "target_dyads": len(targets),
                "coverage": covered / len(targets),
            }
        )
    feature_frame = pd.DataFrame(
        features,
        columns=[
            "node2vec_cosine",
            "node2vec_l2_similarity",
            "node2vec_hadamard_mean",
            "node2vec_coverage",
        ],
    )
    enriched = pd.concat(
        [frame.drop(columns="_anchor_year").reset_index(drop=True), feature_frame], axis=1
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    metadata: dict[str, Any] = {
        "source_panel": str(panel_path),
        "output_panel": str(output_path),
        "fit_rule": "annual snapshots use only dyads anchored before January 1 of target year",
        "dimensions": dimensions,
        "walk_length": walk_length,
        "walks_per_node": walks_per_node,
        "return_parameter_p": return_parameter,
        "inout_parameter_q": inout_parameter,
        "seed": seed,
        "overall_coverage": float(feature_frame["node2vec_coverage"].mean()),
        "snapshots": snapshots,
    }
    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata
