from __future__ import annotations

import random

from connection_dynamics.graph import _biased_walk


def test_biased_walk_stays_on_observed_edges() -> None:
    adjacency = {
        "alice": {"bob": 1.0},
        "bob": {"alice": 1.0, "carol": 1.0},
        "carol": {"bob": 1.0},
    }

    walk = _biased_walk(
        adjacency,
        "alice",
        length=10,
        return_parameter=1.0,
        inout_parameter=0.5,
        random_state=random.Random(42),
    )

    assert walk[0] == "alice"
    assert len(walk) == 10
    assert all(
        right in adjacency[left] for left, right in zip(walk[:-1], walk[1:], strict=True)
    )
