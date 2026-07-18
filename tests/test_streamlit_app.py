from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_demo_renders_default_scenario() -> None:
    app = Path(__file__).resolve().parents[1] / "demo" / "app.py"

    rendered = AppTest.from_file(str(app), default_timeout=30).run()

    assert not rendered.exception
    assert [metric.label for metric in rendered.metric] == [
        "Model-score percentile",
        "Reference band",
        "Observed cohort rate",
    ]
    assert "not a personal probability" in " ".join(
        caption.value for caption in rendered.caption
    ).lower()
