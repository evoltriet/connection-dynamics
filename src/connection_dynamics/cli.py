"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .annotation import (
    annotate_manifest,
    enrich_panel_with_annotations,
    export_annotation_manifest,
    openai_provider_from_environment,
)
from .graph import run_node2vec_enrichment
from .modeling import run_benchmark
from .panel import build_panel, write_panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="connection-dynamics")
    commands = parser.add_subparsers(dest="command", required=True)
    panel = commands.add_parser("build-panel", help="Build an exposure-only dyad panel")
    panel.add_argument("corpora", nargs="+", type=Path)
    panel.add_argument("--output", type=Path, required=True)
    panel.add_argument("--summary", type=Path)
    panel.add_argument(
        "--hash-key-env",
        default="CONNECTION_DYNAMICS_HASH_KEY",
        help="Environment variable containing the private HMAC key for pseudonymous author IDs",
    )
    benchmark = commands.add_parser("benchmark", help="Run temporal XGBoost ablations")
    benchmark.add_argument("panel", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--predictions", type=Path, required=True)
    benchmark.add_argument("--hero-chart", type=Path)
    benchmark.add_argument("--study-label", default="Temporal dyad benchmark")
    benchmark.add_argument("--bootstrap", type=int, default=1_000)
    graph = commands.add_parser(
        "graph-features", help="Add train-snapshot node2vec dyad features"
    )
    graph.add_argument("panel", type=Path)
    graph.add_argument("--output", type=Path, required=True)
    graph.add_argument("--metadata", type=Path, required=True)
    graph.add_argument("--dimensions", type=int, default=16)
    graph.add_argument("--walk-length", type=int, default=12)
    graph.add_argument("--walks-per-node", type=int, default=3)
    export = commands.add_parser(
        "export-annotations", help="Export an outcome-blinded private annotation manifest"
    )
    export.add_argument("corpora", nargs="+", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--metadata", type=Path, required=True)
    export.add_argument("--hash-key-env", default="CONNECTION_DYNAMICS_HASH_KEY")
    export.add_argument("--max-dyads", type=int)
    export.add_argument("--mutual-only", action="store_true")
    annotate = commands.add_parser(
        "annotate", help="Run resumable structured behavioral annotation"
    )
    annotate.add_argument("manifest", type=Path)
    annotate.add_argument("--output", type=Path, required=True)
    annotate.add_argument("--provider", choices=["openai"], default="openai")
    annotate.add_argument("--model", default="gpt-5.6-luna")
    annotate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    annotate.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    annotate.add_argument("--batch-size", type=int, default=20)
    enrich = commands.add_parser(
        "enrich-annotations", help="Join complete message annotations to a dyad panel"
    )
    enrich.add_argument("panel", type=Path)
    enrich.add_argument("annotations", type=Path)
    enrich.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-panel":
        rows, summary = build_panel(args.corpora)
        hash_key = os.environ.get(args.hash_key_env)
        if not hash_key:
            raise SystemExit(f"Set {args.hash_key_env} before writing pseudonymous dyad IDs")
        write_panel(rows, args.output, hash_key)
        payload = json.dumps(summary, indent=2, sort_keys=True)
        print(payload)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(payload + "\n", encoding="utf-8")
        return 0
    if args.command == "benchmark":
        run_benchmark(
            args.panel,
            output_path=args.output,
            predictions_path=args.predictions,
            hero_chart_path=args.hero_chart,
            study_label=args.study_label,
            bootstrap_iterations=args.bootstrap,
        )
        return 0
    if args.command == "graph-features":
        run_node2vec_enrichment(
            args.panel,
            output_path=args.output,
            metadata_path=args.metadata,
            dimensions=args.dimensions,
            walk_length=args.walk_length,
            walks_per_node=args.walks_per_node,
        )
        return 0
    if args.command == "export-annotations":
        hash_key = os.environ.get(args.hash_key_env)
        if not hash_key:
            raise SystemExit(f"Set {args.hash_key_env} before exporting pseudonymous records")
        export_annotation_manifest(
            args.corpora,
            output_path=args.output,
            metadata_path=args.metadata,
            hash_key=hash_key,
            max_dyads=args.max_dyads,
            mutual_only=args.mutual_only,
        )
        return 0
    if args.command == "annotate":
        provider = openai_provider_from_environment(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url_env=args.base_url_env,
        )
        annotate_manifest(
            args.manifest,
            output_path=args.output,
            provider=provider,
            batch_size=args.batch_size,
        )
        return 0
    if args.command == "enrich-annotations":
        enrich_panel_with_annotations(
            args.panel,
            args.annotations,
            output_path=args.output,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
