"""CLI for one-off predictions against a local checkpoint.

Usage:
    python -m app.cli predict "SpaceX caught a Super Heavy booster mid-air on the first try."
    python -m app.cli predict --json "Some sentence"
    echo "Some sentence" | python -m app.cli predict -
"""

from __future__ import annotations

import argparse
import json
import sys

from app.predictor import Predictor


def cmd_predict(args) -> int:
    text = args.text
    if text == "-":
        text = sys.stdin.read().strip()
    if not text:
        print("error: no text provided", file=sys.stderr)
        return 2
    predictor = Predictor()
    pred = predictor.predict(text)
    if args.json:
        print(json.dumps(pred.to_dict(), indent=2))
    else:
        verdict = "CLAIM" if pred.is_claim else "NOT A CLAIM"
        print(f"{verdict}  (confidence: {pred.confidence:.4f})")
        print(f"text: {text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim-detect")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="Predict a single sentence.")
    p.add_argument("text", help='Sentence to classify, or "-" to read stdin.')
    p.add_argument("--json", action="store_true", help="Output JSON.")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
