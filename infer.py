"""CLI wrapper around RLKolAgent for quick testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference.agent import RLKolAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the RL KOL agent")
    parser.add_argument("--text", required=True, help="KOL text input")
    parser.add_argument("--market", required=True, help="Path to market JSON features")
    parser.add_argument("--model", default="models/checkpoints/kolA/policy.pt", help="Policy path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_state = json.loads(Path(args.market).read_text(encoding="utf-8"))
    agent = RLKolAgent(model_path=args.model)
    result = agent.predict(args.text, market_state)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
