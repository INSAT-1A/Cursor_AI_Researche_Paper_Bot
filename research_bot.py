from __future__ import annotations

import argparse
import json

from agents.orchestrator_agent import OrchestratorAgent
from logging_config import setup_logging


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 4-agent research bot.")
    parser.add_argument("query", type=str, help="Research query to investigate")
    return parser


def main() -> None:
    setup_logging()
    args = build_cli().parse_args()

    orchestrator = OrchestratorAgent()
    result = orchestrator.invoke(args.query)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
