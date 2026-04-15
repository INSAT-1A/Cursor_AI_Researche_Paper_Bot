from __future__ import annotations

import argparse
import json
import logging
import sys

from agents.orchestrator_agent import OrchestratorAgent
from logging_config import setup_logging

logger = logging.getLogger("research_bot")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 4-agent research bot.")
    parser.add_argument("query", type=str, help="Research query to investigate")
    return parser


def main() -> int:
    setup_logging()
    args = build_cli().parse_args()

    try:
        orchestrator = OrchestratorAgent()
    except RuntimeError as exc:
        logger.error("%s", exc)
        print(json.dumps({"error": str(exc), "fatal": True}, indent=2), file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Failed to start orchestrator")
        print(json.dumps({"error": str(exc), "fatal": True}, indent=2), file=sys.stderr)
        return 1

    try:
        result = orchestrator.invoke(args.query)
    except Exception as exc:
        logger.exception("Orchestrator.invoke raised unexpectedly")
        print(json.dumps({"error": str(exc), "fatal": True}, indent=2), file=sys.stderr)
        return 1

    try:
        print(json.dumps(result.model_dump(), indent=2))
    except (TypeError, ValueError) as exc:
        logger.exception("Failed to serialize orchestration result")
        print(json.dumps({"error": str(exc), "fatal": True}, indent=2), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(130) from None
