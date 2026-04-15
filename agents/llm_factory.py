from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic

logger = logging.getLogger("llm_factory")

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_TEMPERATURE = 0.0


def create_chat_anthropic(
    model: str = DEFAULT_ANTHROPIC_MODEL,
    temperature: float = DEFAULT_ANTHROPIC_TEMPERATURE,
) -> ChatAnthropic:
    try:
        return ChatAnthropic(model=model, temperature=temperature)
    except Exception as exc:
        logger.exception("Failed to construct ChatAnthropic (model=%s)", model)
        raise RuntimeError(
            "Could not initialize ChatAnthropic. "
            "Check ANTHROPIC_API_KEY and that the model id is valid for your account."
        ) from exc
