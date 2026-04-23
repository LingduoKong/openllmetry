"""Unit tests for OpenAI Agents GenAI event emission helpers."""

from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.instrumentation.openai_agents.config import Config
from opentelemetry.instrumentation.openai_agents.event_emitter import emit_event
from opentelemetry.instrumentation.openai_agents.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _logger():
    logger = MagicMock()
    logger.emit = MagicMock()
    return logger


def _emitted_event(logger):
    return logger.emit.call_args[0][0]


@pytest.fixture(autouse=True)
def reset_event_logger():
    Config.event_logger = None
    yield
    Config.event_logger = None


def test_emit_event_returns_without_emitting_when_events_disabled():
    logger = _logger()
    Config.event_logger = logger

    with patch(
        "opentelemetry.instrumentation.openai_agents.event_emitter.should_emit_events",
        return_value=False,
    ):
        emit_event(MessageEvent(content="hello", role="user"))

    logger.emit.assert_not_called()


def test_message_event_emits_user_content():
    logger = _logger()
    Config.event_logger = logger

    with patch(
        "opentelemetry.instrumentation.openai_agents.event_emitter.should_emit_events",
        return_value=True,
    ), patch(
        "opentelemetry.instrumentation.openai_agents.event_emitter.should_send_prompts",
        return_value=True,
    ):
        emit_event(MessageEvent(content="hello", role="user"))

    event = _emitted_event(logger)
    assert event.name == "gen_ai.user.message"
    assert event.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
    assert event.body == {"content": "hello"}


def test_message_event_redacts_content_and_tool_arguments():
    logger = _logger()
    Config.event_logger = logger

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "function_name": "get_weather",
                "arguments": {"city": "London"},
            },
        }
    ]

    with patch(
        "opentelemetry.instrumentation.openai_agents.event_emitter.should_emit_events",
        return_value=True,
    ), patch(
        "opentelemetry.instrumentation.openai_agents.event_emitter.should_send_prompts",
        return_value=False,
    ):
        emit_event(
            MessageEvent(content="tool call", role="assistant", tool_calls=tool_calls)
        )

    event = _emitted_event(logger)
    assert event.name == "gen_ai.assistant.message"
    assert "content" not in event.body
    assert "arguments" not in event.body["tool_calls"][0]["function"]


def test_choice_event_moves_content_to_output_message():
    logger = _logger()
    Config.event_logger = logger

    with patch(
        "opentelemetry.instrumentation.openai_agents.event_emitter.should_emit_events",
        return_value=True,
    ), patch(
        "opentelemetry.instrumentation.openai_agents.event_emitter.should_send_prompts",
        return_value=True,
    ):
        emit_event(
            ChoiceEvent(
                index=0,
                message={"content": "answer", "role": "assistant"},
                finish_reason="stop",
            )
        )

    event = _emitted_event(logger)
    assert event.name == "gen_ai.choice"
    assert event.body == {
        "index": 0,
        "finish_reason": "stop",
        "gen_ai.output.message": "answer",
    }
