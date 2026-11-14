"""Tests for the conversation agent's turn loop.

These need Home Assistant installed, since the agent subclasses Home
Assistant's ``AbstractConversationAgent``. They skip cleanly when it is absent,
so ``pytest`` still works in a bare virtualenv (see ``tests/test_tools.py``).

The registry glue is stubbed out with a pre-built :class:`Toolset`; what is
under test here is the loop itself: service dispatch, the confidence
confirmation handshake, the text fallback and error containment.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("homeassistant")

try:
    from homeassistant.components import conversation
    from homeassistant.core import Context
except Exception:  # noqa: BLE001 - optional HA extras may be missing
    pytest.skip("Home Assistant conversation component is unavailable", allow_module_level=True)

from custom_components.cactus_home.conversation import CactusConversationAgent  # noqa: E402
from custom_components.cactus_home.tools import (  # noqa: E402
    CAP_BRIGHTNESS,
    CAP_QUERY,
    CAP_STATE,
    DeviceTarget,
    build_toolset,
)

OPTIONS = {
    "max_steps": 4,
    "min_confidence": 0.4,
    "target_cache_ttl": 5.0,
    "enabled_domains": ["light"],
    "assist_only": False,
    "max_targets": 50,
    "max_enum_names": 64,
}


class FakeState:
    def __init__(self, state="on", attributes=None):
        self.state = state
        self.attributes = attributes or {}


class FakeStates:
    def __init__(self, states=None):
        self._states = states or {}

    def get(self, entity_id):
        return self._states.get(entity_id)


class FakeServices:
    def __init__(self):
        self.calls = []
        self.fail = None

    async def async_call(self, domain, service, data=None, blocking=False, context=None, return_response=False):
        if self.fail is not None:
            raise self.fail
        self.calls.append((domain, service, dict(data or {})))


class FakeHass:
    def __init__(self, states=None):
        self.states = FakeStates(states)
        self.services = FakeServices()

    async def async_add_executor_job(self, fn, *args):
        return fn(*args)


class FakeRuntime:
    """Stands in for CactusTextAgent: canned turns, recorded input."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []
        self.schema_calls = []
        self.resets = 0

    def complete(self, text, *, schemas, signature, system=None):
        self.seen.append(text)
        self.schema_calls.append(list(schemas))
        if self.turns:
            return self.turns.pop(0)
        return {"type": "respond", "function_calls": []}

    def reset(self):
        self.resets += 1

    def close(self):
        pass


def make_toolset():
    return build_toolset(
        [
            DeviceTarget(
                target_id="light.kitchen_light",
                name="Kitchen Light",
                domain="light",
                entity_ids=("light.kitchen_light",),
                capabilities=frozenset({CAP_STATE, CAP_BRIGHTNESS, CAP_QUERY}),
            )
        ]
    )


def make_agent(runtime, states=None, *, stub_toolset=True):
    hass = FakeHass(states or {"light.kitchen_light": FakeState("on")})
    agent = CactusConversationAgent(hass, None, dict(OPTIONS), runtime)
    if stub_toolset:
        toolset = make_toolset()
        agent._build_toolset = lambda: toolset  # bypass the registry glue
    return agent, hass


def user_input(text, conversation_id="conv-1", language="en"):
    return conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id=conversation_id,
        device_id=None,
        satellite_id=None,
        language=language,
        agent_id="test-agent",
    )


def spoken(result):
    return result.response.speech["plain"]["speech"]


CALL_TURN = {
    "type": "call",
    "function_calls": [{"name": "set_state", "arguments": {"device": "Kitchen Light", "state": "on"}}],
    "confidence": 0.92,
    "reasoning": "kitchen light -> device",
}
RESPOND_TURN = {"type": "respond", "function_calls": []}


def test_successful_turn_calls_the_service_and_speaks():
    runtime = FakeRuntime([CALL_TURN, RESPOND_TURN])
    agent, hass = make_agent(runtime)

    result = asyncio.run(agent.async_process(user_input("turn on the kitchen light")))

    assert hass.services.calls == [
        ("light", "turn_on", {"entity_id": ["light.kitchen_light"]})
    ]
    assert spoken(result) == "Turned on Kitchen Light."
    assert result.continue_conversation is False
    assert result.conversation_id == "conv-1"


def test_result_is_fed_back_to_the_model():
    runtime = FakeRuntime([CALL_TURN, RESPOND_TURN])
    agent, _ = make_agent(runtime)

    asyncio.run(agent.async_process(user_input("turn on the kitchen light")))

    assert runtime.seen[0] == "turn on the kitchen light"
    assert "executed" in runtime.seen[1]


def test_low_confidence_asks_before_acting_then_honours_yes():
    runtime = FakeRuntime([{**CALL_TURN, "confidence": 0.15}, RESPOND_TURN])
    agent, hass = make_agent(runtime)

    async def scenario():
        return (
            await agent.async_process(user_input("maybe turn on the kitchen light")),
            await agent.async_process(user_input("yes")),
        )

    first, second = asyncio.run(scenario())

    assert "Should I do that?" in spoken(first)
    assert first.continue_conversation is True
    assert hass.services.calls == [("light", "turn_on", {"entity_id": ["light.kitchen_light"]})]
    assert second.continue_conversation is False


def test_confirmation_can_be_declined():
    runtime = FakeRuntime([{**CALL_TURN, "confidence": 0.15}])
    agent, hass = make_agent(runtime)

    async def scenario():
        await agent.async_process(user_input("maybe turn on the kitchen light"))
        return await agent.async_process(user_input("no"))

    second = asyncio.run(scenario())

    assert "cancelled" in spoken(second)
    assert hass.services.calls == []


def test_withheld_call_also_asks_for_confirmation():
    runtime = FakeRuntime([{"type": "call", "function_calls": [], "suppressed_calls": CALL_TURN["function_calls"]}])
    agent, hass = make_agent(runtime)

    result = asyncio.run(agent.async_process(user_input("hmm the kitchen light perhaps")))

    assert "Did you want to" in spoken(result)
    assert result.continue_conversation is True
    assert hass.services.calls == []


def test_no_tool_call_falls_back_to_text_parsing():
    runtime = FakeRuntime([{"type": "call", "function_calls": []}])
    agent, hass = make_agent(runtime)

    result = asyncio.run(agent.async_process(user_input("turn on the Kitchen Light")))

    assert hass.services.calls == [("light", "turn_on", {"entity_id": ["light.kitchen_light"]})]
    assert spoken(result) == "Turned on Kitchen Light."


def test_unparseable_text_is_refused_not_raised():
    runtime = FakeRuntime([{"type": "call", "function_calls": []}])
    agent, hass = make_agent(runtime)

    result = asyncio.run(agent.async_process(user_input("what is the meaning of life")))

    assert "could not connect that" in spoken(result)
    assert hass.services.calls == []


def test_unknown_device_is_explained_not_raised():
    runtime = FakeRuntime(
        [
            {
                "type": "call",
                "function_calls": [{"name": "set_state", "arguments": {"device": "Narnia Lamp", "state": "on"}}],
                "confidence": 0.9,
            }
        ]
    )
    agent, hass = make_agent(runtime)

    result = asyncio.run(agent.async_process(user_input("turn on the narnia lamp")))

    assert "could not find a device" in spoken(result)
    assert hass.services.calls == []


def test_service_failure_is_spoken():
    from homeassistant.exceptions import HomeAssistantError

    runtime = FakeRuntime([CALL_TURN, RESPOND_TURN])
    agent, hass = make_agent(runtime)
    hass.services.fail = HomeAssistantError("device offline")

    result = asyncio.run(agent.async_process(user_input("turn on the kitchen light")))

    assert "device offline" in spoken(result)
    assert result.continue_conversation is False


def test_empty_toolset_is_explained():
    runtime = FakeRuntime([])
    agent, hass = make_agent(runtime)
    agent._build_toolset = lambda: build_toolset([])

    result = asyncio.run(agent.async_process(user_input("turn on the light")))

    assert "exposed to voice control" in spoken(result)
    assert hass.services.calls == []


def test_conversation_change_resets_the_model_context():
    runtime = FakeRuntime([RESPOND_TURN, RESPOND_TURN])
    agent, _ = make_agent(runtime)

    async def scenario():
        await agent.async_process(user_input("hello", conversation_id="conv-a"))
        await agent.async_process(user_input("hello again", conversation_id="conv-b"))

    asyncio.run(scenario())

    assert runtime.resets == 1  # first turn sets the id, the second resets


def test_get_state_answers_from_the_state_machine():
    runtime = FakeRuntime(
        [
            {
                "type": "call",
                "function_calls": [{"name": "get_state", "arguments": {"device": "Kitchen Light"}}],
                "confidence": 0.9,
            },
            RESPOND_TURN,
        ]
    )
    agent, hass = make_agent(runtime, {"light.kitchen_light": FakeState("on")})

    result = asyncio.run(agent.async_process(user_input("is the kitchen light on")))

    assert spoken(result) == "Kitchen Light is on."
    assert hass.services.calls == []


def test_supported_languages_are_the_model_languages():
    agent, _ = make_agent(FakeRuntime([]))
    assert agent.supported_languages == ["en", "de", "fr", "es", "it", "nl", "pl"]


# --------------------------------------------------------------------------
# Spoken numbers
# --------------------------------------------------------------------------
def test_spoken_number_words_reach_the_model_as_digits():
    runtime = FakeRuntime([RESPOND_TURN])
    agent, _ = make_agent(runtime)

    asyncio.run(
        agent.async_process(
            user_input("stelle das licht auf fünfzig Prozent", language="de")
        )
    )

    assert runtime.seen[0] == "stelle das licht auf 50 Prozent"


def test_number_normalisation_is_language_scoped():
    runtime = FakeRuntime([RESPOND_TURN])
    agent, _ = make_agent(runtime)

    # An English pipeline must not convert German number words.
    asyncio.run(agent.async_process(user_input("fünfzig Prozent", language="en")))

    assert runtime.seen[0] == "fünfzig Prozent"


# --------------------------------------------------------------------------
# Engine error: the toolset is part of the prompt
# --------------------------------------------------------------------------
CONTEXT_ERROR_TURN = {
    "type": "call",
    "error": "context length exceeded",
    "error_code": "context_length",
    "function_calls": [],
}


def device_enums(schemas):
    return [
        schema["parameters"]["properties"]["device"].get("enum")
        for schema in schemas
        if "device" in schema["parameters"]["properties"]
    ]


def test_an_engine_error_retries_once_without_name_lists(monkeypatch):
    import custom_components.cactus_home.conversation as conversation_module

    monkeypatch.setattr(
        conversation_module,
        "async_collect_targets",
        lambda hass, options: [
            DeviceTarget(
                target_id="light.kitchen_light",
                name="Kitchen Light",
                domain="light",
                entity_ids=("light.kitchen_light",),
                capabilities=frozenset({CAP_STATE, CAP_BRIGHTNESS, CAP_QUERY}),
            )
        ],
    )

    runtime = FakeRuntime([CONTEXT_ERROR_TURN, CALL_TURN, RESPOND_TURN])
    agent, hass = make_agent(runtime, stub_toolset=False)

    result = asyncio.run(agent.async_process(user_input("turn on the kitchen light")))

    # The first toolset carried name lists, the retry carried none.
    assert len(runtime.schema_calls) == 3
    assert all(enum for enum in device_enums(runtime.schema_calls[0]))
    assert all(enum is None for enum in device_enums(runtime.schema_calls[1]))

    assert agent._enums_disabled is True
    assert hass.services.calls == [("light", "turn_on", {"entity_id": ["light.kitchen_light"]})]
    assert spoken(result) == "Turned on Kitchen Light."


def test_a_persistent_engine_error_explains_the_context_budget():
    runtime = FakeRuntime([CONTEXT_ERROR_TURN, CONTEXT_ERROR_TURN])
    agent, hass = make_agent(runtime)

    result = asyncio.run(agent.async_process(user_input("turn on the kitchen light")))

    assert "context length" in spoken(result)
    assert "device name budget" in spoken(result)
    assert hass.services.calls == []
