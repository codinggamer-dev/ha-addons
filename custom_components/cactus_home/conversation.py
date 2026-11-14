"""The Cactus conversation agent: Needle 3 as Home Assistant's brain.

The agent implements Home Assistant's ``AbstractConversationAgent`` protocol.
Each turn it:

1. snapshots the entities exposed to voice and compiles them into Needle tool
   schemas (see :mod:`.entities` and :mod:`.tools`);
2. decodes one agent turn with Needle, which returns a grammar-constrained tool
   call rather than free text;
3. resolves the spoken device name against the registry and calls the matching
   Home Assistant service;
4. feeds the result back to Needle so multi-step requests keep working, and
   finally composes a spoken reply.

Every blocking call into ``cactus`` happens in an executor, so the event loop
is never blocked. Every failure path ends in a spoken reply, never in an
uncaught exception.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Final

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent

from .const import (
    CONF_ASSIST_ONLY,
    CONF_ENABLED_DOMAINS,
    CONF_EXCLUDE_ENTITIES,
    CONF_LANGUAGE,
    CONF_MAX_ENUM_NAMES,
    CONF_MAX_STEPS,
    CONF_MAX_TARGETS,
    CONF_MAX_TOOL_NAMES,
    CONF_MIN_CONFIDENCE,
    CONF_NORMALIZE_NUMBERS,
    DEFAULT_MAX_ENUM_NAMES,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TARGETS,
    DEFAULT_MAX_TOOL_NAMES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NORMALIZE_NUMBERS,
    MODEL_NEEDLE,
    SUPPORTED_LANGUAGES,
)
from .entities import async_collect_targets
from .needle import CactusError, CactusTextAgent
from .numbers import normalize_spoken_numbers
from .tools import (
    Action,
    DeviceTarget,
    ToolError,
    Toolset,
    build_toolset,
    fallback_parse,
    plan_action,
)

_LOGGER = logging.getLogger(__name__)

_AFFIRMATIVE: Final = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "please do",
        "do it",
        "go ahead",
        "confirm",
        "correct",
        "affirmative",
        "that's right",
        "thats right",
    }
)
_NEGATIVE: Final = frozenset(
    {"no", "nope", "nah", "cancel", "stop", "never mind", "nevermind", "don't", "dont", "no thanks"}
)

_TENSE_FIXES: Final = (
    (re.compile(r"^turned on ", re.IGNORECASE), "turn on "),
    (re.compile(r"^turned off ", re.IGNORECASE), "turn off "),
    (re.compile(r"^locked ", re.IGNORECASE), "lock "),
    (re.compile(r"^unlocked ", re.IGNORECASE), "unlock "),
    (re.compile(r"^opened ", re.IGNORECASE), "open "),
    (re.compile(r"^closed ", re.IGNORECASE), "close "),
    (re.compile(r"^stopped ", re.IGNORECASE), "stop "),
    (re.compile(r"^playing ", re.IGNORECASE), "play "),
    (re.compile(r"^paused ", re.IGNORECASE), "pause "),
    (re.compile(r"^muted ", re.IGNORECASE), "mute "),
    (re.compile(r"^unmuted ", re.IGNORECASE), "unmute "),
    (re.compile(r"^activated ", re.IGNORECASE), "activate "),
    (re.compile(r"^ran ", re.IGNORECASE), "run "),
    (re.compile(r"^skipped ", re.IGNORECASE), "skip "),
    (re.compile(r"^went back ", re.IGNORECASE), "go back "),
)


@dataclass
class _Pending:
    """A low-confidence action awaiting a yes/no from the user."""

    actions: list[Action] = field(default_factory=list)


def _normalize_reply(text: str) -> str:
    return re.sub(r"[^\w\s']", " ", (text or "").casefold()).strip()


def _is_affirmative(text: str) -> bool:
    return _normalize_reply(text) in _AFFIRMATIVE


def _is_negative(text: str) -> bool:
    return _normalize_reply(text) in _NEGATIVE


def _imperative(speech: str) -> str:
    """Turn a past-tense result sentence into an imperative request."""
    text = speech.rstrip(". ")
    for pattern, replacement in _TENSE_FIXES:
        if pattern.match(text):
            return pattern.sub(replacement, text)
    return text[:1].lower() + text[1:]


class CactusConversationAgent(conversation.AbstractConversationAgent):
    """Needle 3 driving the home through Home Assistant service calls."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        options: dict[str, Any],
        runtime: CactusTextAgent,
    ) -> None:
        self.hass = hass
        self.entry = entry
        self._options = options
        self._runtime = runtime

        self._lock = asyncio.Lock()
        self._toolset: Toolset | None = None
        self._toolset_built_at = 0.0
        self._targets_ttl = float(options.get("target_cache_ttl", 5.0))
        self._last_conversation_id: str | None = None
        self._pending: dict[str, _Pending] = {}
        self._enums_disabled = False

        self._max_steps = int(options.get(CONF_MAX_STEPS, DEFAULT_MAX_STEPS))
        self._min_confidence = float(options.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE))

    # -- AbstractConversationAgent protocol --------------------------------
    @property
    def supported_languages(self) -> list[str]:
        """Return the base language tags Needle and Whistle both handle."""
        return list(SUPPORTED_LANGUAGES)

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Process one utterance."""
        response = intent.IntentResponse(language=user_input.language)
        text = (user_input.text or "").strip()

        if not text:
            response.async_set_speech("I did not catch that.")
            return self._result(response, user_input, continue_conversation=False)

        try:
            speech, keep_open = await self._async_converse(user_input, text)
        except CactusError as err:
            _LOGGER.error("Cactus runtime error: %s", err)
            speech, keep_open = (
                "The local Cactus engine is unavailable right now. Check the log for details.",
                False,
            )
        except Exception:  # noqa: BLE001 - the agent must never raise into the pipeline
            _LOGGER.exception("Unexpected failure while handling %r", text)
            speech, keep_open = ("Sorry, something went wrong while handling that request.", False)

        response.async_set_speech(speech)
        return self._result(response, user_input, continue_conversation=keep_open)

    def _result(
        self,
        response: intent.IntentResponse,
        user_input: conversation.ConversationInput,
        *,
        continue_conversation: bool,
    ) -> conversation.ConversationResult:
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
            continue_conversation=continue_conversation,
        )

    # -- toolset plumbing ---------------------------------------------------
    def _build_toolset(self) -> Toolset:
        now = time.monotonic()
        if self._toolset is not None and now - self._toolset_built_at < self._targets_ttl:
            return self._toolset

        targets = async_collect_targets(
            self.hass,
            {
                "enabled_domains": self._options.get(CONF_ENABLED_DOMAINS),
                "exclude_entities": self._options.get(CONF_EXCLUDE_ENTITIES),
                "assist_only": self._options.get(CONF_ASSIST_ONLY, True),
                "max_targets": self._options.get(CONF_MAX_TARGETS, DEFAULT_MAX_TARGETS),
            },
        )
        self._toolset = build_toolset(
            targets,
            max_enum_names=(
                0
                if self._enums_disabled
                else int(self._options.get(CONF_MAX_ENUM_NAMES, DEFAULT_MAX_ENUM_NAMES))
            ),
            max_total_names=int(
                self._options.get(CONF_MAX_TOOL_NAMES, DEFAULT_MAX_TOOL_NAMES)
            ),
        )
        self._toolset_built_at = now
        return self._toolset

    def _system_facts(self, user_input: conversation.ConversationInput) -> str:
        language = self._options.get(CONF_LANGUAGE) or user_input.language or "en"
        if language == "auto":
            language = user_input.language or "en"
        facts = [f"locale: {language}", f"assistant: {MODEL_NEEDLE}"]
        if user_input.device_id:
            device = dr.async_get(self.hass).async_get(user_input.device_id)
            if device is not None:
                name = device.name_by_user or device.name
                if name:
                    facts.append(f"device: {name}")
        return "; ".join(facts)

    def _normalize_numbers(
        self, text: str, user_input: conversation.ConversationInput
    ) -> str:
        """Convert spoken number words to digits before the model sees them.

        Speech recognition writes "50%" as "fünfzig Prozent", and Needle grounds
        a numeric argument only when the digits appear in the request, so the
        quantity never reached the tool call.
        """
        if not self._options.get(CONF_NORMALIZE_NUMBERS, DEFAULT_NORMALIZE_NUMBERS):
            return text

        configured = self._options.get(CONF_LANGUAGE)
        language = configured if configured and configured != "auto" else user_input.language
        normalized = normalize_spoken_numbers(text, language)
        if normalized != text:
            _LOGGER.debug("Normalised spoken numbers: %r -> %r", text, normalized)
        return normalized

    # -- the loop -----------------------------------------------------------
    async def _async_converse(
        self, user_input: conversation.ConversationInput, text: str
    ) -> tuple[str, bool]:
        async with self._lock:
            toolset = self._build_toolset()
            if toolset.empty:
                return (
                    "I could not find any devices exposed to voice control. "
                    "Expose some entities to Assist in the voice assistants settings first.",
                    False,
                )

            conversation_id = user_input.conversation_id or "default"
            text = self._normalize_numbers(text, user_input)

            pending = self._pending.pop(conversation_id, None)
            if pending is not None:
                if _is_affirmative(text):
                    return await self._async_execute_actions(pending.actions, user_input), False
                if _is_negative(text):
                    return "Okay, I cancelled that.", False

            if self._last_conversation_id is not None and self._last_conversation_id != conversation_id:
                await self.hass.async_add_executor_job(self._runtime.reset)
            self._last_conversation_id = conversation_id

            return await self._async_run_agent(user_input, text, toolset, conversation_id)

    async def _async_run_agent(
        self,
        user_input: conversation.ConversationInput,
        text: str,
        toolset: Toolset,
        conversation_id: str,
    ) -> tuple[str, bool]:
        original_text = text
        results: list[str] = []
        system = self._system_facts(user_input)

        step = 0
        retried = False
        while step < max(1, self._max_steps):
            complete = partial(
                self._runtime.complete,
                text,
                schemas=toolset.schemas,
                signature=toolset.signature,
                system=system,
            )
            turn = await self.hass.async_add_executor_job(complete)

            error = turn.get("error")
            if error and not retried and not self._enums_disabled:
                # The toolset is part of Needle's prompt, so an oversized
                # toolset surfaces as an engine error rather than a refusal.
                # Retry once with the name lists removed: the resolver still
                # finds devices by itself, so the command can still be honoured.
                retried = True
                self._enums_disabled = True
                self._toolset = None
                toolset = self._build_toolset()
                _LOGGER.warning(
                    "Needle reported %r; retrying with the device name lists removed",
                    error,
                )
                continue

            if error:
                _LOGGER.warning("Needle reported an error: %s (%s)", error, turn.get("error_code"))
                results.append(
                    f"The local model reported a problem: {error}. If that mentions the "
                    "context length, reduce the entities exposed to Assist or lower the "
                    "device name budget in the options."
                )
                break

            step += 1

            calls = list(turn.get("function_calls") or [])
            suppressed = list(turn.get("suppressed_calls") or [])

            if not calls:
                if suppressed and not results:
                    planned = self._plan_calls(suppressed, toolset)
                    if planned:
                        self._remember_pending(conversation_id, planned)
                        return (
                            "I am not sure I understood that. Did you want to "
                            f"{self._describe(planned)}?",
                            True,
                        )
                break

            actions: list[Action] = []
            problems: list[str] = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                try:
                    actions.append(
                        plan_action(
                            str(call.get("name", "")),
                            call.get("arguments") or {},
                            toolset,
                            min_score=0.55,
                        )
                    )
                except ToolError as err:
                    problems.append(str(err))

            if not actions:
                results.extend(problems)
                break

            confidence = turn.get("confidence")
            if (
                not results
                and confidence is not None
                and confidence < self._min_confidence
            ):
                self._remember_pending(conversation_id, actions)
                return (
                    f"I think you want to {self._describe(actions)}. Should I do that?",
                    True,
                )

            results.append(await self._async_execute_actions(actions, user_input))
            results.extend(problems)

            # Feed the executed results back so multi-step requests continue.
            text = json.dumps(
                {
                    "executed": [
                        {"tool": action.tool, "target": action.target.name}
                        for action in actions
                    ],
                    "self_report": results[-1] if results else "",
                }
            )

        if results:
            spoken = " ".join(dict.fromkeys(results))
            return spoken, False

        return await self._async_fallback(original_text, toolset, user_input)

    def _plan_calls(self, calls: list[Any], toolset: Toolset) -> list[Action]:
        planned: list[Action] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            try:
                planned.append(
                    plan_action(
                        str(call.get("name", "")),
                        call.get("arguments") or {},
                        toolset,
                        min_score=0.55,
                    )
                )
            except ToolError:
                continue
        return planned

    async def _async_fallback(
        self,
        text: str,
        toolset: Toolset,
        user_input: conversation.ConversationInput,
    ) -> tuple[str, bool]:
        """Deterministic text handling when Needle proposes no tool call."""
        action = fallback_parse(text, toolset)
        if action is not None:
            return await self._async_execute_actions([action], user_input), False
        return (
            "I could not connect that to a device or an action I can perform.",
            False,
        )

    def _remember_pending(self, conversation_id: str, actions: list[Action]) -> None:
        """Store a low-confidence action, bounded so it cannot leak."""
        if len(self._pending) >= 32:
            self._pending.pop(next(iter(self._pending)))
        self._pending[conversation_id] = _Pending(actions)

    @staticmethod
    def _describe(actions: list[Action]) -> str:
        return " and ".join(
            _imperative(action.speech) if action.speech else f"check {action.target.name}"
            for action in actions
        )

    async def _async_execute_actions(
        self, actions: list[Action], user_input: conversation.ConversationInput
    ) -> str:
        spoken: list[str] = []
        for action in actions:
            spoken.append(await self._async_execute(action, user_input))
        return " ".join(dict.fromkeys(spoken))

    async def _async_execute(
        self, action: Action, user_input: conversation.ConversationInput
    ) -> str:
        if action.tool == "get_state":
            return self._describe_state(action.target)

        if action.service_domain is None or action.service is None:
            return action.speech or "Done."

        data = {**action.data, "entity_id": list(action.target.entity_ids)}
        try:
            await self.hass.services.async_call(
                action.service_domain,
                action.service,
                data,
                blocking=True,
                context=user_input.context,
                return_response=False,
            )
        except HomeAssistantError as err:
            _LOGGER.warning(
                "Service %s.%s failed for %s: %s",
                action.service_domain,
                action.service,
                action.target.entity_ids,
                err,
            )
            return f"Sorry, I could not control {action.target.name}: {err}"

        return action.speech

    def _describe_state(self, target: DeviceTarget) -> str:
        states = [self.hass.states.get(entity_id) for entity_id in target.entity_ids]
        available = [state for state in states if state is not None]
        if not available:
            return f"{target.name} is not available right now."

        if target.kind == "group":
            on_count = sum(1 for state in available if state.state == "on")
            return f"{on_count} of {len(available)} {target.name} are on."

        state = available[0]
        reading = state.state.replace("_", " ")
        detail = ""
        if target.domain == "climate":
            current = state.attributes.get("current_temperature")
            if current is not None:
                detail = f" at {current} degrees"
        return f"{target.name} is {reading}{detail}."
