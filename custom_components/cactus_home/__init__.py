"""Set up the Cactus Home integration.

One config entry owns two local Cactus resources:

* a Needle 3 agent, registered with the conversation component so it appears in
  the Assist agent picker;
* a Whistle transcriber, exposed through the ``stt`` platform.

Both wrap the blocking Cactus runtime, so all engine work is pushed onto
executor threads.

Home Assistant imports are deliberately deferred to call time. That keeps the
pure modules (:mod:`.tools`, :mod:`.entities`, :mod:`.needle`, :mod:`.audio`)
importable in a plain ``pytest`` run without a Home Assistant installation.
No Home Assistant import happens until Home Assistant itself calls
:func:`async_setup_entry`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .const import (
    CONF_ENABLE_CONVERSATION,
    CONF_ENABLE_STT,
    CONF_MAX_NEW_TOKENS,
    CONF_NEEDLE_WEIGHTS,
    CONF_STT_KEYWORDS,
    CONF_TELEMETRY,
    CONF_THREADS,
    CONF_WHISTLE_WEIGHTS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OPTIONS,
    DOMAIN,
    PLATFORM_STT,
)
from .needle import CactusTextAgent, CactusTranscriber

if TYPE_CHECKING:  # pragma: no cover - typing only
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

    from .conversation import CactusConversationAgent

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [PLATFORM_STT]


@dataclass
class CactusHomeData:
    """Runtime objects shared by the platforms and the conversation agent."""

    options: dict[str, Any]
    runtime: CactusTextAgent
    transcriber: CactusTranscriber
    agent: CactusConversationAgent | None = None
    unsub: list = field(default_factory=list)


def merged_options(entry: ConfigEntry) -> dict[str, Any]:
    """Layer defaults, config data and options into one mapping."""
    return {**DEFAULT_OPTIONS, **entry.data, **entry.options}


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Cactus Home from a config entry."""
    from homeassistant.components import conversation
    from homeassistant.exceptions import ConfigEntryError, ConfigEntryNotReady

    from .conversation import CactusConversationAgent
    from .needle import CactusUnavailable

    options = merged_options(entry)

    if not options[CONF_ENABLE_CONVERSATION] and not options[CONF_ENABLE_STT]:
        raise ConfigEntryError("Enable at least one of the conversation agent or the STT engine.")

    runtime = CactusTextAgent(
        weights=(options.get(CONF_NEEDLE_WEIGHTS) or "").strip() or None,
        max_new_tokens=int(options.get(CONF_MAX_NEW_TOKENS, DEFAULT_MAX_NEW_TOKENS)),
        threads=int(options.get(CONF_THREADS) or 0),
        telemetry=bool(options.get(CONF_TELEMETRY)),
        logger=_LOGGER,
    )
    transcriber = CactusTranscriber(
        weights=(options.get(CONF_WHISTLE_WEIGHTS) or "").strip() or None,
        keywords=_split_keywords(options.get(CONF_STT_KEYWORDS)),
        threads=int(options.get(CONF_THREADS) or 0),
        telemetry=bool(options.get(CONF_TELEMETRY)),
        logger=_LOGGER,
    )

    # Load eagerly so configuration mistakes surface as a retryable setup error
    # instead of failing on the user's first spoken command. The engine and the
    # model weights are downloaded and cached on this first call.
    try:
        if options[CONF_ENABLE_CONVERSATION]:
            await hass.async_add_executor_job(runtime.ensure_loaded)
    except CactusUnavailable as err:
        raise ConfigEntryNotReady(str(err)) from err

    if options[CONF_ENABLE_STT]:
        # Warm the speech model so a failed weights download is logged with its
        # cause at setup time. The failure is deliberately not raised: a broken
        # speech model must not disable a working conversation agent. The
        # weights are retried on the next voice command.
        try:
            await hass.async_add_executor_job(transcriber.warm_up)
        except CactusUnavailable as err:
            _LOGGER.error(
                "Cactus speech-to-text is unavailable: %s. The conversation agent still "
                "works. Restore network access so whistle.cact can download, or set "
                "CACTUS_WHISTLE_WEIGHTS to a local copy. The model is retried on the "
                "next voice command.",
                err,
            )

    data = CactusHomeData(options=options, runtime=runtime, transcriber=transcriber)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = data

    if options[CONF_ENABLE_CONVERSATION]:
        if not hasattr(conversation, "async_set_agent"):
            raise ConfigEntryError(
                "This Home Assistant version does not support registering a conversation agent."
            )
        data.agent = CactusConversationAgent(hass, entry, options, runtime)
        conversation.async_set_agent(hass, entry, data.agent)

    if options[CONF_ENABLE_STT]:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry and release the engine."""
    from homeassistant.components import conversation

    data: CactusHomeData | None = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    unload_ok = True

    if data is not None and data.options.get(CONF_ENABLE_STT):
        unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if data is not None and data.agent is not None:
        _unregister_agent(conversation, hass, entry, data.agent)

    if data is not None:
        await hass.async_add_executor_job(data.runtime.close)
        await hass.async_add_executor_job(data.transcriber.close)

    return unload_ok


def _unregister_agent(conversation: Any, hass: HomeAssistant, entry: ConfigEntry, agent: Any) -> None:
    """Unregister the agent, tolerating the older one-argument signature."""
    try:
        conversation.async_unset_agent(hass, entry)
    except TypeError:  # pragma: no cover - Home Assistant < 2024.6
        conversation.async_unset_agent(hass, agent)


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)


def _split_keywords(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).replace("\n", ",").split(",") if part.strip()]
