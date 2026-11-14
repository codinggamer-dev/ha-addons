"""Config and options flow for Cactus Home.

There is exactly one local Cactus engine per Home Assistant instance, so the
flow enforces a single config entry. All engine tuning lives in options and can
be changed later without touching YAML.
"""

from __future__ import annotations

import os
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv

from .const import (
    ALL_CONTROLLABLE_DOMAINS,
    CONF_ASSIST_ONLY,
    CONF_AUTO_STT_KEYWORDS,
    CONF_DEBUG_AUDIO_DIR,
    CONF_ENABLED_DOMAINS,
    CONF_ENABLE_CONVERSATION,
    CONF_ENABLE_STT,
    CONF_EXCLUDE_ENTITIES,
    CONF_LANGUAGE,
    CONF_MAX_ENUM_NAMES,
    CONF_MAX_NEW_TOKENS,
    CONF_MAX_STEPS,
    CONF_MAX_TARGETS,
    CONF_MIN_CONFIDENCE,
    CONF_NAME,
    CONF_NEEDLE_WEIGHTS,
    CONF_STT_KEYWORDS,
    CONF_TARGET_TTL,
    CONF_TELEMETRY,
    CONF_THREADS,
    CONF_WHISTLE_WEIGHTS,
    DEFAULT_ENABLED_DOMAINS,
    DEFAULT_LANGUAGE,
    DEFAULT_OPTIONS,
    DOMAIN,
    NAME,
    SUPPORTED_LANGUAGES,
)

_DOMAIN_CHOICES = {domain: domain.replace("_", " ") for domain in ALL_CONTROLLABLE_DOMAINS}


def _as_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).replace("\n", ",").split(",") if part.strip()]


def _schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the shared setup/options form."""
    get = lambda key: defaults.get(key, DEFAULT_OPTIONS.get(key))  # noqa: E731

    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=get(CONF_NAME) or NAME): cv.string,
            vol.Required(
                CONF_ENABLE_CONVERSATION, default=bool(get(CONF_ENABLE_CONVERSATION))
            ): cv.boolean,
            vol.Required(CONF_ENABLE_STT, default=bool(get(CONF_ENABLE_STT))): cv.boolean,
            vol.Required(CONF_ASSIST_ONLY, default=bool(get(CONF_ASSIST_ONLY))): cv.boolean,
            vol.Required(
                CONF_ENABLED_DOMAINS, default=list(get(CONF_ENABLED_DOMAINS) or DEFAULT_ENABLED_DOMAINS)
            ): cv.multi_select(_DOMAIN_CHOICES),
            vol.Optional(
                CONF_EXCLUDE_ENTITIES, default=", ".join(_as_list(get(CONF_EXCLUDE_ENTITIES)))
            ): cv.string,
            vol.Optional(CONF_NEEDLE_WEIGHTS, default=str(get(CONF_NEEDLE_WEIGHTS) or "")): cv.string,
            vol.Optional(
                CONF_WHISTLE_WEIGHTS, default=str(get(CONF_WHISTLE_WEIGHTS) or "")
            ): cv.string,
            vol.Required(
                CONF_LANGUAGE, default=str(get(CONF_LANGUAGE) or DEFAULT_LANGUAGE)
            ): vol.In(["auto", *SUPPORTED_LANGUAGES]),
            vol.Required(
                CONF_MIN_CONFIDENCE, default=float(get(CONF_MIN_CONFIDENCE))
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Required(
                CONF_MAX_STEPS, default=int(get(CONF_MAX_STEPS))
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=8)),
            vol.Required(
                CONF_MAX_NEW_TOKENS, default=int(get(CONF_MAX_NEW_TOKENS))
            ): vol.All(vol.Coerce(int), vol.Range(min=32, max=1024)),
            vol.Required(
                CONF_MAX_TARGETS, default=int(get(CONF_MAX_TARGETS))
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=1000)),
            vol.Required(
                CONF_MAX_ENUM_NAMES, default=int(get(CONF_MAX_ENUM_NAMES))
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=512)),
            vol.Optional(
                CONF_STT_KEYWORDS, default=str(get(CONF_STT_KEYWORDS) or "")
            ): cv.string,
            vol.Required(
                CONF_AUTO_STT_KEYWORDS, default=bool(get(CONF_AUTO_STT_KEYWORDS))
            ): cv.boolean,
            vol.Optional(
                CONF_DEBUG_AUDIO_DIR, default=str(get(CONF_DEBUG_AUDIO_DIR) or "")
            ): cv.string,
            vol.Required(
                CONF_THREADS, default=int(get(CONF_THREADS))
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=64)),
            vol.Required(CONF_TELEMETRY, default=bool(get(CONF_TELEMETRY))): cv.boolean,
            vol.Optional(
                CONF_TARGET_TTL, default=float(get(CONF_TARGET_TTL))
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=120.0)),
        }
    )


def _normalize(user_input: dict[str, Any]) -> dict[str, Any]:
    data = dict(user_input)
    data[CONF_EXCLUDE_ENTITIES] = _as_list(data.get(CONF_EXCLUDE_ENTITIES))
    data[CONF_STT_KEYWORDS] = ", ".join(_as_list(data.get(CONF_STT_KEYWORDS)))
    data[CONF_NEEDLE_WEIGHTS] = str(data.get(CONF_NEEDLE_WEIGHTS) or "").strip()
    data[CONF_WHISTLE_WEIGHTS] = str(data.get(CONF_WHISTLE_WEIGHTS) or "").strip()
    return data


def _validate(user_input: dict[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    if not user_input.get(CONF_ENABLE_CONVERSATION) and not user_input.get(CONF_ENABLE_STT):
        errors["base"] = "nothing_enabled"
    for key in (CONF_NEEDLE_WEIGHTS, CONF_WHISTLE_WEIGHTS):
        raw = str(user_input.get(key) or "").strip()
        if raw and not os.path.isfile(os.path.expanduser(raw)):
            errors[key] = "weights_not_found"
    return errors


class CactusHomeConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial setup."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show the setup form and create the single config entry."""
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate(user_input)
            if not errors:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                data = _normalize(user_input)
                return self.async_create_entry(title=data[CONF_NAME], data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=_schema(user_input or {}),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow handler."""
        return CactusHomeOptionsFlow()


class CactusHomeOptionsFlow(OptionsFlow):
    """Handle reconfiguration of an existing entry."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show the options form."""
        errors: dict[str, str] = {}

        if user_input is not None:
            errors = _validate(user_input)
            if not errors:
                data = _normalize(user_input)
                self.hass.config_entries.async_update_entry(
                    self.config_entry, title=data[CONF_NAME]
                )
                return self.async_create_entry(title="", data=data)

        defaults = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(
            step_id="init",
            data_schema=_schema(defaults),
            errors=errors,
        )
