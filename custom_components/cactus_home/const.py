"""Constants for the Cactus Home integration.

Kept free of Home Assistant imports so the values can be reused by tests and
by the standalone smoke script under ``scripts/``.
"""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "cactus_home"
NAME: Final = "Cactus Home"
MANUFACTURER: Final = "Cactus Compute"
MODEL_NEEDLE: Final = "Needle 3"
MODEL_WHISTLE: Final = "Whistle"

# Platforms forwarded from the config entry. ``stt`` is the only real platform
# (the conversation agent is registered directly with the conversation
# component), and it can be turned off in the options flow.
PLATFORM_STT: Final = "stt"

# --------------------------------------------------------------------------
# Config keys
# --------------------------------------------------------------------------
CONF_NAME: Final = "name"
CONF_ENABLE_CONVERSATION: Final = "enable_conversation"
CONF_ENABLE_STT: Final = "enable_stt"
CONF_ASSIST_ONLY: Final = "assist_only"
CONF_ENABLED_DOMAINS: Final = "enabled_domains"
CONF_EXCLUDE_ENTITIES: Final = "exclude_entities"
CONF_NEEDLE_WEIGHTS: Final = "needle_weights"
CONF_WHISTLE_WEIGHTS: Final = "whistle_weights"
CONF_LANGUAGE: Final = "language"
CONF_MIN_CONFIDENCE: Final = "min_confidence"
CONF_MAX_STEPS: Final = "max_steps"
CONF_MAX_NEW_TOKENS: Final = "max_new_tokens"
CONF_MAX_TARGETS: Final = "max_targets"
CONF_MAX_ENUM_NAMES: Final = "max_enum_names"
CONF_TELEMETRY: Final = "telemetry"
CONF_THREADS: Final = "threads"
CONF_STT_KEYWORDS: Final = "stt_keywords"
CONF_AUTO_STT_KEYWORDS: Final = "auto_stt_keywords"
CONF_DEBUG_AUDIO_DIR: Final = "debug_audio_dir"
CONF_TARGET_TTL: Final = "target_cache_ttl"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
# Domains whose entities can be turned into tools. Deliberately restricted to
# domains this integration knows how to drive through a service call.
DEFAULT_ENABLED_DOMAINS: Final = [
    "light",
    "switch",
    "fan",
    "cover",
    "lock",
    "climate",
    "water_heater",
    "media_player",
    "scene",
    "script",
    "input_boolean",
    "humidifier",
    "siren",
]

ALL_CONTROLLABLE_DOMAINS: Final = ["light", "switch", "fan", "cover", "lock", "climate", "water_heater", "media_player", "scene", "script", "input_boolean", "humidifier", "siren"]

DEFAULT_ENABLE_CONVERSATION: Final = True
DEFAULT_ENABLE_STT: Final = True
DEFAULT_ASSIST_ONLY: Final = True
DEFAULT_LANGUAGE: Final = "auto"
DEFAULT_MIN_CONFIDENCE: Final = 0.4
DEFAULT_MAX_STEPS: Final = 4
DEFAULT_MAX_NEW_TOKENS: Final = 256
DEFAULT_MAX_TARGETS: Final = 200
DEFAULT_MAX_ENUM_NAMES: Final = 64
DEFAULT_TELEMETRY: Final = False
DEFAULT_THREADS: Final = 0
DEFAULT_STT_KEYWORDS: Final = ""
DEFAULT_TARGET_TTL: Final = 5.0

# Experimental: bias the speech decoder toward the names of the devices in this
# home. Off by default. The cap keeps the bias from diluting, and places that
# are not device names are left out entirely.
DEFAULT_AUTO_STT_KEYWORDS: Final = False
AUTO_STT_KEYWORD_LIMIT: Final = 32
AUTO_STT_KEYWORD_MAX_WORDS: Final = 4
STT_KEYWORD_CACHE_TTL: Final = 30.0

# Confidence floor for emitting a deterministic action without asking first.
CONFIDENCE_CONFIRM_FLOOR: Final = 0.1

# Languages Whistle and Needle 3 both cover.
SUPPORTED_LANGUAGES: Final = ("en", "de", "fr", "es", "it", "nl", "pl")

# Human words per domain, used to name area/floor groups ("Kitchen lights").
DOMAIN_WORDS: Final = {
    "light": "lights",
    "switch": "switches",
    "fan": "fans",
    "cover": "covers",
    "lock": "locks",
    "climate": "thermostats",
    "water_heater": "water heaters",
    "media_player": "media players",
    "scene": "scenes",
    "script": "scripts",
    "input_boolean": "toggles",
    "humidifier": "humidifiers",
    "siren": "sirens",
}

# Defaults handed to the config flow for each option.
DEFAULT_OPTIONS: Final = {
    CONF_NAME: NAME,
    CONF_ENABLE_CONVERSATION: DEFAULT_ENABLE_CONVERSATION,
    CONF_ENABLE_STT: DEFAULT_ENABLE_STT,
    CONF_ASSIST_ONLY: DEFAULT_ASSIST_ONLY,
    CONF_ENABLED_DOMAINS: DEFAULT_ENABLED_DOMAINS,
    CONF_EXCLUDE_ENTITIES: [],
    CONF_NEEDLE_WEIGHTS: "",
    CONF_WHISTLE_WEIGHTS: "",
    CONF_LANGUAGE: DEFAULT_LANGUAGE,
    CONF_MIN_CONFIDENCE: DEFAULT_MIN_CONFIDENCE,
    CONF_MAX_STEPS: DEFAULT_MAX_STEPS,
    CONF_MAX_NEW_TOKENS: DEFAULT_MAX_NEW_TOKENS,
    CONF_MAX_TARGETS: DEFAULT_MAX_TARGETS,
    CONF_MAX_ENUM_NAMES: DEFAULT_MAX_ENUM_NAMES,
    CONF_TELEMETRY: DEFAULT_TELEMETRY,
    CONF_THREADS: DEFAULT_THREADS,
    CONF_STT_KEYWORDS: DEFAULT_STT_KEYWORDS,
    CONF_AUTO_STT_KEYWORDS: DEFAULT_AUTO_STT_KEYWORDS,
    CONF_DEBUG_AUDIO_DIR: "",
    CONF_TARGET_TTL: DEFAULT_TARGET_TTL,
}
