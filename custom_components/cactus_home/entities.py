"""Home Assistant glue: read the registries, produce :class:`DeviceTarget` objects.

This module is the only place that knows about Home Assistant registries. It
collects the entities the user has exposed to voice, works out what each one
can do, and hands the result to :func:`.tools.synthesize_groups`, which adds
area/domain groups ("Kitchen lights") so grouped commands resolve
deterministically.
"""

from __future__ import annotations

from typing import Any, Final

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

try:  # Home Assistant >= 2025.x moved exposure out of helpers.entity
    from homeassistant.components.homeassistant.exposed_entities import (
        async_should_expose,
    )
except ImportError:  # pragma: no cover - older Home Assistant
    from homeassistant.helpers.entity import async_should_expose  # type: ignore[no-redef]

from .tools import (
    CAP_BRIGHTNESS,
    CAP_COVER,
    CAP_LOCK,
    CAP_MEDIA,
    CAP_POSITION,
    CAP_QUERY,
    CAP_SCENE,
    CAP_SCRIPT,
    CAP_STATE,
    CAP_TEMPERATURE,
    CAP_VOLUME,
    DeviceTarget,
    normalize_name,
    synthesize_groups,
)

# Feature enums are stable public APIs, but importing them defensively keeps the
# integration loadable on Home Assistant builds where a domain was reshuffled.
try:
    from homeassistant.components.climate import ClimateEntityFeature
except ImportError:  # pragma: no cover
    ClimateEntityFeature = None  # type: ignore[assignment]

try:
    from homeassistant.components.cover import CoverEntityFeature
except ImportError:  # pragma: no cover
    CoverEntityFeature = None  # type: ignore[assignment]

try:
    from homeassistant.components.fan import FanEntityFeature
except ImportError:  # pragma: no cover
    FanEntityFeature = None  # type: ignore[assignment]

try:
    from homeassistant.components.media_player import MediaPlayerEntityFeature
except ImportError:  # pragma: no cover
    MediaPlayerEntityFeature = None  # type: ignore[assignment]

# Domains that have a plain on/off state worth exposing as `set_state`.
_STATE_DOMAINS: Final = frozenset(
    {"light", "switch", "fan", "media_player", "climate", "humidifier", "siren", "input_boolean"}
)


def _has_feature(features: int, enum: Any, member: str) -> bool:
    if not features or enum is None:
        return False
    value = getattr(enum, member, None)
    if value is None:
        return False
    try:
        return bool(features & int(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def _light_supports_brightness(attributes: dict[str, Any]) -> bool:
    modes = attributes.get("supported_color_modes")
    if isinstance(modes, (list, tuple, set)) and modes:
        return any(str(mode) not in ("onoff", "unknown") for mode in modes)
    if "brightness" in attributes:
        return True
    # Legacy light entities only report supported_features; brightness was bit 0.
    features = attributes.get("supported_features")
    return bool(features and int(features) & 1)


def _capabilities(domain: str, attributes: dict[str, Any]) -> frozenset[str]:
    """Work out which tool families an entity actually supports."""
    capabilities: set[str] = set()
    features = int(attributes.get("supported_features") or 0)

    if domain in _STATE_DOMAINS:
        capabilities.add(CAP_STATE)
        capabilities.add(CAP_QUERY)

    if domain == "light":
        if _light_supports_brightness(attributes):
            capabilities.add(CAP_BRIGHTNESS)
    elif domain == "fan":
        if _has_feature(features, FanEntityFeature, "SET_SPEED") or "percentage" in attributes:
            capabilities.add(CAP_BRIGHTNESS)
    elif domain == "cover":
        capabilities.add(CAP_COVER)
        capabilities.add(CAP_QUERY)
        if _has_feature(features, CoverEntityFeature, "SET_POSITION") or "current_position" in attributes:
            capabilities.add(CAP_POSITION)
    elif domain == "climate":
        if features == 0 or _has_feature(features, ClimateEntityFeature, "TARGET_TEMPERATURE"):
            capabilities.add(CAP_TEMPERATURE)
        capabilities.add(CAP_QUERY)
    elif domain == "water_heater":
        capabilities.add(CAP_TEMPERATURE)
        capabilities.add(CAP_QUERY)
    elif domain == "media_player":
        capabilities.add(CAP_MEDIA)
        capabilities.add(CAP_QUERY)
        if features == 0 or _has_feature(features, MediaPlayerEntityFeature, "VOLUME_SET"):
            capabilities.add(CAP_VOLUME)
    elif domain == "lock":
        capabilities.add(CAP_LOCK)
        capabilities.add(CAP_QUERY)
    elif domain == "scene":
        capabilities.add(CAP_SCENE)
    elif domain == "script":
        capabilities.add(CAP_SCRIPT)
    else:
        capabilities.add(CAP_QUERY)

    return frozenset(capabilities)


def _aliases(entry: er.RegistryEntry) -> list[str]:
    """Return the user-set voice aliases, ignoring computed-name placeholders."""
    aliases = getattr(entry, "aliases", None)
    if not aliases:
        return []
    return [alias for alias in aliases if isinstance(alias, str)]


def async_collect_targets(hass: HomeAssistant, options: dict[str, Any]) -> list[DeviceTarget]:
    """Build the target list for the current registry and options."""
    enabled_domains = set(options.get("enabled_domains") or [])
    excluded = set(options.get("exclude_entities") or [])
    assist_only = bool(options.get("assist_only", True))
    max_targets = int(options.get("max_targets", 200))

    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)

    entities: list[DeviceTarget] = []
    for entry in entity_registry.entities.values():
        domain = entry.entity_id.split(".", 1)[0]
        if enabled_domains and domain not in enabled_domains:
            continue
        if entry.entity_id in excluded:
            continue
        if entry.disabled_by is not None or entry.hidden_by is not None:
            continue
        if assist_only and not async_should_expose(hass, "conversation", entry.entity_id):
            continue

        state = hass.states.get(entry.entity_id)
        if state is None or state.state in ("unavailable", "unknown"):
            continue

        device = device_registry.async_get(entry.device_id) if entry.device_id else None
        area_id = entry.area_id or (device.area_id if device is not None else None)
        area_entry = area_registry.async_get_area(area_id) if area_id else None
        area_name = area_entry.name if area_entry is not None else None

        friendly = state.attributes.get("friendly_name")
        device_name = None
        if device is not None:
            device_name = device.name_by_user or device.name

        canonical = entry.name or friendly or entry.original_name or device_name or entry.entity_id
        canonical = str(canonical)

        names: list[str] = list(_aliases(entry))
        if friendly and friendly != canonical:
            names.append(str(friendly))
        if device_name and device_name != canonical:
            names.append(device_name)
        if area_name and not normalize_name(canonical).startswith(normalize_name(area_name)):
            # "Kitchen light" for an entity named just "Light".
            names.append(f"{area_name} {canonical}")

        entities.append(
            DeviceTarget(
                target_id=entry.entity_id,
                name=canonical,
                domain=domain,
                entity_ids=(entry.entity_id,),
                capabilities=_capabilities(domain, dict(state.attributes)),
                names=tuple(names),
                area=area_name,
                kind="entity",
            )
        )

    entities.sort(key=lambda target: (target.name.casefold(), target.target_id))
    entities = entities[:max_targets]
    return [*entities, *synthesize_groups(entities)]
