"""Turn Home Assistant entities into Needle 3 tool schemas, and tool calls back
into Home Assistant service calls.

Everything in this module is deliberately free of Home Assistant imports:

* :mod:`entities` walks the registries and produces :class:`DeviceTarget`
  objects;
* :func:`build_toolset` compiles those targets into grammar-constrained JSON
  schemas that Needle 3 can decode exactly;
* :func:`plan_action` maps a decoded tool call onto a concrete service call,
  resolving the spoken target name deterministically;
* :func:`fallback_parse` is the text-only safety net used when Needle returns
  no tool call at all.

Keeping the module importable without Home Assistant means the interesting
logic is unit-testable with plain ``pytest``.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence

from .const import DOMAIN_WORDS

# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------
CAP_STATE: Final = "state"
CAP_BRIGHTNESS: Final = "brightness"
CAP_TEMPERATURE: Final = "temperature"
CAP_POSITION: Final = "position"
CAP_VOLUME: Final = "volume"
CAP_MEDIA: Final = "media"
CAP_LOCK: Final = "lock"
CAP_COVER: Final = "cover"
CAP_SCENE: Final = "scene"
CAP_SCRIPT: Final = "script"
CAP_QUERY: Final = "query"

MEDIA_COMMANDS: Final = {
    "play": ("media_play", "Playing {name}."),
    "pause": ("media_pause", "Paused {name}."),
    "stop": ("media_stop", "Stopped {name}."),
    "next": ("media_next_track", "Skipped to the next track on {name}."),
    "previous": ("media_previous_track", "Went back on {name}."),
    "mute": ("volume_mute", "Muted {name}."),
    "unmute": ("volume_mute", "Unmuted {name}."),
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceTarget:
    """One addressable thing the model may control.

    ``entity_ids`` holds more than one entry for area/domain groups
    ("Kitchen lights"), which is how grouped commands stay deterministic.
    """

    target_id: str
    name: str
    domain: str
    entity_ids: tuple[str, ...]
    capabilities: frozenset[str]
    names: tuple[str, ...] = ()
    area: str | None = None
    kind: str = "entity"

    @property
    def spoken_names(self) -> tuple[str, ...]:
        """All acceptable ways to refer to this target, canonical name first."""
        return (self.name, *self.names)

    @property
    def entity_id(self) -> str:
        """Primary entity id; for a group this is the first member."""
        return self.entity_ids[0]


@dataclass(frozen=True)
class Action:
    """A decoded, executable intent."""

    tool: str
    target: DeviceTarget
    speech: str
    service_domain: str | None = None
    service: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


class ToolError(RuntimeError):
    """Base class for tool decoding failures."""


class UnknownToolError(ToolError):
    """The model returned a tool that is not part of this toolset."""


class ResolutionError(ToolError):
    """The spoken target could not be resolved to exactly one device."""


class MissingArgumentError(ToolError):
    """A required argument was absent from the decoded call."""


@dataclass(frozen=True)
class ToolDef:
    """Static description of one emitted tool."""

    name: str
    description: str
    capability: str | None
    target_arg: str = "target"
    extra_properties: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    required_extra: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()


TOOL_DEFS: Final[tuple[ToolDef, ...]] = (
    ToolDef(
        name="set_state",
        description="Turn a device on or off.",
        capability=CAP_STATE,
        target_arg="device",
        extra_properties={
            "state": {
                "type": "string",
                "enum": ["on", "off"],
                "description": "Whether to switch the device on or off.",
            }
        },
        required_extra=("state",),
        triggers=(r"\b(turn|switch|power|flip)\b.*\b(on|off)\b", r"\btoggle\b"),
    ),
    ToolDef(
        name="set_brightness",
        description="Set the brightness of a light or fan, as a percentage.",
        capability=CAP_BRIGHTNESS,
        target_arg="device",
        extra_properties={
            "brightness": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Brightness as a percentage from 1 to 100.",
            }
        },
        required_extra=("brightness",),
        triggers=(r"\b(dim|brighten|brightness)\b",),
    ),
    ToolDef(
        name="set_temperature",
        description="Set the target temperature of a thermostat or water heater.",
        capability=CAP_TEMPERATURE,
        target_arg="device",
        extra_properties={
            "temperature": {
                "type": "number",
                "minimum": -20,
                "maximum": 60,
                "description": "Target temperature in degrees Celsius.",
            }
        },
        required_extra=("temperature",),
        triggers=(r"\b(temperature|degrees|warmer|cooler)\b",),
    ),
    ToolDef(
        name="set_position",
        description="Move a cover, blind or shutter to a percentage position.",
        capability=CAP_POSITION,
        target_arg="device",
        extra_properties={
            "position": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Position as a percentage, 0 is fully closed, 100 fully open.",
            }
        },
        required_extra=("position",),
    ),
    ToolDef(
        name="set_volume",
        description="Set the volume of a media player, as a percentage.",
        capability=CAP_VOLUME,
        target_arg="device",
        extra_properties={
            "volume": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Volume as a percentage from 0 to 100.",
            }
        },
        required_extra=("volume",),
    ),
    ToolDef(
        name="media_control",
        description="Control media playback on a media player.",
        capability=CAP_MEDIA,
        target_arg="device",
        extra_properties={
            "command": {
                "type": "string",
                "enum": ["play", "pause", "stop", "next", "previous", "mute", "unmute"],
                "description": "The playback command to send.",
            }
        },
        required_extra=("command",),
        triggers=(r"\b(play|pause|resume|skip|mute|unmute)\b",),
    ),
    ToolDef(
        name="lock_unlock",
        description="Lock or unlock a lock.",
        capability=CAP_LOCK,
        target_arg="device",
        extra_properties={
            "state": {
                "type": "string",
                "enum": ["lock", "unlock"],
                "description": "Whether to lock or unlock the lock.",
            }
        },
        required_extra=("state",),
        triggers=(r"\b(lock|unlock)\b",),
    ),
    ToolDef(
        name="control_cover",
        description="Open, close or stop a cover, blind or shutter.",
        capability=CAP_COVER,
        target_arg="device",
        extra_properties={
            "state": {
                "type": "string",
                "enum": ["open", "close", "stop"],
                "description": "Whether to open, close or stop the cover.",
            }
        },
        required_extra=("state",),
        triggers=(r"\b(open|close|shut)\b.*\b(blind|curtain|shutter|garage|cover|door)\b",),
    ),
    ToolDef(
        name="activate_scene",
        description="Activate a scene.",
        capability=CAP_SCENE,
        target_arg="scene",
        triggers=(r"\bscene\b",),
    ),
    ToolDef(
        name="run_script",
        description="Run a script or automation routine.",
        capability=CAP_SCRIPT,
        target_arg="script",
        triggers=(r"\b(run|start|execute)\b.*\b(script|routine)\b",),
    ),
    ToolDef(
        name="get_state",
        description="Look up the current state of a device.",
        capability=CAP_QUERY,
        target_arg="device",
        triggers=(r"\b(is|are|what|which|status|state)\b",),
    ),
)

TOOL_DEFS_BY_NAME: Final[dict[str, ToolDef]] = {tool.name: tool for tool in TOOL_DEFS}


# --------------------------------------------------------------------------
# Name normalisation and matching
# --------------------------------------------------------------------------
_ARTICLES: Final = ("the ", "a ", "an ", "my ", "our ", "your ")


def normalize_name(value: str) -> str:
    """Fold a spoken name into a comparable key."""
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    changed = True
    while changed:
        changed = False
        for article in _ARTICLES:
            if text.startswith(article):
                text = text[len(article) :]
                changed = True
    return text


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = normalize_name(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving a spoken name against a candidate pool."""

    target: DeviceTarget | None
    score: float
    ambiguous: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.target is not None


def _score_candidate(query: str, key: str) -> float:
    """Score a normalised query against a normalised candidate name.

    Deliberately conservative: character similarity only counts when the two
    share at least one word, so "garage door opener" cannot drift onto
    "Front Door" just because both contain "door".
    """
    if not key:
        return 0.0
    if query == key:
        return 1.0

    query_words = set(query.split())
    key_words = set(key.split())
    shared = query_words & key_words

    best = 0.0

    if key_words and key_words <= query_words:
        # The candidate's name is fully contained in what was said.
        best = max(best, 0.9)
    if query_words and query_words <= key_words:
        best = max(best, 0.85)

    if query_words and key_words:
        best = max(best, 0.35 * (len(shared) / len(query_words | key_words)))

    if shared:
        best = max(best, difflib.SequenceMatcher(None, query, key).ratio() * 0.9)

    return best


def resolve_target(
    query: str,
    candidates: Sequence[DeviceTarget],
    *,
    min_score: float = 0.55,
    ambiguity_margin: float = 0.02,
) -> Resolution:
    """Resolve a spoken name to a single candidate, or report ambiguity."""
    normalized = normalize_name(query)
    if not normalized:
        return Resolution(None, 0.0)

    scored: list[tuple[float, DeviceTarget]] = []
    for candidate in candidates:
        best_for_candidate = 0.0
        for spoken in candidate.spoken_names:
            best_for_candidate = max(best_for_candidate, _score_candidate(normalized, normalize_name(spoken)))
        if best_for_candidate > 0.0:
            scored.append((best_for_candidate, candidate))

    if not scored:
        return Resolution(None, 0.0)

    scored.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    top_score, top_target = scored[0]
    if top_score < min_score:
        return Resolution(None, top_score)

    rivals = [
        other.name
        for score, other in scored[1:]
        if score >= top_score - ambiguity_margin and other.target_id != top_target.target_id
    ]
    if rivals:
        return Resolution(None, top_score, tuple(_unique(rivals)))

    return Resolution(top_target, top_score)


# --------------------------------------------------------------------------
# Toolset construction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Toolset:
    """A compiled toolset: schemas for Needle plus lookup tables for decoding."""

    schemas: tuple[Mapping[str, Any], ...]
    targets: tuple[DeviceTarget, ...]
    signature: str
    index: Mapping[str, DeviceTarget]
    ambiguous: Mapping[str, tuple[str, ...]]
    by_capability: Mapping[str, tuple[DeviceTarget, ...]]

    @property
    def empty(self) -> bool:
        return not self.targets

    def candidates_for(self, capability: str | None) -> tuple[DeviceTarget, ...]:
        if capability is None:
            return self.targets
        return self.by_capability.get(capability, ())

    def resolve(self, query: str, capability: str | None, *, min_score: float = 0.55) -> Resolution:
        candidates = self.candidates_for(capability)
        resolution = resolve_target(query, candidates, min_score=min_score)
        if resolution.ok:
            return resolution

        # Fall back to the raw index so an exact alias still lands even when the
        # capability pool was built from a narrower set of targets.
        key = normalize_name(query)
        if key in self.ambiguous:
            return Resolution(None, resolution.score, self.ambiguous[key])
        indexed = self.index.get(key)
        if indexed is not None:
            return Resolution(indexed, 1.0)
        return resolution


def _target_property(
    candidates: Sequence[DeviceTarget],
    arg_name: str,
    max_enum_names: int,
    description: str,
) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": "string", "description": description}
    if max_enum_names <= 0:
        return prop

    all_names = _unique(name for target in candidates for name in target.spoken_names)
    if len(all_names) <= max_enum_names:
        prop["enum"] = all_names
        return prop

    canonical = _unique(target.name for target in candidates)
    if len(canonical) <= max_enum_names:
        prop["enum"] = canonical
    return prop


def build_toolset(
    targets: Sequence[DeviceTarget],
    *,
    max_enum_names: int = 64,
) -> Toolset:
    """Compile device targets into Needle tool schemas.

    Capping how many entities reach the toolset is the registry-discovery
    concern of :mod:`.entities`, not of schema compilation, so it is not done
    here.
    """
    limited = tuple(targets)

    by_capability: dict[str, list[DeviceTarget]] = {}
    for target in limited:
        for capability in target.capabilities:
            by_capability.setdefault(capability, []).append(target)

    index: dict[str, DeviceTarget] = {}
    ambiguous: dict[str, set[str]] = {}
    for target in limited:
        for spoken in target.spoken_names:
            key = normalize_name(spoken)
            if not key:
                continue
            holder = index.get(key)
            if holder is not None and holder.target_id != target.target_id:
                ambiguous.setdefault(key, set()).add(holder.name)
                ambiguous[key].add(target.name)
                index.pop(key, None)
            elif holder is None and key not in ambiguous:
                index[key] = target

    schemas: list[dict[str, Any]] = []
    for tool in TOOL_DEFS:
        candidates = by_capability.get(tool.capability or "", []) if tool.capability else limited
        if not candidates:
            continue

        properties: dict[str, Any] = {
            tool.target_arg: _target_property(
                candidates,
                tool.target_arg,
                max_enum_names,
                f"The name of the {tool.target_arg} to act on, exactly as listed.",
            )
        }
        properties.update(tool.extra_properties)

        schema: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [tool.target_arg, *tool.required_extra],
            },
        }
        if tool.triggers:
            schema["triggers"] = list(tool.triggers)
        schemas.append(schema)

    from .needle import toolset_signature

    return Toolset(
        schemas=tuple(schemas),
        targets=limited,
        signature=toolset_signature(schemas),
        index=index,
        ambiguous={key: tuple(sorted(values)) for key, values in ambiguous.items()},
        by_capability={key: tuple(values) for key, values in by_capability.items()},
    )


# --------------------------------------------------------------------------
# Call decoding
# --------------------------------------------------------------------------
def _coerce_int(value: Any, *, low: int, high: int) -> int:
    number = int(round(float(value)))
    return max(low, min(high, number))


def _coerce_float(value: Any, *, low: float, high: float) -> float:
    number = float(value)
    return max(low, min(high, number))


def plan_action(
    tool_name: str,
    arguments: Mapping[str, Any],
    toolset: Toolset,
    *,
    min_score: float = 0.55,
) -> Action:
    """Decode one Needle tool call into an executable :class:`Action`.

    Raises :class:`ToolError` subclasses when the call cannot be honoured; the
    conversation agent turns those into spoken replies rather than exceptions.
    """
    tool = TOOL_DEFS_BY_NAME.get(tool_name)
    if tool is None:
        raise UnknownToolError(f"unknown tool {tool_name!r}")

    raw_target = arguments.get(tool.target_arg)
    if raw_target is None:
        raise MissingArgumentError(f"{tool_name} was called without '{tool.target_arg}'")

    resolution = toolset.resolve(str(raw_target), tool.capability, min_score=min_score)
    if not resolution.ok:
        if resolution.ambiguous:
            options = ", ".join(resolution.ambiguous)
            raise ResolutionError(f"I found more than one match for {raw_target!r}: {options}. Which one did you mean?")
        raise ResolutionError(f"I could not find a device called {raw_target!r}.")

    target = resolution.target
    assert target is not None

    if tool.name == "set_state":
        state = str(arguments.get("state", "")).lower()
        if state not in ("on", "off"):
            raise MissingArgumentError("set_state needs 'on' or 'off'")
        verb = "turn_on" if state == "on" else "turn_off"
        return Action(
            tool="set_state",
            target=target,
            speech=f"Turned {state} {target.name}.",
            service_domain=target.domain,
            service=verb,
        )

    if tool.name == "set_brightness":
        brightness = _coerce_int(arguments.get("brightness"), low=1, high=100)
        if target.domain == "fan":
            return Action(
                tool="set_brightness",
                target=target,
                speech=f"Set {target.name} to {brightness} percent.",
                service_domain="fan",
                service="set_percentage",
                data={"percentage": brightness},
            )
        return Action(
            tool="set_brightness",
            target=target,
            speech=f"Set {target.name} brightness to {brightness} percent.",
            service_domain="light",
            service="turn_on",
            data={"brightness_pct": brightness},
        )

    if tool.name == "set_temperature":
        temperature = _coerce_float(arguments.get("temperature"), low=-20.0, high=60.0)
        service = "set_temperature"
        return Action(
            tool="set_temperature",
            target=target,
            speech=f"Set {target.name} to {temperature:g} degrees.",
            service_domain=target.domain,
            service=service,
            data={"temperature": temperature},
        )

    if tool.name == "set_position":
        position = _coerce_int(arguments.get("position"), low=0, high=100)
        return Action(
            tool="set_position",
            target=target,
            speech=f"Set {target.name} to {position} percent open.",
            service_domain="cover",
            service="set_cover_position",
            data={"position": position},
        )

    if tool.name == "set_volume":
        volume = _coerce_int(arguments.get("volume"), low=0, high=100)
        return Action(
            tool="set_volume",
            target=target,
            speech=f"Set {target.name} volume to {volume} percent.",
            service_domain="media_player",
            service="volume_set",
            data={"volume_level": round(volume / 100.0, 2)},
        )

    if tool.name == "media_control":
        command = str(arguments.get("command", "")).lower()
        if command not in MEDIA_COMMANDS:
            raise MissingArgumentError(f"media_control does not understand {command!r}")
        service, phrase = MEDIA_COMMANDS[command]
        data: dict[str, Any] = {}
        if command == "mute":
            data["is_volume_muted"] = True
        elif command == "unmute":
            data["is_volume_muted"] = False
        return Action(
            tool="media_control",
            target=target,
            speech=phrase.format(name=target.name),
            service_domain="media_player",
            service=service,
            data=data,
        )

    if tool.name == "lock_unlock":
        state = str(arguments.get("state", "")).lower()
        if state not in ("lock", "unlock"):
            raise MissingArgumentError("lock_unlock needs 'lock' or 'unlock'")
        return Action(
            tool="lock_unlock",
            target=target,
            speech=f"{'Locked' if state == 'lock' else 'Unlocked'} {target.name}.",
            service_domain="lock",
            service=state,
        )

    if tool.name == "control_cover":
        state = str(arguments.get("state", "")).lower()
        if state not in ("open", "close", "stop"):
            raise MissingArgumentError("control_cover needs 'open', 'close' or 'stop'")
        phrase = {"open": "Opened", "close": "Closed", "stop": "Stopped"}[state]
        return Action(
            tool="control_cover",
            target=target,
            speech=f"{phrase} {target.name}.",
            service_domain="cover",
            service=f"{state}_cover",
        )

    if tool.name == "activate_scene":
        return Action(
            tool="activate_scene",
            target=target,
            speech=f"Activated {target.name}.",
            service_domain="scene",
            service="turn_on",
        )

    if tool.name == "run_script":
        return Action(
            tool="run_script",
            target=target,
            speech=f"Ran {target.name}.",
            service_domain="script",
            service="turn_on",
        )

    if tool.name == "get_state":
        return Action(tool="get_state", target=target, speech="")

    raise UnknownToolError(f"tool {tool_name!r} has no execution handler")  # pragma: no cover


# --------------------------------------------------------------------------
# Text-only fallback
# --------------------------------------------------------------------------
_PLEASE = re.compile(r"\s*please\s*[.!]?\s*$", re.IGNORECASE)
_ON_OFF_AFTER = re.compile(r"^\s*(?:turn|switch|power|flip)\s+(?P<state>on|off)\s+(?:the\s+)?(?P<name>.+)$", re.IGNORECASE)
_ON_OFF_BEFORE = re.compile(r"^\s*(?:turn|switch|power|flip)\s+(?:the\s+)?(?P<name>.+?)\s+(?P<state>on|off)$", re.IGNORECASE)
_LOCK = re.compile(r"^\s*(?P<state>lock|unlock)\s+(?:the\s+)?(?P<name>.+)$", re.IGNORECASE)
_COVER = re.compile(r"^\s*(?P<state>open|close|shut)\s+(?:the\s+)?(?P<name>.+)$", re.IGNORECASE)
_DIM = re.compile(r"^\s*(?:dim|brighten)\s+(?:the\s+)?(?P<name>.+?)\s+to\s+(?P<value>\d{1,3})\s*(?:percent|%)?$", re.IGNORECASE)
_SET_TO = re.compile(r"^\s*set\s+(?:the\s+)?(?P<name>.+?)\s+to\s+(?P<value>-?\d{1,3})\s*(?:percent|%)?(?: degrees?)?$", re.IGNORECASE)
_MEDIA = re.compile(r"^\s*(?P<command>play|pause|stop|mute|unmute)\s+(?:the\s+)?(?P<name>.+)$", re.IGNORECASE)


def _clean(text: str) -> str:
    text = _PLEASE.sub("", text or "")
    return text.strip().strip(".!")


def fallback_parse(text: str, toolset: Toolset, *, min_score: float = 0.55) -> Action | None:
    """Deterministic regex fallback used when Needle returns no tool call.

    Returns ``None`` when nothing plausible was recognised; the caller then
    replies with a plain refusal.
    """
    candidate_text = _clean(text)
    if not candidate_text:
        return None

    def attempt(tool: str, name: str, extra: Mapping[str, Any]) -> Action | None:
        try:
            return plan_action(tool, {**extra, _target_key(tool): name}, toolset, min_score=min_score)
        except ToolError:
            return None

    match = _ON_OFF_AFTER.match(candidate_text) or _ON_OFF_BEFORE.match(candidate_text)
    if match:
        action = attempt("set_state", match.group("name"), {"state": match.group("state").lower()})
        if action:
            return action

    match = _LOCK.match(candidate_text)
    if match:
        action = attempt("lock_unlock", match.group("name"), {"state": match.group("state").lower()})
        if action:
            return action

    match = _COVER.match(candidate_text)
    if match:
        state = "close" if match.group("state").lower() == "shut" else match.group("state").lower()
        action = attempt("control_cover", match.group("name"), {"state": state})
        if action:
            return action

    match = _DIM.match(candidate_text)
    if match:
        action = attempt("set_brightness", match.group("name"), {"brightness": int(match.group("value"))})
        if action:
            return action

    match = _SET_TO.match(candidate_text)
    if match:
        value = float(match.group("value"))
        name = match.group("name")
        for tool, key in (
            ("set_temperature", "temperature"),
            ("set_brightness", "brightness"),
            ("set_position", "position"),
            ("set_volume", "volume"),
        ):
            action = attempt(tool, name, {key: value})
            if action:
                return action

    match = _MEDIA.match(candidate_text)
    if match:
        action = attempt("media_control", match.group("name"), {"command": match.group("command").lower()})
        if action:
            return action

    return None


def _target_key(tool_name: str) -> str:
    return TOOL_DEFS_BY_NAME[tool_name].target_arg


# --------------------------------------------------------------------------
# Group synthesis
# --------------------------------------------------------------------------
# Domains where a "group" makes no sense: a scene or a script is already a
# named routine, and calling two of them at once is never what the user meant.
UNGROUPABLE_DOMAINS: Final = frozenset({"scene", "script"})


def _common_capabilities(members: Sequence[DeviceTarget]) -> frozenset[str]:
    """Intersection of what every member can do, so a group call cannot lie."""
    common = set(members[0].capabilities)
    for member in members[1:]:
        common &= set(member.capabilities)
    return frozenset(common) if common else frozenset({CAP_QUERY})


def synthesize_groups(
    targets: Sequence[DeviceTarget],
    *,
    words: Mapping[str, str] | None = None,
) -> list[DeviceTarget]:
    """Add area-scoped and global groups ("Kitchen lights", "All lights").

    A group is a normal :class:`DeviceTarget` whose ``entity_ids`` holds every
    member, so a grouped command becomes one deterministic service call with an
    entity list rather than a per-device guess.
    """
    domain_words = words if words is not None else DOMAIN_WORDS

    used: set[str] = set()
    for target in targets:
        used.update(normalize_name(name) for name in target.spoken_names)

    by_area_domain: dict[tuple[str, str], list[DeviceTarget]] = {}
    by_domain: dict[str, list[DeviceTarget]] = {}
    for target in targets:
        if target.kind != "entity" or target.domain in UNGROUPABLE_DOMAINS:
            continue
        by_domain.setdefault(target.domain, []).append(target)
        if target.area:
            by_area_domain.setdefault((target.area, target.domain), []).append(target)

    groups: list[DeviceTarget] = []

    def add_group(name: str, alternate: str, domain: str, members: list[DeviceTarget], area: str | None) -> None:
        key = normalize_name(name)
        if key in used or not members:
            return
        used.add(key)
        used.add(normalize_name(alternate))
        groups.append(
            DeviceTarget(
                target_id=f"group:{area or 'all'}:{domain}",
                name=name,
                domain=domain,
                entity_ids=tuple(member.entity_id for member in members),
                capabilities=_common_capabilities(members),
                names=(alternate,),
                area=area,
                kind="group",
            )
        )

    def word_for(domain: str) -> str:
        return domain_words.get(domain, f"{domain.replace('_', ' ')}s")

    for (area, domain), members in by_area_domain.items():
        if len(members) < 2:
            continue
        word = word_for(domain)
        add_group(f"{area} {word}", f"{word} in {area}", domain, members, area)

    for domain, members in by_domain.items():
        if len(members) < 2:
            continue
        word = word_for(domain)
        add_group(f"All {word}", f"every {word}", domain, members, None)

    return groups
