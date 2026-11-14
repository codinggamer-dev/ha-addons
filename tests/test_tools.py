"""Unit tests for the pure tool-compilation and decoding logic."""

from __future__ import annotations

import pytest

from custom_components.cactus_home.tools import (
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
    MissingArgumentError,
    ResolutionError,
    UnknownToolError,
    build_toolset,
    fallback_parse,
    normalize_name,
    plan_action,
    resolve_target,
    synthesize_groups,
)

ALL_CAPS = frozenset(
    {
        CAP_STATE,
        CAP_BRIGHTNESS,
        CAP_TEMPERATURE,
        CAP_POSITION,
        CAP_VOLUME,
        CAP_MEDIA,
        CAP_LOCK,
        CAP_COVER,
        CAP_SCENE,
        CAP_SCRIPT,
        CAP_QUERY,
    }
)


def target(name, domain, caps=ALL_CAPS, **kwargs):
    return DeviceTarget(
        target_id=kwargs.pop("target_id", name.lower().replace(" ", "_")),
        name=name,
        domain=domain,
        entity_ids=kwargs.pop("entity_ids", (f"{domain}.{name.lower().replace(' ', '_')}",)),
        capabilities=frozenset(caps),
        **kwargs,
    )


@pytest.fixture
def targets():
    return [
        target("Kitchen Light", "light", names=("the kitchen lamp",), area="Kitchen"),
        target("Desk Fan", "fan", caps={CAP_STATE, CAP_BRIGHTNESS, CAP_QUERY}),
        target("Living Room Thermostat", "climate", caps={CAP_STATE, CAP_TEMPERATURE, CAP_QUERY}),
        target("Bedroom Blind", "cover", caps={CAP_COVER, CAP_POSITION, CAP_QUERY}),
        target("Office Speaker", "media_player", caps={CAP_STATE, CAP_MEDIA, CAP_VOLUME, CAP_QUERY}),
        target("Front Door", "lock", caps={CAP_LOCK, CAP_QUERY}),
        target("Movie Night", "scene", caps={CAP_SCENE}),
        target("Goodnight", "script", caps={CAP_SCRIPT}),
    ]


@pytest.fixture
def toolset(targets):
    return build_toolset(targets, max_enum_names=64)


# --------------------------------------------------------------------------
# Schema generation
# --------------------------------------------------------------------------
def test_every_tool_schema_is_well_formed(toolset):
    names = {schema["name"] for schema in toolset.schemas}
    assert {"set_state", "set_brightness", "set_temperature", "set_position"} <= names
    assert {"set_volume", "media_control", "lock_unlock", "control_cover"} <= names
    assert {"activate_scene", "run_script", "get_state"} <= names

    for schema in toolset.schemas:
        assert schema["description"]
        params = schema["parameters"]
        assert params["type"] == "object"
        assert params["required"], schema["name"]
        for required in params["required"]:
            assert required in params["properties"]


def test_enum_contains_spoken_names(toolset):
    set_state = next(s for s in toolset.schemas if s["name"] == "set_state")
    enum = set_state["parameters"]["properties"]["device"]["enum"]
    assert "Kitchen Light" in enum
    assert "the kitchen lamp" in enum


def test_tools_without_candidates_are_omitted():
    only_lights = build_toolset(
        [target("Lamp", "light", caps={CAP_STATE, CAP_BRIGHTNESS, CAP_QUERY})]
    )
    names = {schema["name"] for schema in only_lights.schemas}
    assert "set_state" in names
    assert "set_brightness" in names
    assert "lock_unlock" not in names
    assert "activate_scene" not in names
    assert "media_control" not in names


def test_enum_is_dropped_when_too_many_names():
    many = [target(f"Light {index}", "light") for index in range(30)]
    toolset = build_toolset(many, max_enum_names=10)
    set_state = next(s for s in toolset.schemas if s["name"] == "set_state")
    assert "enum" not in set_state["parameters"]["properties"]["device"]


def test_signature_is_stable():
    targets = [target("Kitchen Light", "light")]
    assert build_toolset(targets).signature == build_toolset(targets).signature


# --------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------
def test_normalize_name_strips_articles_and_case():
    assert normalize_name("The Kitchen Light!") == "kitchen light"
    assert normalize_name("MY desk fan") == "desk fan"


def test_resolve_exact_and_alias(toolset):
    assert toolset.resolve("Kitchen Light", CAP_STATE).target.name == "Kitchen Light"
    assert toolset.resolve("the kitchen lamp", CAP_STATE).target.name == "Kitchen Light"


def test_resolve_is_fuzzy_about_casing_and_punctuation(toolset):
    resolution = toolset.resolve("kitchen-light", CAP_STATE)
    assert resolution.ok
    assert resolution.target.name == "Kitchen Light"


def test_unknown_target_is_not_resolved(toolset):
    assert not toolset.resolve("garage door opener", CAP_STATE).ok


def test_ambiguous_names_are_reported():
    duplicates = [target("Light", "light", target_id="a"), target("Light", "light", target_id="b")]
    resolution = resolve_target("light", duplicates)
    assert not resolution.ok
    assert resolution.ambiguous


# --------------------------------------------------------------------------
# Call decoding
# --------------------------------------------------------------------------
def test_plan_set_state_on_switch():
    toolset = build_toolset([target("Coffee Maker", "switch")])
    action = plan_action("set_state", {"device": "Coffee Maker", "state": "on"}, toolset)
    assert action.service_domain == "switch"
    assert action.service == "turn_on"
    assert action.speech == "Turned on Coffee Maker."


def test_plan_brightness_for_light_and_fan(toolset):
    light = plan_action("set_brightness", {"device": "Kitchen Light", "brightness": 30}, toolset)
    assert (light.service_domain, light.service) == ("light", "turn_on")
    assert light.data["brightness_pct"] == 30

    fan = plan_action("set_brightness", {"device": "Desk Fan", "brightness": 55}, toolset)
    assert (fan.service_domain, fan.service) == ("fan", "set_percentage")
    assert fan.data["percentage"] == 55


def test_plan_temperature(toolset):
    action = plan_action(
        "set_temperature", {"device": "Living Room Thermostat", "temperature": 21.5}, toolset
    )
    assert action.service_domain == "climate"
    assert action.service == "set_temperature"
    assert action.data["temperature"] == 21.5


def test_plan_position_volume_media_lock_cover_scene_script(toolset):
    position = plan_action("set_position", {"device": "Bedroom Blind", "position": 40}, toolset)
    assert position.data["position"] == 40

    volume = plan_action("set_volume", {"device": "Office Speaker", "volume": 25}, toolset)
    assert volume.data["volume_level"] == 0.25

    media = plan_action("media_control", {"device": "Office Speaker", "command": "pause"}, toolset)
    assert media.service == "media_pause"

    mute = plan_action("media_control", {"device": "Office Speaker", "command": "mute"}, toolset)
    assert mute.data["is_volume_muted"] is True

    lock = plan_action("lock_unlock", {"device": "Front Door", "state": "lock"}, toolset)
    assert lock.service == "lock"

    cover = plan_action("control_cover", {"device": "Bedroom Blind", "state": "open"}, toolset)
    assert cover.service == "open_cover"

    scene = plan_action("activate_scene", {"scene": "Movie Night"}, toolset)
    assert (scene.service_domain, scene.service) == ("scene", "turn_on")

    script = plan_action("run_script", {"script": "Goodnight"}, toolset)
    assert script.service_domain == "script"


def test_get_state_has_no_service(toolset):
    action = plan_action("get_state", {"device": "Kitchen Light"}, toolset)
    assert action.service_domain is None
    assert action.tool == "get_state"


def test_out_of_range_values_are_clamped(toolset):
    action = plan_action("set_brightness", {"device": "Kitchen Light", "brightness": 250}, toolset)
    assert action.data["brightness_pct"] == 100


def test_unknown_tool_is_rejected(toolset):
    with pytest.raises(UnknownToolError):
        plan_action("explode", {"device": "Kitchen Light"}, toolset)


def test_missing_target_is_rejected(toolset):
    with pytest.raises(MissingArgumentError):
        plan_action("set_state", {"state": "on"}, toolset)


def test_unresolvable_target_is_rejected(toolset):
    with pytest.raises(ResolutionError):
        plan_action("set_state", {"device": "Narnia Lamp", "state": "on"}, toolset)


def test_ambiguous_target_asks_for_clarification():
    toolset = build_toolset(
        [target("Light", "light", target_id="a"), target("Light", "light", target_id="b")]
    )
    with pytest.raises(ResolutionError) as err:
        plan_action("set_state", {"device": "Light", "state": "on"}, toolset)
    assert "more than one match" in str(err.value)


# --------------------------------------------------------------------------
# Text fallback
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "tool"),
    [
        ("turn on the Kitchen Light", "set_state"),
        ("switch the Desk Fan off", "set_state"),
        ("dim the Kitchen Light to 20", "set_brightness"),
        ("lock the Front Door", "lock_unlock"),
        ("open the Bedroom Blind", "control_cover"),
        ("pause the Office Speaker", "media_control"),
    ],
)
def test_fallback_parses_common_phrasings(toolset, text, tool):
    action = fallback_parse(text, toolset)
    assert action is not None, text
    assert action.tool == tool


def test_fallback_returns_none_for_chit_chat(toolset):
    assert fallback_parse("what is the meaning of life", toolset) is None


# --------------------------------------------------------------------------
# Group synthesis
# --------------------------------------------------------------------------
def test_area_group_uses_the_capability_intersection():
    members = [
        target("Ceiling", "light", caps={CAP_STATE, CAP_BRIGHTNESS, CAP_QUERY}, area="Kitchen"),
        target("Counter", "light", caps={CAP_STATE, CAP_QUERY}, area="Kitchen"),
    ]
    groups = {group.name: group for group in synthesize_groups(members)}
    kitchen = groups["Kitchen lights"]

    assert kitchen.kind == "group"
    assert kitchen.entity_ids == ("light.ceiling", "light.counter")
    # The counter cannot dim, so the group must not offer brightness either.
    assert kitchen.capabilities == frozenset({CAP_STATE, CAP_QUERY})


def test_area_group_needs_two_members():
    members = [
        target("Ceiling", "light", area="Kitchen"),
        target("Desk", "light", area="Bedroom"),
    ]
    assert "Kitchen lights" not in {group.name for group in synthesize_groups(members)}


def test_global_group_is_synthesised():
    members = [target("Alpha", "light"), target("Beta", "light")]
    assert "All lights" in {group.name for group in synthesize_groups(members)}


def test_scenes_and_scripts_are_never_grouped():
    members = [target("One", "scene"), target("Two", "scene"), target("A", "script"), target("B", "script")]
    assert synthesize_groups(members) == []


def test_group_alternate_name_resolves_through_the_toolset():
    members = [
        target("Ceiling", "light", area="Kitchen"),
        target("Counter", "light", area="Kitchen"),
    ]
    toolset = build_toolset([*members, *synthesize_groups(members)])
    resolution = toolset.resolve("lights in Kitchen", CAP_STATE)

    assert resolution.ok
    assert resolution.target.kind == "group"


def test_group_name_collision_is_skipped():
    members = [
        target("Kitchen lights", "switch", area="Kitchen"),
        target("Ceiling", "light", area="Kitchen"),
        target("Counter", "light", area="Kitchen"),
    ]
    assert "Kitchen lights" not in {group.name for group in synthesize_groups(members)}
