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
    device_keywords,
    fallback_parse,
    normalize_name,
    plan_action,
    resolve_target,
    synthesize_groups,
    TOOL_DEFS_BY_NAME,
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
# Compound names (split or joined by the speech model)
# --------------------------------------------------------------------------
def compound_toolset(*names):
    return build_toolset(
        [target(name, "light", caps={CAP_STATE, CAP_QUERY}) for name in names]
    )


def test_resolves_a_compound_that_speech_split_in_two():
    toolset = compound_toolset("Lichtstreifen")

    resolution = toolset.resolve("Licht Streifen", CAP_STATE)

    assert resolution.ok
    assert resolution.target.name == "Lichtstreifen"


def test_resolves_a_joined_name_against_a_split_entity():
    toolset = compound_toolset("Licht Streifen")

    assert toolset.resolve("Lichtstreifen", CAP_STATE).ok


def test_compound_matching_keeps_unrelated_compounds_apart():
    # Differs only in the prefix, so a loose character comparison would match.
    toolset = compound_toolset("Wohnzimmerlicht")

    assert not toolset.resolve("Schlafzimmerlicht", CAP_STATE).ok


def test_compound_matching_does_not_match_a_different_device():
    toolset = compound_toolset("Living Room Light")

    assert not toolset.resolve("kitchenlight", CAP_STATE).ok


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


# --------------------------------------------------------------------------
# Keyword biasing for speech recognition
# --------------------------------------------------------------------------
def test_device_keywords_take_canonical_names_before_aliases():
    targets = [
        target("Lichtstreifen", "light", caps={CAP_STATE, CAP_QUERY}, names=("der streifen",)),
        target("Stehlampe", "light", caps={CAP_STATE, CAP_QUERY}, names=("die lampe",)),
    ]

    assert device_keywords(targets, limit=2) == ["Lichtstreifen", "Stehlampe"]


def test_device_keywords_are_capped():
    targets = [target(f"Lamp {index}", "light", caps={CAP_STATE, CAP_QUERY}) for index in range(10)]

    assert len(device_keywords(targets, limit=3)) == 3


def test_device_keywords_skip_groups_and_long_names():
    grouped = DeviceTarget(
        target_id="group:all:light",
        name="All lights",
        domain="light",
        entity_ids=("light.a",),
        capabilities=frozenset({CAP_QUERY}),
        kind="group",
    )
    verbose = target(
        "The very long descriptive name of a lamp", "light", caps={CAP_STATE, CAP_QUERY}
    )

    assert device_keywords([grouped, verbose], max_words=4) == []


def test_device_keywords_dedupe_case_insensitively_and_skip_short_names():
    targets = [
        target("Lamp", "light", caps={CAP_STATE, CAP_QUERY}, names=("lamp",)),
        target("A", "light", caps={CAP_STATE, CAP_QUERY}),
    ]

    assert device_keywords(targets) == ["Lamp"]


# --------------------------------------------------------------------------
# Prompt budget: the whole toolset is part of the model prompt
# --------------------------------------------------------------------------
def target_enums(toolset):
    """Return the device-name enum of every tool that has one."""
    enums = []
    for schema in toolset.schemas:
        arg = TOOL_DEFS_BY_NAME[schema["name"]].target_arg
        prop = schema["parameters"]["properties"].get(arg, {})
        if "enum" in prop:
            enums.append(prop["enum"])
    return enums


def test_total_budget_keeps_lists_that_fit_the_home():
    targets = [
        target(f"Lamp {index}", "light", caps={CAP_STATE, CAP_QUERY}) for index in range(30)
    ]

    toolset = build_toolset(targets, max_enum_names=512, max_total_names=120)
    enums = target_enums(toolset)

    assert enums, "the tools still carry names"
    assert sum(len(enum) for enum in enums) <= 120
    assert all(len(enum) == 30 for enum in enums)


def test_total_budget_drops_lists_that_cannot_fit():
    targets = [
        target(f"Lamp {index}", "light", caps={CAP_STATE, CAP_QUERY}) for index in range(40)
    ]

    toolset = build_toolset(targets, max_enum_names=512, max_total_names=20)

    # Two tools, so the cap is ten names each, and forty names cannot fit in
    # ten. The lists are dropped rather than truncated: see
    # test_no_partial_name_list_is_ever_emitted.
    assert target_enums(toolset) == []
    # The devices are still addressable, the resolver does the matching.
    assert toolset.resolve("Lamp 37", CAP_STATE).ok


def test_no_partial_name_list_is_ever_emitted():
    """A truncated list would let the model name the wrong device.

    The schema grammar only permits a name that is listed, so a list holding
    some but not all devices can force the model to answer with a device the
    user never mentioned. Either every name is listed or none is.
    """
    targets = [
        target(f"Lamp {index}", "light", caps={CAP_STATE, CAP_QUERY}) for index in range(40)
    ]

    for budget in (0, 5, 20, 39, 79):
        toolset = build_toolset(targets, max_enum_names=512, max_total_names=budget)
        for enum in target_enums(toolset):
            assert len(enum) == 40

    # A budget of eighty names is exactly two tools of forty, so the lists fit.
    fitting = build_toolset(targets, max_enum_names=512, max_total_names=80)
    assert all(len(enum) == 40 for enum in target_enums(fitting))


def test_zero_total_budget_disables_name_lists():
    targets = [
        target(f"Lamp {index}", "light", caps={CAP_STATE, CAP_QUERY}) for index in range(5)
    ]

    toolset = build_toolset(targets, max_total_names=0)

    assert target_enums(toolset) == []
    # The devices are still addressable, just not listed in the prompt.
    assert toolset.resolve("Lamp 3", CAP_STATE).ok


def test_per_tool_ceiling_drops_the_list_when_it_cannot_hold_every_name():
    targets = [
        target(f"Lamp {index}", "light", caps={CAP_STATE, CAP_QUERY}) for index in range(6)
    ]

    # A huge total budget means "no total budget" and leaves the per-tool
    # ceiling as the only constraint.
    assert target_enums(build_toolset(targets, max_enum_names=4, max_total_names=10**6)) == []

    enums = target_enums(build_toolset(targets, max_enum_names=6, max_total_names=10**6))
    assert enums
    assert all(len(enum) == 6 for enum in enums)


def test_number_words_are_folded_into_names():
    """An alias spelled with words matches an entity named with digits."""
    assert normalize_name("Licht drei") == normalize_name("Licht 3")
    assert normalize_name("Light three") == normalize_name("Light 3")
