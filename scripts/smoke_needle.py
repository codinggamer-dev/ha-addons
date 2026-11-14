#!/usr/bin/env python3
"""End-to-end smoke test for Needle 3 and Whistle without Home Assistant.

It exercises the exact code paths the integration uses:

* compile demo entities into Needle tool schemas (``tools.build_toolset``);
* decode a handful of natural-language commands with the real model, then map
  each decoded call onto a Home Assistant service call (``tools.plan_action``);
* optionally transcribe a WAV file with Whistle, or a synthesised tone.

Usage:

    python scripts/smoke_needle.py
    python scripts/smoke_needle.py --say "dim the kitchen light to 20"
    python scripts/smoke_needle.py --wav clip.wav
    python scripts/smoke_needle.py --synth            # generate a test tone and transcribe it

The first run downloads the Cactus engine and the Needle/Whistle weights into
the cache (``~/.cache/cactus`` by default) and needs network access. Later runs
are fully offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_components.cactus_home import audio  # noqa: E402
from custom_components.cactus_home.needle import (  # noqa: E402
    CactusError,
    CactusTextAgent,
    CactusTranscriber,
    load_cactus,
)
from custom_components.cactus_home.tools import (  # noqa: E402
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
    ToolError,
    build_toolset,
    fallback_parse,
    plan_action,
)

QUERIES = [
    "turn on the kitchen light",
    "switch the desk fan off",
    "dim the kitchen light to 20 percent",
    "set the living room thermostat to 21 degrees",
    "open the bedroom blind",
    "lock the front door",
    "pause the office speaker",
    "activate movie night",
    "run goodnight",
    "is the front door locked",
    "what is the weather tomorrow",
]


def demo_targets() -> list[DeviceTarget]:
    def make(name, domain, caps, entity_id=None, **kwargs):
        return DeviceTarget(
            target_id=entity_id or f"{domain}.{name.lower().replace(' ', '_')}",
            name=name,
            domain=domain,
            entity_ids=(entity_id or f"{domain}.{name.lower().replace(' ', '_')}",),
            capabilities=frozenset(caps),
            **kwargs,
        )

    every = {
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
    return [
        make("Kitchen Light", "light", every, names=("the kitchen lamp",), area="Kitchen"),
        make("Kitchen Ceiling", "light", every, area="Kitchen"),
        make("Desk Fan", "fan", {CAP_STATE, CAP_BRIGHTNESS, CAP_QUERY}),
        make(
            "Living Room Thermostat",
            "climate",
            {CAP_STATE, CAP_TEMPERATURE, CAP_QUERY},
            area="Living Room",
        ),
        make("Bedroom Blind", "cover", {CAP_COVER, CAP_POSITION, CAP_QUERY}, area="Bedroom"),
        make(
            "Office Speaker",
            "media_player",
            {CAP_STATE, CAP_MEDIA, CAP_VOLUME, CAP_QUERY},
            area="Office",
        ),
        make("Front Door", "lock", {CAP_LOCK, CAP_QUERY}),
        make("Movie Night", "scene", {CAP_SCENE}),
        make("Goodnight", "script", {CAP_SCRIPT}),
    ]


def show_schemas(toolset) -> None:
    print(f"Compiled {len(toolset.schemas)} tools from {len(toolset.targets)} targets:")
    for schema in toolset.schemas:
        properties = sorted(schema["parameters"]["properties"])
        enum = schema["parameters"]["properties"].get("device", {}).get("enum") or schema[
            "parameters"
        ]["properties"].get("scene", {}).get("enum")
        suffix = f" ({len(enum)} names)" if enum else ""
        print(f"  - {schema['name']}({', '.join(properties)}){suffix}")
    print()


def decode_queries(toolset, queries, max_new_tokens: int) -> None:
    runtime = CactusTextAgent(max_new_tokens=max_new_tokens, telemetry=False)
    print("Loading Needle 3 (downloads on first use)...")
    runtime.ensure_loaded()

    for query in queries:
        print(f"> {query}")
        try:
            turn = runtime.complete(
                query, schemas=toolset.schemas, signature=toolset.signature
            )
        except CactusError as err:
            print(f"  error: {err}\n")
            continue

        calls = turn.get("function_calls") or []
        suppressed = turn.get("suppressed_calls") or []
        print(f"  confidence={turn.get('confidence')} reasoning={turn.get('reasoning')!r}")

        if calls:
            for call in calls:
                try:
                    action = plan_action(call["name"], call.get("arguments") or {}, toolset)
                except ToolError as err:
                    print(f"  !! {call['name']} {call.get('arguments')} -> {err}")
                    continue
                target = ", ".join(action.target.entity_ids)
                if action.service_domain:
                    print(
                        f"  -> {action.service_domain}.{action.service} "
                        f"data={json.dumps({**action.data, 'entity_id': target})}"
                    )
                else:
                    print(f"  -> read state of {target}")
                print(f"     would say: {action.speech or '(state read)'}")
        elif suppressed:
            print(f"  (withheld, low confidence): {suppressed}")
        else:
            fallback = fallback_parse(query, toolset)
            if fallback is not None:
                print(f"  no tool call; text fallback -> {fallback.tool}: {fallback.speech}")
            else:
                print("  refused (no matching tool)")
        print()


def transcribe_file(path: str, language: str | None) -> None:
    print(f"Transcribing {path} with Whistle...")
    transcriber = CactusTranscriber(telemetry=False)
    transcriber.ensure_loaded()
    import wave

    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = audio.to_whistle_samples(
        frames, sample_width=width * 8, channels=channels, sample_rate=rate
    )
    print(f"  audio: {channels}ch {width * 8}bit {rate}Hz -> {samples.shape[0]} samples @16kHz")
    print(f"  text: {transcriber.transcribe(samples, language=language)!r}")


def synth_and_transcribe(language: str | None) -> None:
    """Whistle should return empty text for a pure tone: a clean failure path."""
    import numpy as np

    samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000 * 2) / 16000)).astype("float32")
    path = Path(tempfile.gettempdir()) / "cactus_tone.wav"
    audio.write_wav16(str(path), samples)
    transcribe_file(str(path), language)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--say", action="append", default=[], help="Extra command to decode")
    parser.add_argument("--wav", help="WAV file to transcribe with Whistle")
    parser.add_argument("--synth", action="store_true", help="Transcribe a synthesised tone")
    parser.add_argument("--language", default=None, help="Whistle language code (en/de/fr/... )")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--schemas-only", action="store_true", help="Skip loading the model")
    args = parser.parse_args()

    toolset = build_toolset(demo_targets(), max_enum_names=64)
    show_schemas(toolset)

    if args.schemas_only:
        return 0

    try:
        load_cactus(telemetry=False)
    except CactusError as err:
        print(f"Cactus is not available: {err}")
        print("Install it with: pip install 'cactus-compute>=4.0'")
        return 1

    decode_queries(toolset, [*args.say, *QUERIES], args.max_new_tokens)

    if args.wav:
        transcribe_file(args.wav, args.language)
    if args.synth:
        synth_and_transcribe(args.language)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
