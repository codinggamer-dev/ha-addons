# Cactus Home — local voice & automation for Home Assistant

A fully native Home Assistant custom integration that runs **Needle 3** as the
conversation/tool-calling agent and **Whistle** as the speech-to-text engine —
both locally, through one Python package (`cactus-compute`).

No cloud calls, no API keys, no YAML. Everything is configured through the
Home Assistant UI.

```
 microphone ──▶ Whistle (STT) ──▶ Needle 3 (tool call) ──▶ Home Assistant service
                    ▲                      ▲
                    │                      │
            16 kHz mono PCM        entity registry ──▶ constrained JSON schemas
```

---

## 1. What you get

| Provides | Type | Backed by |
| --- | --- | --- |
| Speech-to-text | native `stt` platform | `cactus.transcribe` (Whistle) |
| Conversation agent | `AbstractConversationAgent` | `cactus.Cactus` (Needle 3) |

Because Needle 3 is *grammar-constrained* — every token it emits is checked
against your tool schemas — a decoded call is always well-formed. Home
Assistant's exposed entities are compiled into those schemas, so the model can
only name devices that actually exist.

* **Deterministic.** The model never writes the service call. It emits a name
  from a closed set, which this integration resolves against the entity
  registry and turns into a `hass.services.async_call`.
* **Non-blocking.** Every call into `cactus` runs in an executor thread.
* **Never crashes the pipeline.** No tool call, a withheld call, a resolution
  failure or a broken audio stream all end in a spoken reply.

---

## 2. Requirements

* Home Assistant **2025.2** or newer (the integration uses the current
  `conversation` and `stt` platform APIs). Verified against **2026.11.1** on
  Python 3.14.
* Python **3.11+** (Home Assistant's bundled Python).
* `cactus-compute>=4.0` (declared in `manifest.json`, installed automatically
  by Home Assistant).
* Any CPU. ARM64 and x86-64 are both supported by the Cactus runtime.

> First setup downloads the Cactus engine (< 1 MB) and the model weights
> (`needle3.cact` ≈ 35 MB, `whistle.cact`) into `~/.cache/cactus`. Later runs
> are fully offline.

---

## 3. Installation

### Option A — HACS (recommended)

1. HACS → **Integrations** → ⋮ → **Custom repositories**.
2. Add this repository URL, category **Integration**.
3. Install **Cactus Home**, then restart Home Assistant.
4. **Settings → Devices & Services → Add Integration → Cactus Home**.

### Option B — manual

```bash
git clone <this-repo> /tmp/cactus-home
cp -r /tmp/cactus-home/custom_components/cactus_home \
      /config/custom_components/cactus_home
```

Then restart Home Assistant and add the integration from the UI.

If your Home Assistant host has no internet access, install the dependency into
the Home Assistant venv first:

```bash
pip install "cactus-compute>=4.0"
```

---

## 4. Configuration

The setup form (and later, **Configure** on the integration card) exposes:

| Option | Default | Meaning |
| --- | --- | --- |
| Name | `Cactus Home` | Title shown when picking the agent in Assist. |
| Enable the Needle 3 conversation agent | on | Register the agent with Assist. |
| Enable the Whistle speech-to-text engine | on | Expose the STT entity. |
| Only control entities exposed to Assist | on | Restrict tools to entities you exposed. Strongly recommended. |
| Entity domains to expose as tools | 13 common domains | Which domains become tools (`light`, `switch`, `cover`, …). |
| Entities to never expose | empty | Comma-separated `entity_id` blocklist. |
| Path to a tuned Needle `.cact` archive | empty | Use a fine-tuned tool surface instead of base Needle 3. |
| Path to a tuned Whistle `.cact` archive | empty | Use a fine-tuned speech model. |
| Language | `auto` | Whistle language; `auto` follows the pipeline's tag. |
| Confidence needed before acting without asking | `0.4` | Below this, the agent asks for a yes/no confirmation. |
| Maximum tool-calling steps per request | `4` | Multi-step requests (e.g. check then act). |
| Maximum tokens per model turn | `256` | Raise for very large toolsets. |
| Maximum number of entities compiled into tools | `200` | Hard cap; extras are dropped. |
| Maximum names listed per tool | `64` | Larger sets fall back to a free-text name argument. `0` disables enums. |
| Words and names to favour when transcribing | empty | Comma-separated keyword biasing, e.g. `Siobhan, Krzysztof`. |
| Directory to save the last audio clip to | empty | When set, the exact 16 kHz clip handed to Whistle is written to `cactus_last_utterance.wav` there. Debugging only. |
| CPU threads for the model | `0` | `0` = all cores available to the process. |
| Allow anonymous Cactus usage telemetry | **off** | When off, `CACTUS_TELEMETRY=0` and `DO_NOT_TRACK` are set. |
| Seconds to cache the entity/tool snapshot | `5.0` | Lower = picks up registry edits faster. |

### Expose your entities

This is the step people miss. With **Only control entities exposed to Assist**
enabled (the default), the agent can only see entities you have exposed:

1. **Settings → Voice assistants → Expose** tab.
2. Expose the lights, switches, covers, thermostats and media players you want
   to control.
3. Add **aliases** here for any name you say that differs from the entity's
   name ("the big lamp", "living room blinds"). Aliases are compiled into the
   tool schemas, so the model can emit them verbatim.

### Pick the agent and the STT engine

1. **Settings → Voice assistants → your pipeline** (or create one).
2. **Conversation agent → Cactus Home**.
3. **Speech-to-text → Whistle**.

---

## 5. What it understands

Tool count is deliberately small — Needle 3 reads the whole toolset as part of
its prompt and accuracy drops as toolsets grow.

| Tool | Arguments | Home Assistant service |
| --- | --- | --- |
| `set_state` | `device`, `state` = `on`/`off` | `<domain>.turn_on` / `.turn_off` |
| `set_brightness` | `device`, `brightness` 1–100 | `light.turn_on` (`brightness_pct`) or `fan.set_percentage` |
| `set_temperature` | `device`, `temperature` °C | `climate.set_temperature` / `water_heater.set_temperature` |
| `set_position` | `device`, `position` 0–100 | `cover.set_cover_position` |
| `set_volume` | `device`, `volume` 0–100 | `media_player.volume_set` |
| `media_control` | `device`, `command` | `media_player.media_play` / `media_pause` / `media_stop` / `media_next_track` / `media_previous_track` / `volume_mute` |
| `lock_unlock` | `device`, `state` = `lock`/`unlock` | `lock.lock` / `lock.unlock` |
| `control_cover` | `device`, `state` = `open`/`close`/`stop` | `cover.open_cover` / `.close_cover` / `.stop_cover` |
| `activate_scene` | `scene` | `scene.turn_on` |
| `run_script` | `script` | `script.turn_on` |
| `get_state` | `device` | no service — reads the state and speaks it |

Only tools with at least one eligible entity are emitted, so a home with no
locks never sees `lock_unlock`.

### Groups come for free

For every area/domain with two or more members, the integration synthesises a
group target — `Kitchen lights`, `lights in Kitchen` — plus a global
`All lights`. Groups are real targets with the **intersection** of their
members' capabilities, so "turn off the kitchen lights" resolves to one
deterministic list of entity ids instead of a guess.

### Example utterances

```
"turn on the kitchen light"
"switch the desk fan off"
"dim the kitchen light to 20 percent"
"set the living room thermostat to 21 degrees"
"open the bedroom blind"
"lock the front door"
"pause the office speaker"
"activate movie night"
"run goodnight"
"is the front door locked"
```

### When the model is unsure

| Situation | Behaviour |
| --- | --- |
| Tool call with confidence ≥ threshold | executed immediately |
| Tool call with confidence < threshold | agent asks *"I think you want to …, should I do that?"* and waits for yes/no |
| Call withheld by the engine (`suppressed_calls`) | same confirmation flow |
| No tool call at all | deterministic regex fallback over the same target list |
| Fallback finds nothing | spoken refusal |

---

## 6. Project layout

```
custom_components/cactus_home/
├── __init__.py        # config entry: builds runtime objects, registers the agent
├── config_flow.py     # UI setup + options (single entry, validated weights paths)
├── const.py           # domain, option keys, defaults, domain word map
├── conversation.py    # AbstractConversationAgent: the agent loop + service calls
├── stt.py             # SpeechToTextEntity backed by Whistle
├── entities.py        # HA glue: registries → DeviceTarget objects, groups
├── tools.py           # pure logic: tool schemas, name resolution, call decoding
├── needle.py          # pure logic: thread-safe wrappers around `cactus`
├── audio.py           # pure logic: PCM → 16 kHz mono float32
├── manifest.json
├── strings.json
└── translations/en.json
tests/                 # pure logic, plus HA-dependent tests that skip without HA
scripts/smoke_needle.py
.github/workflows/ci.yml
hacs.json
```

The four **pure logic** modules (`tools`, `needle`, `audio`, and the parts of
`entities` that do not touch the registries) import nothing from Home
Assistant. `__init__.py` defers every Home Assistant import to call time, which
is what lets the test suite run in a plain virtualenv.

### Threading model

`cactus` inference is synchronous and CPU-bound. `CactusTextAgent` and
`CactusTranscriber` serialise their calls behind a `threading.Lock`, and the
integration always invokes them through `hass.async_add_executor_job`. The
event loop never blocks. Needle agents are rebuilt only when the compiled
toolset signature actually changes (entity added/removed, alias edited), not on
every turn.

---

## 7. Local development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements_test.txt

# Tool compilation, name resolution, call decoding, audio conversion:
pytest -q
```

`tests/test_tools.py` and `tests/test_audio.py` need nothing but `pytest` and
`numpy`. `tests/test_agent_loop.py` exercises the turn loop against Home
Assistant's real `conversation` types and skips automatically when
`homeassistant` is not installed.

### End-to-end without Home Assistant

`scripts/smoke_needle.py` compiles demo entities, runs the real model, and
prints the service call each utterance maps to:

```bash
pip install "cactus-compute>=4.0"

python scripts/smoke_needle.py --schemas-only        # no model download
python scripts/smoke_needle.py                       # downloads engine + weights
python scripts/smoke_needle.py --say "dim the kitchen light to 20"
python scripts/smoke_needle.py --wav clip.wav        # Whistle transcription
python scripts/smoke_needle.py --synth               # tone should transcribe to ""
```

### Iterating against a live Home Assistant

The integration is a plain directory. Symlink it into a development config dir
and restart:

```bash
ln -s "$PWD/custom_components/cactus_home" /path/to/config/custom_components/
```

Useful environment variables (all documented by Cactus Compute):

| Variable | Purpose |
| --- | --- |
| `CACTUS_LIB_PATH` | Point at a pre-downloaded engine binary (air-gapped hosts). |
| `CACTUS_NEEDLE_WEIGHTS`, `CACTUS_WHISTLE_WEIGHTS` | Point at local `.cact` archives. |
| `CACTUS_TELEMETRY=0`, `DO_NOT_TRACK=1` | Opt out of anonymous usage counts. |
| `HF_HUB_OFFLINE=1` | Fail fast instead of trying to fetch missing files. |

Pre-download everything on a connected machine:

```bash
cactus fetch
cactus download needle3
cactus download whistle
```

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Config entry not ready: the 'cactus-compute' package is not installed` | The dependency could not be installed. Install it in the Home Assistant venv (see §3) and reload. |
| Agent missing from the Assist dropdown | Check **Enable the Needle 3 conversation agent**, then restart. `async_set_agent` registration happens at setup. |
| "I could not find any devices exposed to voice control" | Expose entities under **Voice assistants → Expose**, or turn off **Only control entities exposed to Assist**. |
| "I could not find a device called …" | The name was not recognised. Add an alias in the Expose tab, or lower **Confidence needed before acting**. |
| "I found more than one match for …" | Two entities share a name. Add distinguishing aliases. |
| Wrong device acts | Check the aliases; the resolver prefers exact matches, then token overlap, then edit distance. |
| Transcription returns empty | Whistle detects silence/noise and returns `""` by design. Check the microphone and the pipeline's VAD settings. |
| Transcription invents words for music | Expected — put a VAD or wake word in front of an always-on microphone. |
| Commands are slow | Lower **Maximum tokens per model turn**, raise **CPU threads**, or reduce **Maximum number of entities**. |
| Model misses a command it should know | Likely fit, not a bug: Needle 3's base weights do not pass every vendor acceptance suite. Fine-tune it (see below) or trim the toolset. |

### Fine-tuning the tool surface

The fastest accuracy win is training Needle 3 on *your* home's toolset:

```bash
pip install "cactus-compute[train]"
cactus generate-data --tools my_tools.json --num-samples 500 --output data.jsonl
cactus finetune data.jsonl --epochs 3
cactus build checkpoints/needle3.safetensors --lora checkpoints/needle_lora.safetensors --out my.cact
```

Then set **Path to a tuned Needle `.cact` archive** to `my.cact`. The smoke
script writes the same schemas the integration uses, so you can dump them with
`--schemas-only` and adapt them into a training file.

---

## 9. Privacy

Inference is local. Audio never leaves the machine. When **Allow anonymous
Cactus usage telemetry** is off (the default), the integration sets
`CACTUS_TELEMETRY=0`, `NEEDLE_TELEMETRY=0` and `CACTUS_DO_NOT_TRACK=1` before
the engine is imported. Cactus Compute's telemetry is documented as function
name, package version, OS and a random install id — never prompts, outputs or
audio.

---

## 10. Known limitations

* Needle 3 has **no tool-retrieval head**: the whole toolset is part of the
  prompt, so very large homes are capped and enums fall back to free-text
  names past `max_enum_names`.
* No free-text conversation. Off-topic requests are refused — this is a device
  controller, not a chatbot, and Needle 3 generates no free text.
* Only domains in the enabled set are controllable. Adding a domain means
  adding a service mapping to `PLAN` in `tools.py`.
* Streaming partial transcripts are not implemented; the STT entity returns one
  final result (Home Assistant's default progressive wrapper handles that
  correctly).

---

## 11. Publishing checklist

This repository is laid out the way HACS expects — `hacs.json` plus
`custom_components/cactus_home/` at the root — and is already live at
<https://github.com/codinggamer-dev/ha-addons> with a public repo, description,
topics and a `v1.0.0` release.

Steps that still need doing when you next touch the repo:

- [ ] Add a screenshot of the config flow to the README.
- [ ] Bump `version` in `custom_components/cactus_home/manifest.json` and cut a
      new GitHub release for any user-visible change. HACS reads the latest
      release tag as the version it shows; a tag alone is not enough.

Submitting the repository for inclusion in the HACS store (the default
repositories list) has extra requirements beyond a custom repository: a
[brand icon](https://github.com/home-assistant/brands), the
[HACS Action](https://github.com/hacs/action) and
[hassfest](https://github.com/home-assistant/actions#hassfest) passing with no
ignores, and a PR to [hacs/default](https://github.com/hacs/default). Add those
workflows to `.github/workflows/` only when you are ready for that review.

### Ongoing pushes

`gh` is installed at `~/.local/bin/gh` and authenticated as `codinggamer-dev`.
Send CI a green build before tagging:

```bash
pytest -q
git push                                   # works while the repo-local credential helper is set
```

## License

MIT — see `LICENSE`. `cactus-compute` itself is Apache-2.0.
