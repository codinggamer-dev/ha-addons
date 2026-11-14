"""Thin, thread-safe wrappers around the ``cactus`` runtime.

No Home Assistant imports: everything here runs inside an executor thread and
can be exercised from the command line (see ``scripts/smoke_needle.py``).

The wrappers exist for three reasons:

* ``cactus`` is imported lazily so the integration can report a clean
  "engine unavailable" error instead of failing at module import time.
* Every call into the engine is blocking and CPU bound, so it is serialised
  behind a lock and always invoked from an executor.
* The Needle agent is bound to one toolset; when Home Assistant's entity
  registry changes the toolset, the agent has to be rebuilt. That bookkeeping
  lives here, keyed on a signature of the tool schemas.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Mapping, Sequence

_LOGGER = logging.getLogger(__name__)


class CactusError(RuntimeError):
    """Base class for runtime failures."""


class CactusUnavailable(CactusError):
    """The ``cactus`` package or its engine could not be loaded."""


class CactusRuntimeError(CactusError):
    """The engine loaded but the call failed."""


def _apply_runtime_env(*, telemetry: bool) -> None:
    """Set the documented telemetry opt-outs before the engine is imported."""
    if telemetry:
        return
    os.environ.setdefault("CACTUS_TELEMETRY", "0")
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("CACTUS_DO_NOT_TRACK", "1")


def load_cactus(telemetry: bool = False):
    """Import and return the ``cactus`` package, or raise :class:`CactusUnavailable`."""
    _apply_runtime_env(telemetry=telemetry)
    try:
        import cactus  # noqa: PLC0415 - deliberate lazy import
    except ImportError as err:  # pragma: no cover - environment dependent
        raise CactusUnavailable(
            "the 'cactus-compute' package is not installed; install it with "
            "'pip install cactus-compute>=4.0'"
        ) from err
    return cactus


class CactusTextAgent:
    """A Needle 3 agent bound to one toolset, rebuilt when the toolset changes."""

    def __init__(
        self,
        *,
        weights: str | None = None,
        system: str | None = None,
        max_new_tokens: int = 256,
        threads: int = 0,
        telemetry: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._weights = weights or None
        self._system = system
        self._max_new_tokens = max_new_tokens
        self._threads = threads
        self._telemetry = telemetry
        self._log = logger or _LOGGER

        self._lock = threading.RLock()
        self._cactus: Any | None = None
        self._agent: Any | None = None
        self._signature: str | None = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        """Return True once the engine has been imported successfully."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Import the engine (and pin the thread count) exactly once."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            cactus = load_cactus(self._telemetry)
            if self._threads and self._threads > 0:
                try:
                    cactus.set_threads(int(self._threads))
                except Exception:  # pragma: no cover - engine version dependent
                    self._log.debug("cactus.set_threads(%s) failed", self._threads, exc_info=True)
            self._cactus = cactus
            self._loaded = True
            self._log.info(
                "Cactus runtime ready (Needle 3%s)",
                f", weights={self._weights}" if self._weights else " (base weights)",
            )

    def _ensure_agent(
        self,
        schemas: Sequence[Mapping[str, Any]],
        signature: str,
        system: str | None,
    ) -> Any:
        self.ensure_loaded()
        if self._agent is not None and self._signature == signature:
            return self._agent

        assert self._cactus is not None
        try:
            agent = self._cactus.Cactus(
                tools=list(schemas),
                system=system if system is not None else self._system,
                weights=self._weights,
            )
        except Exception as err:  # pragma: no cover - engine/model dependent
            raise CactusUnavailable(f"could not start the Needle agent: {err}") from err

        self._agent = agent
        self._signature = signature
        self._log.debug("Needle agent (re)built for %d tools", len(schemas))
        return agent

    def complete(
        self,
        text: str,
        *,
        schemas: Sequence[Mapping[str, Any]],
        signature: str,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Run one turn of the agent loop and return the response envelope."""
        with self._lock:
            agent = self._ensure_agent(schemas, signature, system)
            try:
                result = agent.complete(text, max_new_tokens=self._max_new_tokens)
            except Exception as err:
                raise CactusRuntimeError(f"Needle failed to process the request: {err}") from err
            if not isinstance(result, dict):
                raise CactusRuntimeError(f"unexpected Needle response: {result!r}")
            return result

    def reset(self) -> None:
        """Rewind the conversation, keeping the toolset and weights loaded."""
        with self._lock:
            if self._agent is None:
                return
            try:
                self._agent.reset()
            except Exception:  # pragma: no cover - engine version dependent
                self._log.debug("agent.reset() failed", exc_info=True)

    def close(self) -> None:
        """Drop references so the engine can be garbage collected."""
        with self._lock:
            self._agent = None
            self._signature = None
            self._cactus = None
            self._loaded = False


class CactusTranscriber:
    """Whistle speech-to-text, called from an executor thread."""

    def __init__(
        self,
        *,
        weights: str | None = None,
        keywords: Sequence[str] | None = None,
        telemetry: bool = False,
        threads: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._weights = weights or None
        self._keywords = list(keywords) if keywords else None
        self._telemetry = telemetry
        self._threads = threads
        self._log = logger or _LOGGER

        self._lock = threading.RLock()
        self._cactus: Any | None = None
        self._loaded = False
        self._warmed = False

    @property
    def loaded(self) -> bool:
        """Return True once the engine has been imported successfully."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Import the engine exactly once."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            cactus = load_cactus(self._telemetry)
            if self._threads and self._threads > 0:
                try:
                    cactus.set_threads(int(self._threads))
                except Exception:  # pragma: no cover - engine version dependent
                    self._log.debug("cactus.set_threads(%s) failed", self._threads, exc_info=True)
            self._cactus = cactus
            self._loaded = True
            self._log.info("Cactus runtime ready (Whistle)")

    def warm_up(self) -> None:
        """Load the speech weights and run one inference pass.

        Called during config entry setup. Without it the weights load lazily on
        the first spoken command, so a failed download or a broken install
        reaches the user as a bare "speech-to-text failed" instead of a setup
        error that names the cause.
        """
        self.ensure_loaded()
        if self._warmed:
            return
        with self._lock:
            if self._warmed:
                return

            import numpy as np

            assert self._cactus is not None
            try:
                self._cactus.transcribe(
                    np.zeros(16000, dtype=np.float32),
                    language=None,
                    keywords=None,
                    weights=self._weights,
                )
            except Exception as err:
                raise CactusUnavailable(
                    f"the Whistle speech model could not be loaded: {err}"
                ) from err
            self._warmed = True
            self._log.info("Whistle speech model warmed up and ready")

    def transcribe(self, samples, language: str | None = None) -> str:
        """Transcribe 16 kHz mono float samples and return the text."""
        self.ensure_loaded()
        assert self._cactus is not None
        with self._lock:
            try:
                result = self._cactus.transcribe(
                    samples,
                    language=language,
                    keywords=self._keywords,
                    weights=self._weights,
                )
            except Exception as err:
                raise CactusRuntimeError(f"Whistle transcription failed: {err}") from err

        if isinstance(result, Mapping):
            return str(result.get("text") or "").strip()
        if isinstance(result, str):
            return result.strip()
        raise CactusRuntimeError(f"unexpected Whistle response: {result!r}")

    def close(self) -> None:
        """Drop engine references."""
        with self._lock:
            self._cactus = None
            self._loaded = False
            self._warmed = False


def toolset_signature(schemas: Sequence[Mapping[str, Any]]) -> str:
    """Stable hash of a toolset, used to decide when to rebuild the agent."""
    payload = json.dumps(list(schemas), sort_keys=True, separators=(",", ":"), default=str)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
