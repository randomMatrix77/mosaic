"""Small typed runtime-event protocol for local Mosaic requests.

The protocol deliberately carries only user-safe lifecycle information. It
never forwards prompts, tool arguments, raw tool syntax, or hidden reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from time import time
from typing import Any, Callable, Mapping


EVENT_TYPES = frozenset({
    "request_started", "model_loading", "model_ready", "prompt_processing",
    "tool_started", "tool_completed", "tool_failed", "answer_started",
    "answer_delta", "answer_completed", "request_cancelled", "request_failed",
})


class GenerationCancelled(RuntimeError):
    """Native generation observed a user cancellation request."""


def _serialize(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass
class RequestEventEmitter:
    """Validates and forwards events to Kotlin or an automated-test sink."""

    request_id: str
    sink: Any | None = None
    _sequence: int = 0

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported Mosaic event type: {event_type}.")
        self._sequence += 1
        # The event payload remains deliberately display-safe. Kotlin owns no
        # model logic, but can now order lifecycle events even while native
        # token deltas arrive directly through JNI.
        event = {
            "request_id": self.request_id,
            "sequence": self._sequence,
            "timestamp_ms": int(time() * 1000),
            "type": event_type,
            **payload,
        }
        if self.sink is None:
            return event
        serialized = _serialize(event)
        # Chaquopy represents all Java objects as potentially callable when
        # they implement a functional interface. Prefer the explicit method
        # contract first: PythonEventSink is deliberately not a SAM callback.
        on_event = getattr(self.sink, "onEvent", None)
        if on_event is not None:
            on_event(serialized)
        elif callable(self.sink):
            self.sink(serialized)
        else:
            raise TypeError("Request event sink must expose onEvent or be callable.")
        return event


class SafeAnswerStreamer:
    """Withhold native control syntax until a user-visible answer is certain.

    Qwen's tool template begins a tool call with ``<tool_call>``. Its optional
    thinking wrapper begins with ``<think>``. The native parser still remains
    authoritative, but delaying a small prefix ensures neither construct ever
    reaches the user as an answer delta.
    """

    _TOOL_PREFIX = "<tool_call>"
    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"

    def __init__(self, emit_delta: Callable[[str], None]) -> None:
        self._emit_delta = emit_delta
        self._pending = ""
        self._mode = "undecided"  # undecided, thinking, answer, tool
        self._started = False

    @property
    def visible_text(self) -> str:
        return self._pending if self._mode == "answer" else ""

    def accept(self, piece: str) -> None:
        if not piece or self._mode == "tool":
            return
        if self._mode == "answer":
            self._emit(piece)
            return
        self._pending += piece
        self._resolve_prefix(final=False)

    def finish(self, *, is_tool_call: bool) -> None:
        if is_tool_call:
            self._pending = ""
            self._mode = "tool"
            return
        self._resolve_prefix(final=True)
        if self._mode == "thinking":
            # A non-thinking request should not normally reach this state. Do
            # not expose an unterminated hidden block if the model does.
            self._pending = ""
            return
        if self._mode == "undecided" and self._pending:
            self._mode = "answer"
            pending, self._pending = self._pending, ""
            self._emit(pending)

    def _resolve_prefix(self, *, final: bool) -> None:
        while self._mode in {"undecided", "thinking"}:
            stripped = self._pending.lstrip()
            leading = self._pending[:len(self._pending) - len(stripped)]
            if self._mode == "undecided":
                if self._TOOL_PREFIX.startswith(stripped) and not final:
                    return
                if stripped.startswith(self._TOOL_PREFIX):
                    self._pending = ""
                    self._mode = "tool"
                    return
                if self._THINK_OPEN.startswith(stripped) and not final:
                    return
                if stripped.startswith(self._THINK_OPEN):
                    self._pending = stripped[len(self._THINK_OPEN):]
                    self._mode = "thinking"
                    continue
                self._mode = "answer"
                pending, self._pending = leading + stripped, ""
                self._emit(pending)
                return

            close_at = self._pending.find(self._THINK_CLOSE)
            if close_at < 0:
                if final:
                    self._pending = ""
                return
            self._pending = self._pending[close_at + len(self._THINK_CLOSE):]
            self._mode = "undecided"

    def _emit(self, piece: str) -> None:
        if not piece:
            return
        self._started = True
        self._emit_delta(piece)
