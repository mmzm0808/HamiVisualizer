"""Shared identity rules for editable hopping relations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .expression import StrengthExpression, classify_strength_expression


def _as_int(value: object, default: int = -1) -> int:
    """Convert persisted/editor values without letting malformed rows crash UI."""
    try:
        # bool is technically an int, but is never a meaningful site/offset.
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _offset_pair(hop: Mapping[str, object]) -> tuple[int, int]:
    offset = hop.get("cell_offset")
    if offset is not None and isinstance(offset, Sequence) and not isinstance(offset, (str, bytes)):
        if len(offset) >= 2:
            return _as_int(offset[0], 0), _as_int(offset[1], 0)
    return _as_int(hop.get("off_x", 0), 0), _as_int(hop.get("off_y", 0), 0)


def canonical_geometry(hop: Mapping[str, object]) -> tuple[int, int, int, int]:
    """Canonicalize a directed endpoint/offset tuple with its reverse.

    Model files are user-editable.  A partially edited row must remain
    inspectable and comparable instead of raising from an ``int`` conversion
    while the canvas is refreshing.
    """
    if not isinstance(hop, Mapping):
        hop = {}
    fr = _as_int(hop.get("from_site", -1))
    to = _as_int(hop.get("to_site", -1))
    ox, oy = _offset_pair(hop)
    forward = (fr, to, ox, oy)
    reverse = (to, fr, -forward[2], -forward[3])
    return min(forward, reverse)


@dataclass(frozen=True)
class AmplitudeFamily:
    kind: str
    parameter: str | None = None
    signature: str = ""
    identity: str = ""
    editable: bool = False
    reason: str = ""


def classify_amplitude(text: object) -> AmplitudeFamily:
    """Return a stable identity family without using numeric parameter values."""
    try:
        result: StrengthExpression = classify_strength_expression(str(text))
    except Exception as exc:  # defensive boundary for editor/model payloads
        signature = _stable_signature(text)
        return AmplitudeFamily(
            "unsupported", signature=signature,
            identity=f"unsupported:{signature}", reason=str(exc), editable=False,
        )
    if result.kind == "unsupported":
        # Keep distinct malformed/complex source expressions isolated.
        signature = _stable_signature(text)
        return AmplitudeFamily(
            "unsupported", signature=signature,
            identity=f"unsupported:{signature}", reason=result.reason, editable=False,
        )
    identity = result.kind if result.kind == "literal" else f"linear:{result.parameter}"
    return AmplitudeFamily(
        result.kind, result.parameter, identity=identity, editable=True,
    )


def _stable_signature(value: object) -> str:
    return str(value).strip().replace(" ", "")


def _normalized(hop: Mapping[str, object], key: str, default: str) -> str:
    value = hop.get(key, default)
    return str(default if value is None else value).strip().replace(" ", "")


def editor_parameter_key(hop: Mapping[str, object]) -> tuple:
    if not isinstance(hop, Mapping):
        hop = {}
    family = classify_amplitude(hop.get("amplitude", "1.0"))
    return (
        _normalized(hop, "name", "t"),
        _normalized(hop, "phase_mode", "none"),
        _normalized(hop, "phase", "0"),
        family.kind,
        family.parameter,
        family.signature,
    )


def editor_relation_key(hop: dict) -> tuple:
    return canonical_geometry(hop) + editor_parameter_key(hop)
