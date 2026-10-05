"""Versioned vision recommendations; no actuator or bus I/O occurs here.

The electrical controller decides whether a recommendation may cause movement.
``strike_permit`` is a time-limited vision condition, never a strike command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
import math
import struct
from typing import Any, Mapping
import zlib


SCHEMA_VERSION = 1
CAN_FD_PAYLOAD_SIZE = 48
CAN_FD_MAGIC = b"NGV1"


class Direction(IntEnum):
    BACKWARDS = 0
    UPRIGHT = 1
    FORWARDS = 2
    UNKNOWN = 255


class Mode(IntEnum):
    SIMULATION = 0
    LIVE = 1


# Stable bit assignments for the optional binary representation.
BLOCK_REASON_BITS = {
    "stale_frame": 0,
    "unknown_depth": 1,
    "uncalibrated": 2,
    "low_quality": 3,
    "target_missing": 4,
    "finger_missing": 5,
    "unreachable": 6,
    "simulation": 7,
    "direction_unknown": 8,
    "controller_unarmed": 9,
    "invalid_geometry": 10,
    "occluded": 11,
    "calibration_unvalidated": 12,
    "no_stroke_calibration": 13,
    "camera_moved": 14,
    "depth_out_of_range": 15,
}
_KNOWN_REASON_MASK = sum(1 << bit for bit in BLOCK_REASON_BITS.values())
_WIRE_PREFIX = struct.Struct("<4sBBBBIIQHHfB3xI4x")
_UINT32 = struct.Struct("<I")


def _integer(name: str, value: Any, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _finite(name: str, value: Any, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


@dataclass(frozen=True)
class VisionTelemetry:
    session_id: int
    sequence: int
    capture_age_ms: int
    valid_for_ms: int
    mode: Mode
    direction: Direction
    strike_permit: bool
    gap_mm: float | None
    quality: float
    scale_valid: bool
    block_reasons: tuple[str, ...] = ()
    timestamp_ms: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _integer("schema_version", self.schema_version, SCHEMA_VERSION, SCHEMA_VERSION)
        _integer("session_id", self.session_id, 0, 0xFFFFFFFF)
        _integer("sequence", self.sequence, 0, 0xFFFFFFFF)
        _integer("timestamp_ms", self.timestamp_ms, 0, 0xFFFFFFFFFFFFFFFF)
        _integer("capture_age_ms", self.capture_age_ms, 0, 0xFFFF)
        _integer("valid_for_ms", self.valid_for_ms, 1, 0xFFFF)
        _integer("mode", self.mode, 0, 1)
        _integer("direction", self.direction, 0, 255)
        try:
            object.__setattr__(self, "mode", Mode(self.mode))
            object.__setattr__(self, "direction", Direction(self.direction))
        except ValueError as exc:
            raise ValueError("unsupported mode or direction") from exc
        if not isinstance(self.strike_permit, bool) or not isinstance(self.scale_valid, bool):
            raise ValueError("strike_permit and scale_valid must be booleans")
        if self.gap_mm is not None:
            object.__setattr__(self, "gap_mm", _finite("gap_mm", self.gap_mm, 0, 1_000_000))
        object.__setattr__(self, "quality", _finite("quality", self.quality, 0, 1))
        if not isinstance(self.block_reasons, (tuple, list)):
            raise ValueError("block_reasons must be a list or tuple")
        if any(not isinstance(reason, str) or reason not in BLOCK_REASON_BITS
               for reason in self.block_reasons):
            raise ValueError("unsupported block reason")
        if len(set(self.block_reasons)) != len(self.block_reasons):
            raise ValueError("duplicate block reason")
        object.__setattr__(self, "block_reasons", tuple(self.block_reasons))
        if self.strike_permit:
            if self.mode != Mode.LIVE:
                raise ValueError("SIMULATION packets must never carry strike_permit=true")
            if self.direction == Direction.UNKNOWN or not self.scale_valid or self.gap_mm is None:
                raise ValueError("permit requires known direction, metric gap, and valid scale")
            if self.block_reasons or self.quality <= 0 or self.capture_age_ms >= self.valid_for_ms:
                raise ValueError("blocked, zero-quality, or stale packets cannot carry a permit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "capture_age_ms": self.capture_age_ms,
            "valid_for_ms": self.valid_for_ms,
            "mode": self.mode.name,
            "direction": int(self.direction),
            "strike_permit": self.strike_permit,
            "gap_mm": self.gap_mm,
            "quality": self.quality,
            "scale_valid": self.scale_valid,
            "block_reasons": list(self.block_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisionTelemetry":
        if not isinstance(value, Mapping):
            raise ValueError("telemetry must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("telemetry must contain exactly the schema v1 fields")
        fields = dict(value)
        if fields["mode"] not in ("SIMULATION", "LIVE"):
            raise ValueError("mode must be SIMULATION or LIVE")
        fields["mode"] = Mode[fields["mode"]]
        return cls(**fields)


def encode_json(packet: VisionTelemetry) -> str:
    if not isinstance(packet, VisionTelemetry):
        raise ValueError("expected VisionTelemetry")
    return json.dumps(packet.to_dict(), allow_nan=False, separators=(",", ":"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON token: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(payload: str | bytes) -> VisionTelemetry:
    try:
        value = json.loads(payload, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("invalid JSON telemetry") from exc
    return VisionTelemetry.from_dict(value)


def is_actionable(
    packet: VisionTelemetry,
    elapsed_since_receipt_ms: int = 0,
    *,
    transport_delay_budget_ms: int = 0,
    local_max_age_ms: int = 100,
) -> bool:
    """Check vision freshness/conditions, not actuator authorization.

    ``capture_age_ms`` is measured on the source. Add receiver elapsed time and
    a verified transport delay bound; source/receiver monotonic clocks differ.
    The default zero transport budget makes this a bench freshness check only.
    A hardware receiver must establish its bus delay budget and local interlocks.
    """
    _integer("elapsed_since_receipt_ms", elapsed_since_receipt_ms, 0, 0xFFFFFFFFFFFFFFFF)
    _integer("transport_delay_budget_ms", transport_delay_budget_ms, 0, 0xFFFFFFFFFFFFFFFF)
    _integer("local_max_age_ms", local_max_age_ms, 1, 0xFFFF)
    age = packet.capture_age_ms + elapsed_since_receipt_ms + transport_delay_budget_ms
    return bool(
        packet.mode == Mode.LIVE
        and packet.strike_permit
        and packet.direction != Direction.UNKNOWN
        and packet.scale_valid
        and packet.gap_mm is not None
        and packet.quality > 0
        and not packet.block_reasons
        and age < min(packet.valid_for_ms, local_max_age_ms)
    )


def encode_can_fd(packet: VisionTelemetry) -> bytes:
    """Build the optional 48-byte CAN-FD data payload; do not transmit it."""
    flags = int(packet.strike_permit) | (int(packet.scale_valid) << 1)
    reason_mask = sum(1 << BLOCK_REASON_BITS[reason] for reason in packet.block_reasons)
    # Use floor so binary quantization never increases the source quality score.
    quality_u8 = int(packet.quality * 255)
    if packet.strike_permit and quality_u8 == 0:
        raise ValueError("permitted packet quality cannot be represented in binary")
    prefix = _WIRE_PREFIX.pack(
        CAN_FD_MAGIC, packet.schema_version, int(packet.mode), int(packet.direction), flags,
        packet.session_id, packet.sequence, packet.timestamp_ms,
        packet.capture_age_ms, packet.valid_for_ms,
        -1.0 if packet.gap_mm is None else packet.gap_mm, quality_u8, reason_mask,
    )
    return prefix + _UINT32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


def decode_can_fd(payload: bytes) -> VisionTelemetry:
    if not isinstance(payload, (bytes, bytearray)) or len(payload) != CAN_FD_PAYLOAD_SIZE:
        raise ValueError("CAN-FD telemetry must contain exactly 48 bytes")
    prefix = payload[:44]
    if (_UINT32.unpack(payload[44:])[0] != zlib.crc32(prefix) & 0xFFFFFFFF):
        raise ValueError("CAN-FD telemetry CRC mismatch")
    if payload[33:36] != bytes(3) or payload[40:44] != bytes(4):
        raise ValueError("reserved CAN-FD bytes must be zero")
    (magic, version, mode, direction, flags, session, sequence, timestamp,
     capture_age, valid_for, gap, quality, reasons) = _WIRE_PREFIX.unpack(prefix)
    if magic != CAN_FD_MAGIC or flags & ~3 or reasons & ~_KNOWN_REASON_MASK:
        raise ValueError("unsupported CAN-FD magic, flags, or reason bits")
    return VisionTelemetry(
        session_id=session, sequence=sequence, capture_age_ms=capture_age,
        valid_for_ms=valid_for, mode=mode, direction=direction,
        strike_permit=bool(flags & 1), gap_mm=None if gap == -1.0 else gap,
        quality=quality / 255.0, scale_valid=bool(flags & 2),
        block_reasons=tuple(reason for reason, bit in BLOCK_REASON_BITS.items() if reasons & (1 << bit)),
        timestamp_ms=timestamp, schema_version=version,
    )


class SequenceTracker:
    """Reject duplicate/out-of-order uint32 counters within an authorized session.

    ``reset(session_id)`` must be called through a receiver's session handshake
    or manual bench pairing. A changed session never silently becomes accepted.
    Session IDs and CRCs are not authentication or protection against attackers.
    """

    def __init__(self) -> None:
        self.session_id: int | None = None
        self.sequence: int | None = None

    def reset(self, session_id: int) -> None:
        self.session_id = _integer("session_id", session_id, 0, 0xFFFFFFFF)
        self.sequence = None

    def accept(self, packet: VisionTelemetry) -> bool:
        if self.session_id is None or packet.session_id != self.session_id:
            return False
        if self.sequence is not None:
            delta = (packet.sequence - self.sequence) & 0xFFFFFFFF
            if delta == 0 or delta >= 0x80000000:
                return False
        self.sequence = packet.sequence
        return True
