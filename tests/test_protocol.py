import json
import math
from pathlib import Path
import struct
import unittest
import zlib

from protocol import (
    Direction, Mode, SequenceTracker, VisionTelemetry, decode_can_fd, decode_json,
    encode_can_fd, encode_json, is_actionable,
)


def packet(**updates):
    fields = dict(
        session_id=42, sequence=7, timestamp_ms=12000, capture_age_ms=20,
        valid_for_ms=100, mode=Mode.LIVE, direction=Direction.FORWARDS,
        strike_permit=True, gap_mm=8.25, quality=0.8, scale_valid=True,
        block_reasons=(),
    )
    fields.update(updates)
    return VisionTelemetry(**fields)


def repair_crc(payload):
    payload[44:] = struct.pack("<I", zlib.crc32(payload[:44]) & 0xFFFFFFFF)
    return bytes(payload)


class ProtocolTests(unittest.TestCase):
    def test_json_roundtrip_with_all_required_fields(self):
        source = packet()
        self.assertEqual(decode_json(encode_json(source)), source)
        schema_path = Path(__file__).resolve().parents[1] / "bus_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(source.to_dict()))

    def test_unknown_gap_stays_unknown_and_zero_means_contact(self):
        unknown = packet(strike_permit=False, gap_mm=None, scale_valid=False,
                         direction=Direction.UNKNOWN, block_reasons=("unknown_depth",))
        self.assertIsNone(decode_json(encode_json(unknown)).gap_mm)
        self.assertIsNone(decode_can_fd(encode_can_fd(unknown)).gap_mm)
        self.assertFalse(is_actionable(unknown))
        self.assertEqual(decode_can_fd(encode_can_fd(packet(gap_mm=0))).gap_mm, 0)

    def test_simulation_cannot_carry_permit(self):
        with self.assertRaises(ValueError):
            packet(mode=Mode.SIMULATION)
        simulated = packet(mode=Mode.SIMULATION, strike_permit=False,
                           block_reasons=("simulation",))
        self.assertFalse(is_actionable(simulated))

    def test_inconsistent_permissions_are_rejected(self):
        for changes in (
            dict(direction=Direction.UNKNOWN), dict(gap_mm=None),
            dict(scale_valid=False), dict(quality=0), dict(block_reasons=("occluded",)),
            dict(capture_age_ms=100),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                packet(**changes)

    def test_freshness_uses_total_age_and_strict_boundary(self):
        source = packet()
        self.assertTrue(is_actionable(source, 69, transport_delay_budget_ms=10))
        self.assertFalse(is_actionable(source, 70, transport_delay_budget_ms=10))
        self.assertFalse(is_actionable(source, 0, local_max_age_ms=20))
        self.assertFalse(is_actionable(packet(valid_for_ms=500), 80))
        with self.assertRaises(ValueError):
            is_actionable(source, -1)

    def test_nonfinite_and_boolean_numbers_rejected(self):
        for field in ("gap_mm", "quality"):
            for value in (math.nan, math.inf, -math.inf, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    packet(**{field: value})
        for field in ("sequence", "capture_age_ms", "schema_version"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                packet(**{field: True})

    def test_json_rejects_nonfinite_duplicate_and_extra_fields(self):
        text = encode_json(packet())
        invalid = (
            text.replace('"quality":0.8', '"quality":NaN'),
            text.replace('"quality":0.8', '"quality":1e999'),
            text[:-1] + ',"sequence":8}',
            text[:-1] + ',"new_field":1}',
            '[1,2,3]',
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_json(value)

    def test_invalid_enums_and_types_rejected(self):
        for changes in (
            dict(direction=3), dict(mode=2), dict(session_id=-1),
            dict(sequence=2**32), dict(valid_for_ms=0), dict(strike_permit=1),
            dict(block_reasons=("unknown_label",)),
            dict(block_reasons=("occluded", "occluded")),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                packet(**changes)

    def test_can_fd_layout_and_roundtrip(self):
        source = packet()
        encoded = encode_can_fd(source)
        self.assertEqual(len(encoded), 48)
        self.assertEqual(encoded[:8], b"NGV1\x01\x01\x02\x03")
        self.assertEqual(struct.unpack_from("<I", encoded, 8)[0], 42)
        self.assertEqual(struct.unpack_from("<f", encoded, 28)[0], 8.25)
        self.assertEqual(decode_can_fd(encoded), source)

    def test_binary_quality_never_rounds_up(self):
        source = packet(quality=0.819)
        decoded = decode_can_fd(encode_can_fd(source))
        self.assertLessEqual(decoded.quality, source.quality)
        self.assertLess(source.quality - decoded.quality, 1 / 255)
        with self.assertRaises(ValueError):
            encode_can_fd(packet(quality=0.001))

    def test_corruption_reserved_bytes_and_unknown_bits_rejected(self):
        original = encode_can_fd(packet())
        bad_crc = bytearray(original)
        bad_crc[12] ^= 1
        with self.assertRaises(ValueError):
            decode_can_fd(bytes(bad_crc))
        for offset, value in ((33, 1), (40, 1), (7, 0x80), (39, 0x80), (4, 2)):
            mutated = bytearray(original)
            mutated[offset] = value
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                decode_can_fd(repair_crc(mutated))
        with self.assertRaises(ValueError):
            decode_can_fd(original[:47])

    def test_binary_rejects_simulated_permit_and_nan_gap_even_with_valid_crc(self):
        original = encode_can_fd(packet())
        simulated = bytearray(original)
        simulated[5] = 0
        with self.assertRaises(ValueError):
            decode_can_fd(repair_crc(simulated))
        invalid_depth = bytearray(original)
        invalid_depth[28:32] = struct.pack("<f", math.nan)
        with self.assertRaises(ValueError):
            decode_can_fd(repair_crc(invalid_depth))

    def test_block_reasons_binary_roundtrip(self):
        blocked = packet(strike_permit=False,
                         block_reasons=("stale_frame", "occluded", "camera_moved"))
        self.assertEqual(decode_can_fd(encode_can_fd(blocked)), blocked)
        self.assertFalse(is_actionable(blocked))

    def test_sequence_requires_explicit_session_pairing_and_rejects_replay(self):
        tracker = SequenceTracker()
        source = packet()
        self.assertFalse(tracker.accept(source))
        tracker.reset(42)
        self.assertTrue(tracker.accept(source))
        self.assertFalse(tracker.accept(source))
        self.assertFalse(tracker.accept(packet(sequence=6)))
        self.assertFalse(tracker.accept(packet(session_id=43, sequence=0)))
        self.assertTrue(tracker.accept(packet(sequence=8)))
        tracker.reset(43)
        self.assertTrue(tracker.accept(packet(session_id=43, sequence=0)))

    def test_sequence_rollover_and_half_range_rejection(self):
        tracker = SequenceTracker()
        tracker.reset(42)
        self.assertTrue(tracker.accept(packet(sequence=0xFFFFFFFF)))
        self.assertTrue(tracker.accept(packet(sequence=0)))
        self.assertFalse(tracker.accept(packet(sequence=0xFFFFFFFF)))
        self.assertFalse(tracker.accept(packet(sequence=0x80000000)))
        self.assertTrue(tracker.accept(packet(sequence=1)))


if __name__ == "__main__":
    unittest.main()
