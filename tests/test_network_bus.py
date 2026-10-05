import time
import unittest

from network_bus import UDPReceiver, UDPPublisher
from protocol import Direction, Mode, VisionTelemetry


def packet(sequence=1, session=42, **updates):
    values = dict(session_id=session, sequence=sequence, capture_age_ms=0,
                  valid_for_ms=1000, mode=Mode.LIVE, direction=Direction.FORWARDS,
                  strike_permit=True, gap_mm=5., quality=.8, scale_valid=True)
    values.update(updates)
    return VisionTelemetry(**values)


class UDPTests(unittest.TestCase):
    def setUp(self):
        # Every test binds and transmits ONLY loopback on dynamically chosen ports.
        self.receiver = UDPReceiver("127.0.0.1", 0, paired_session_id=42,
                                    transport_delay_bound_ms=0, max_age_ms=100)
        self.sender = UDPPublisher(*self.receiver.local_address)
        self.receiver.expected_peer = self.sender.local_address

    def tearDown(self):
        self.sender.close()
        self.receiver.close()

    def send(self, value):
        self.sender.publish(value)
        return self.receiver.receive(timeout=.2)

    def test_exact_payload_roundtrip_and_expiry(self):
        result = self.send(packet())
        self.assertTrue(result.actionable)
        self.assertEqual(result.packet, packet())
        time.sleep(.12)
        self.assertIsNone(self.receiver.latest())

    def test_wrong_peer_corruption_and_wrong_length_are_rejected(self):
        with UDPPublisher(*self.receiver.local_address) as stranger:
            stranger.publish(packet())
            self.assertIsNone(self.receiver.receive(timeout=.2))
        for payload in (b"bad", bytes(48), bytes(49)):
            self.sender._socket.send(payload)
            self.assertIsNone(self.receiver.receive(timeout=.2))
        self.assertIsNone(self.receiver.latest())

    def test_duplicate_and_backward_packets_do_not_refresh_watchdog(self):
        self.receiver.max_age_ms = 70
        self.assertTrue(self.send(packet(3)).actionable)
        time.sleep(.04)
        self.assertIsNone(self.send(packet(3)))
        self.assertIsNone(self.send(packet(2)))
        time.sleep(.04)
        self.assertIsNone(self.receiver.latest())

    def test_changed_session_revokes_pairing_until_explicit_reset(self):
        self.assertTrue(self.send(packet()).actionable)
        restarted = self.send(packet(sequence=0, session=43))
        self.assertIsNotNone(restarted)
        self.assertFalse(restarted.actionable)
        self.assertIsNone(self.receiver.paired_session_id)
        self.assertFalse(self.send(packet(sequence=1, session=43)).actionable)
        self.receiver.pair_session(43)
        self.assertTrue(self.send(packet(sequence=2, session=43)).actionable)

    def test_unverified_delay_unpaired_or_unpinned_peer_never_actionable(self):
        self.receiver.transport_delay_bound_ms = None
        self.assertFalse(self.send(packet()).actionable)
        self.receiver.transport_delay_bound_ms = 0
        self.receiver.expected_peer = None
        self.assertFalse(self.send(packet(2)).actionable)
        self.receiver.expected_peer = self.sender.local_address
        self.receiver.pair_session(43)
        self.assertFalse(self.send(packet(3, session=42)).actionable)

    def test_delay_bound_counts_against_total_frame_age(self):
        self.receiver.transport_delay_bound_ms = 80
        self.assertIsNone(self.send(packet(capture_age_ms=25)))

    def test_simulation_data_cannot_be_actionable(self):
        simulated = packet(mode=Mode.SIMULATION, direction=Direction.UNKNOWN,
                           strike_permit=False, gap_mm=None, scale_valid=False,
                           block_reasons=("simulation",))
        self.assertFalse(self.send(simulated).actionable)

    def test_sequence_rollover_accepted_and_old_packets_rejected(self):
        self.assertTrue(self.send(packet(sequence=0xFFFFFFFF)).actionable)
        self.assertTrue(self.send(packet(sequence=0)).actionable)
        self.assertIsNone(self.send(packet(sequence=0xFFFFFFFF)))


if __name__ == "__main__":
    unittest.main()
