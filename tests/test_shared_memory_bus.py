import multiprocessing as mp
import os
import queue
import secrets
import struct
import time
import unittest
import zlib

from protocol import Direction, Mode, VisionTelemetry
from shared_memory_bus import (SharedMemoryPublisher, SharedMemorySubscriber,
                               WriterBusyError)


def make_packet(session, sequence=1, **updates):
    fields = dict(session_id=session, sequence=sequence, capture_age_ms=0,
                  valid_for_ms=1000, mode=Mode.LIVE, direction=Direction.FORWARDS,
                  strike_permit=True, gap_mm=float(sequence * 2 + 1),
                  quality=.8, scale_valid=True)
    fields.update(updates)
    return VisionTelemetry(**fields)


def writer_process(name, ready, stop):
    with SharedMemoryPublisher(name) as publisher:
        publisher.publish(make_packet(publisher.session_id, 0))
        ready.put(publisher.session_id)
        for sequence in range(1, 2000):
            if stop.wait(.001):
                break
            publisher.publish(make_packet(publisher.session_id, sequence))


def torn_writer_process(name, ready, enter_lock):
    publisher = SharedMemoryPublisher(name)
    publisher.publish(make_packet(publisher.session_id))
    ready.put(publisher.session_id)
    enter_lock.wait(5)
    with publisher._data_lock.locked():
        publisher._memory.buf[44:68] = bytes(24)  # Simulate a partial crashed copy.
        ready.put("locked")
        time.sleep(30)


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.name = "ng_test_" + secrets.token_hex(8)

    def test_roundtrip_uses_matching_session_and_unknown_stays_null(self):
        with SharedMemoryPublisher(self.name) as publisher:
            with SharedMemorySubscriber(self.name) as reader:
                self.assertIsNone(reader.read())
                publisher.publish(make_packet(publisher.session_id))
                snapshot = reader.read()
                self.assertIsNotNone(snapshot)
                self.assertEqual(snapshot.packet.session_id, publisher.session_id)
                self.assertEqual(snapshot.packet.gap_mm, 3.)
                self.assertTrue(snapshot.actionable)
                self.assertEqual(snapshot.publisher_pid, os.getpid())
                publisher.publish(make_packet(publisher.session_id, 2,
                    mode=Mode.SIMULATION, direction=Direction.UNKNOWN,
                    strike_permit=False, gap_mm=None, scale_valid=False,
                    block_reasons=("simulation", "unknown_depth")))
                snapshot = reader.read()
                self.assertIsNone(snapshot.packet.gap_mm)
                self.assertFalse(snapshot.actionable)

    def test_wrong_session_duplicate_and_second_writer_rejected(self):
        with SharedMemoryPublisher(self.name) as publisher:
            with self.assertRaises(WriterBusyError):
                SharedMemoryPublisher(self.name)
            with self.assertRaises(ValueError):
                publisher.publish(make_packet(publisher.session_id ^ 1))
            publisher.publish(make_packet(publisher.session_id, 3))
            for sequence in (3, 2):
                with self.assertRaises(ValueError):
                    publisher.publish(make_packet(publisher.session_id, sequence))

    def test_capture_age_and_publisher_age_expire_permission(self):
        with SharedMemoryPublisher(self.name) as publisher:
            with SharedMemorySubscriber(self.name) as reader:
                publisher.publish(make_packet(publisher.session_id, capture_age_ms=35,
                                              valid_for_ms=100))
                self.assertIsNotNone(reader.read(max_age_ms=50))
                time.sleep(.03)
                self.assertIsNone(reader.read(max_age_ms=50))

    def test_crc_reserved_fields_and_malformed_payload_rejected(self):
        with SharedMemoryPublisher(self.name) as publisher:
            with SharedMemorySubscriber(self.name) as reader:
                publisher.publish(make_packet(publisher.session_id))
                valid = bytes(publisher._memory.buf[:96])
                for offset in (0, 8, 12, 36, 44, 92):
                    corrupted = bytearray(valid)
                    corrupted[offset] ^= 1
                    if offset != 92:
                        corrupted[92:] = struct.pack("<I", zlib.crc32(corrupted[:92]) & 0xFFFFFFFF)
                    with publisher._data_lock.locked():
                        publisher._memory.buf[:96] = corrupted
                    with self.subTest(offset=offset):
                        self.assertIsNone(reader.read())

    def test_clean_close_clears_old_permission_and_restart_changes_session(self):
        publisher = SharedMemoryPublisher(self.name)
        reader = SharedMemorySubscriber(self.name)
        try:
            old_session = publisher.session_id
            publisher.publish(make_packet(old_session))
            self.assertIsNotNone(reader.read())
            publisher.close()
            self.assertIsNone(reader.read())
            with SharedMemoryPublisher(self.name) as restarted:
                self.assertNotEqual(restarted.session_id, old_session)
                self.assertIsNone(reader.read())
                restarted.publish(make_packet(restarted.session_id))
                if os.name == "nt":
                    self.assertEqual(reader.read().packet.session_id, restarted.session_id)
                else:
                    # POSIX unlink leaves old mappings; explicitly reattach.
                    reader.close()
                    reader = SharedMemorySubscriber(self.name)
                    self.assertEqual(reader.read().packet.session_id, restarted.session_id)
        finally:
            reader.close()
            publisher.close()

    def test_actual_multiprocess_snapshots_are_coherent_and_writer_exclusive(self):
        context = mp.get_context("spawn")
        ready, stop = context.Queue(), context.Event()
        process = context.Process(target=writer_process, args=(self.name, ready, stop))
        process.start()
        reader = None
        try:
            session = ready.get(timeout=10)
            with self.assertRaises(WriterBusyError):
                SharedMemoryPublisher(self.name)
            reader = SharedMemorySubscriber(self.name)
            seen = set()
            deadline = time.monotonic() + .3
            while time.monotonic() < deadline:
                snapshot = reader.read(max_age_ms=1000)
                if snapshot is not None:
                    self.assertEqual(snapshot.packet.session_id, session)
                    self.assertEqual(snapshot.packet.gap_mm, snapshot.packet.sequence * 2 + 1)
                    self.assertTrue(snapshot.actionable)
                    seen.add(snapshot.write_counter)
            self.assertGreater(len(seen), 10)
        finally:
            stop.set()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            if reader is not None:
                reader.close()
            ready.close()
        self.assertEqual(process.exitcode, 0)

    def test_crashed_publisher_expires_and_releases_writer_claim(self):
        context = mp.get_context("spawn")
        ready, stop = context.Queue(), context.Event()
        process = context.Process(target=writer_process, args=(self.name, ready, stop))
        process.start()
        reader = None
        try:
            old_session = ready.get(timeout=10)
            reader = SharedMemorySubscriber(self.name)
            process.terminate()
            process.join(5)
            time.sleep(.12)
            self.assertIsNone(reader.read(max_age_ms=100))
            with SharedMemoryPublisher(self.name) as restarted:
                self.assertNotEqual(restarted.session_id, old_session)
                self.assertIsNone(reader.read())
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if reader is not None:
                reader.close()
            ready.close()

    def test_kill_during_protected_partial_copy_is_rejected(self):
        context = mp.get_context("spawn")
        ready, enter_lock = context.Queue(), context.Event()
        process = context.Process(target=torn_writer_process, args=(self.name, ready, enter_lock))
        process.start()
        reader = None
        try:
            ready.get(timeout=10)
            reader = SharedMemorySubscriber(self.name)
            enter_lock.set()
            self.assertEqual(ready.get(timeout=10), "locked")
            process.terminate()
            process.join(5)
            self.assertIsNone(reader.read(max_age_ms=1000))
            self.assertIsNone(reader.read(max_age_ms=1000))
            with SharedMemoryPublisher(self.name):
                self.assertIsNone(reader.read())
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if reader is not None:
                reader.close()
            ready.close()


if __name__ == "__main__":
    unittest.main()
