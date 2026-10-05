import ast
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import struct
import tempfile
import time
import unittest
import zlib

from jetpack_single.mapped_bus import (MappedPublisher, MappedSubscriber,
                                       WriterBusyError)
from protocol import Direction, Mode, VisionTelemetry


def packet(session, sequence=1, **changes):
    values = dict(session_id=session, sequence=sequence, capture_age_ms=0,
                  valid_for_ms=1000, mode=Mode.LIVE, direction=Direction.FORWARDS,
                  strike_permit=True, gap_mm=float(sequence * 2 + 1),
                  quality=.8, scale_valid=True)
    values.update(changes)
    return VisionTelemetry(**values)


def publish_process(directory, name, ready, stop):
    with MappedPublisher(name, directory) as publisher:
        publisher.publish(packet(publisher.session_id, 0))
        ready.put(publisher.session_id)
        for sequence in range(1, 2000):
            if stop.wait(.001):
                break
            publisher.publish(packet(publisher.session_id, sequence))


def torn_process(directory, name, ready, enter):
    publisher = MappedPublisher(name, directory)
    publisher.publish(packet(publisher.session_id))
    ready.put(publisher.session_id)
    enter.wait(5)
    with publisher._data.locked():
        publisher._map[44:68] = bytes(24)
        publisher._map.flush()
        ready.put("locked")
        time.sleep(30)


def short_reader_process(directory, name, ready):
    with MappedSubscriber(name, directory) as reader:
        ready.put(reader.read(max_age_ms=1000) is not None)


class JetPackMappedBusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = self.temp.name
        self.name = "ng_mmap_" + secrets.token_hex(6)

    def tearDown(self):
        self.temp.cleanup()

    def test_python38_grammar_and_no_resource_tracker_dependency(self):
        source = (Path(__file__).resolve().parents[1] /
                  "jetpack_single" / "mapped_bus.py").read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 8))
        self.assertNotIn("multiprocessing.shared_memory", source)

    def test_roundtrip_exact96_byte_layout_and_simulation_blocks(self):
        with MappedPublisher(self.name, self.directory) as publisher:
            with MappedSubscriber(self.name, self.directory) as reader:
                self.assertIsNone(reader.read())
                publisher.publish(packet(publisher.session_id))
                snapshot = reader.read()
                self.assertIsNotNone(snapshot)
                self.assertTrue(snapshot.actionable)
                self.assertEqual(snapshot.packet.gap_mm, 3.)
                self.assertEqual(snapshot.publisher_pid, os.getpid())
                self.assertEqual(publisher.path.stat().st_size, 96)
                block = publisher.path.read_bytes()
                self.assertEqual(block[:8], b"NGSHM1\0\0")
                self.assertEqual(struct.unpack_from("<HHHH", block, 8), (1, 44, 48, 1))
                self.assertEqual(block[44:48], b"NGV1")
                publisher.publish(packet(publisher.session_id, 2,
                    mode=Mode.SIMULATION, direction=Direction.UNKNOWN,
                    strike_permit=False, gap_mm=None, scale_valid=False,
                    block_reasons=("simulation",)))
                snapshot = reader.read()
                self.assertFalse(snapshot.actionable)
                self.assertIsNone(snapshot.packet.gap_mm)

    def test_writer_exclusion_bad_session_and_duplicate_rejected(self):
        with MappedPublisher(self.name, self.directory) as publisher:
            with self.assertRaises(WriterBusyError):
                MappedPublisher(self.name, self.directory)
            with self.assertRaises(ValueError):
                publisher.publish(packet(publisher.session_id ^ 1))
            publisher.publish(packet(publisher.session_id))
            with self.assertRaises(ValueError):
                publisher.publish(packet(publisher.session_id))

    def test_source_and_publisher_age_expire(self):
        with MappedPublisher(self.name, self.directory) as publisher:
            with MappedSubscriber(self.name, self.directory) as reader:
                publisher.publish(packet(publisher.session_id, capture_age_ms=35,
                                         valid_for_ms=100))
                self.assertIsNotNone(reader.read(max_age_ms=50))
                time.sleep(.03)
                self.assertIsNone(reader.read(max_age_ms=50))

    def test_unrelated_bus_and_lock_files_are_preserved(self):
        path = Path(self.directory) / self.name
        unrelated = b"U" * 96
        path.write_bytes(unrelated)
        if os.name == "posix":
            path.chmod(0o600)
        with self.assertRaises(ValueError):
            MappedPublisher(self.name, self.directory)
        self.assertEqual(path.read_bytes(), unrelated)
        other_name = self.name + "_other"
        lock = Path(self.directory) / (other_name + ".writer.lock")
        lock.write_bytes(b"unrelated lock contents")
        if os.name == "posix":
            lock.chmod(0o600)
        with self.assertRaises(ValueError):
            MappedPublisher(other_name, self.directory)
        self.assertEqual(lock.read_bytes(), b"unrelated lock contents")

    def test_hardlink_and_posix_symlink_and_permissions_refused(self):
        original = Path(self.directory) / "other_file"
        original.write_bytes(b"protected" * 12)
        original.chmod(0o600)
        os.link(str(original), str(Path(self.directory) / self.name))
        with self.assertRaises(ValueError):
            MappedPublisher(self.name, self.directory)
        self.assertEqual(original.read_bytes(), b"protected" * 12)
        if os.name == "posix":
            symlink_name = self.name + "_symlink"
            os.symlink(str(original), str(Path(self.directory) / symlink_name))
            with self.assertRaises((OSError, ValueError)):
                MappedPublisher(symlink_name, self.directory)
            unsafe_name = self.name + "_permissions"
            unsafe_path = Path(self.directory) / unsafe_name
            unsafe_path.write_bytes(b"P" * 96)
            unsafe_path.chmod(0o666)
            with self.assertRaises(PermissionError):
                MappedPublisher(unsafe_name, self.directory)

    def test_invalid_crc_reserved_bytes_and_future_timestamp_rejected(self):
        with MappedPublisher(self.name, self.directory) as publisher:
            with MappedSubscriber(self.name, self.directory) as reader:
                publisher.publish(packet(publisher.session_id))
                original = publisher._map[:]
                for offset in (36, 44, 92):
                    block = bytearray(original)
                    block[offset] ^= 1
                    if offset != 92:
                        block[92:] = struct.pack("<I", zlib.crc32(block[:92]) & 0xFFFFFFFF)
                    with publisher._data.locked():
                        publisher._map[:] = block
                    self.assertIsNone(reader.read())
                block = bytearray(original)
                struct.pack_into("<Q", block, 24, time.monotonic_ns() + 1000000000)
                block[92:] = struct.pack("<I", zlib.crc32(block[:92]) & 0xFFFFFFFF)
                with publisher._data.locked():
                    publisher._map[:] = block
                self.assertIsNone(reader.read())

    def test_clean_restart_reuses_inode_and_resets_session(self):
        publisher = MappedPublisher(self.name, self.directory)
        with MappedSubscriber(self.name, self.directory) as reader:
            old_session = publisher.session_id
            old_inode = publisher.path.stat().st_ino
            publisher.publish(packet(old_session))
            self.assertTrue(reader.read().actionable)
            publisher.close()
            self.assertIsNone(reader.read())
            self.assertTrue(publisher.path.exists())
            with MappedPublisher(self.name, self.directory) as restarted:
                self.assertNotEqual(restarted.session_id, old_session)
                self.assertEqual(restarted.path.stat().st_ino, old_inode)
                self.assertIsNone(reader.read())
                restarted.publish(packet(restarted.session_id))
                self.assertEqual(reader.read().packet.session_id, restarted.session_id)

    def test_actual_process_reader_exit_does_not_unlink_or_invalidate_bus(self):
        context = mp.get_context("spawn")
        ready = context.Queue()
        with MappedPublisher(self.name, self.directory) as publisher:
            publisher.publish(packet(publisher.session_id))
            process = context.Process(target=short_reader_process,
                                      args=(self.directory, self.name, ready))
            process.start()
            self.assertTrue(ready.get(timeout=10))
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            self.assertTrue(publisher.path.exists())
            publisher.publish(packet(publisher.session_id, 2))
            with MappedSubscriber(self.name, self.directory) as reader:
                self.assertEqual(reader.read().packet.sequence, 2)
        ready.close()

    def test_actual_concurrent_snapshots_and_crash_expiry(self):
        context = mp.get_context("spawn")
        ready, stop = context.Queue(), context.Event()
        process = context.Process(target=publish_process,
                                  args=(self.directory, self.name, ready, stop))
        process.start()
        reader = None
        try:
            session = ready.get(timeout=10)
            with self.assertRaises(WriterBusyError):
                MappedPublisher(self.name, self.directory)
            reader = MappedSubscriber(self.name, self.directory)
            seen = set()
            deadline = time.monotonic() + .25
            while time.monotonic() < deadline:
                snapshot = reader.read(max_age_ms=1000)
                if snapshot is not None:
                    self.assertEqual(snapshot.packet.session_id, session)
                    self.assertEqual(snapshot.packet.gap_mm, snapshot.packet.sequence * 2 + 1)
                    seen.add(snapshot.write_counter)
            self.assertGreater(len(seen), 10)
            process.terminate()
            process.join(5)
            time.sleep(.12)
            self.assertIsNone(reader.read())
            with MappedPublisher(self.name, self.directory) as restarted:
                self.assertNotEqual(restarted.session_id, session)
                self.assertIsNone(reader.read())
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if reader is not None:
                reader.close()
            ready.close()

    def test_killing_writer_mid_copy_rejects_torn_snapshot_and_allows_restart(self):
        context = mp.get_context("spawn")
        ready, enter = context.Queue(), context.Event()
        process = context.Process(target=torn_process,
                                  args=(self.directory, self.name, ready, enter))
        process.start()
        reader = None
        try:
            old_session = ready.get(timeout=10)
            reader = MappedSubscriber(self.name, self.directory)
            enter.set()
            self.assertEqual(ready.get(timeout=10), "locked")
            process.terminate()
            process.join(5)
            self.assertIsNone(reader.read(max_age_ms=1000))
            with MappedPublisher(self.name, self.directory) as restarted:
                self.assertNotEqual(restarted.session_id, old_session)
                restarted.publish(packet(restarted.session_id))
                self.assertTrue(reader.read().actionable)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if reader is not None:
                reader.close()
            ready.close()


if __name__ == "__main__":
    unittest.main()
