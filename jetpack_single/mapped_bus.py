"""Python 3.8 file-backed local telemetry for JetPack 5.1.2.

Linux: /dev/shm/<name> is a 96-byte mmap file; flock protects copies and a
separate lifetime writer lease. No multiprocessing resource tracker is used.
Windows byte-range locks are a test fallback, not the NEON deployment path.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import errno
import math
import mmap
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import tempfile
import threading
import time
import zlib

from protocol import (VisionTelemetry, SequenceTracker, encode_can_fd,
                      decode_can_fd, is_actionable)


DEFAULT_NAME = "neurogrip_vision_v1"
BLOCK_SIZE = 96
HEADER_SIZE = 44
MAGIC = b"NGSHM1\x00\x00"
_HEADER = struct.Struct("<8sHHHHIIQI8x")
_CRC = struct.Struct("<I")


class WriterBusyError(RuntimeError):
    """An active publisher owns the named writer lease."""


def _paths(name, directory):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", name):
        raise ValueError("name must be 1-120 ASCII letters/digits/_.-")
    if name in (".", ".."):
        raise ValueError("name cannot be a directory component")
    if directory is None:
        directory = "/dev/shm" if os.name == "posix" else tempfile.gettempdir()
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("bus directory must already exist")
    return root / name, root / (name + ".data.lock"), root / (name + ".writer.lock")


def _validate_fd(fd, path):
    opened = os.fstat(fd)
    named = os.lstat(str(path))
    if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode):
        raise ValueError("bus/lock must be a regular file, never a symlink or device")
    if opened.st_nlink != 1 or named.st_nlink != 1:
        raise ValueError("bus/lock hard links are refused")
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError("bus/lock name changed while opening")
    if os.name == "posix":
        if opened.st_uid != os.getuid():
            raise PermissionError("bus/lock file belongs to another user")
        if opened.st_mode & 0o077:
            raise PermissionError("bus/lock file must have owner-only mode 0600")
    return opened


def _open_checked(path, create=False, readonly=False):
    flags = os.O_RDONLY if readonly else os.O_RDWR
    flags |= (getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) |
              getattr(os, "O_NONBLOCK", 0))
    created = False
    if create:
        try:
            fd = os.open(str(path), flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            fd = os.open(str(path), flags)
    else:
        fd = os.open(str(path), flags)
    try:
        os.set_inheritable(fd, False)
        _validate_fd(fd, path)
        return fd, created
    except Exception:
        os.close(fd)
        raise


if os.name == "nt":
    import msvcrt

    def _try_lock(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise

    def _unlock(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


class _FileLock:
    def __init__(self, path, kind, claim=False):
        self.path = path
        self.fd = None
        self._thread = threading.RLock()
        self._claim = False
        marker = ("NEUROGRIP_MMAP_LOCK_V1:" + kind + "\n").encode("ascii")
        fd, created = _open_checked(path, create=True)
        self.fd = fd
        try:
            if created:
                if os.write(fd, marker) != len(marker):
                    raise OSError("incomplete lock-file marker write")
                os.fsync(fd)
            if _validate_fd(fd, path).st_size != len(marker):
                raise ValueError("unrecognized pre-existing lock file; refusing overwrite")
            # Win32 range locks can prevent reading the first marker byte while
            # the existing owner holds it. Read/validate the marker suffix only;
            # the exact length plus suffix identify this lock namespace.
            os.lseek(fd, 1, os.SEEK_SET)
            if os.read(fd, len(marker) - 1) != marker[1:]:
                raise ValueError("unrecognized pre-existing lock file; refusing overwrite")
            if claim:
                if not _try_lock(fd):
                    raise WriterBusyError("another publisher owns this mmap bus")
                self._claim = True
        except Exception:
            self.close()
            raise

    @contextmanager
    def locked(self, timeout_ms=50):
        deadline = time.monotonic() + timeout_ms / 1000.0
        with self._thread:
            while not _try_lock(self.fd):
                if time.monotonic() >= deadline:
                    raise TimeoutError("mmap bus lock timeout")
                time.sleep(.001)
            try:
                _validate_fd(self.fd, self.path)
                yield
            finally:
                _unlock(self.fd)

    def close(self):
        if self.fd is not None:
            try:
                if self._claim:
                    _unlock(self.fd)
            finally:
                os.close(self.fd)
                self.fd = None


def _empty(session):
    prefix = _HEADER.pack(MAGIC, 1, HEADER_SIZE, 48, 0, session, 0, 0,
                          os.getpid()) + bytes(48)
    return prefix + _CRC.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


def _recognized(block):
    if len(block) != BLOCK_SIZE:
        return False
    magic, version, header, payload, flags = struct.unpack_from("<8sHHHH", block)
    return magic == MAGIC and version == 1 and header == HEADER_SIZE and payload == 48 and flags in (0, 1)


@dataclass(frozen=True)
class MappedSnapshot:
    packet: VisionTelemetry
    publisher_age_ms: float
    actionable: bool
    write_counter: int
    publisher_pid: int


class MappedPublisher:
    def __init__(self, name=DEFAULT_NAME, directory=None, create=True):
        if not isinstance(create, bool):
            raise ValueError("create must be boolean")
        self.path, data_path, writer_path = _paths(name, directory)
        self.name = name
        self.session_id = secrets.randbits(32)
        self._lease = self._data = self._map = self._fd = None
        self._closed = False
        self._counter = 0
        self._sequences = SequenceTracker()
        self._sequences.reset(self.session_id)
        try:
            self._lease = _FileLock(writer_path, "writer", claim=True)
            self._data = _FileLock(data_path, "data")
            with self._data.locked():
                self._fd, created = _open_checked(self.path, create=create)
                if created:
                    if os.write(self._fd, _empty(self.session_id)) != BLOCK_SIZE:
                        raise OSError("incomplete bus initialization")
                    os.fsync(self._fd)
                else:
                    if _validate_fd(self._fd, self.path).st_size != BLOCK_SIZE:
                        raise ValueError("pre-existing bus is not 96 bytes; refusing overwrite")
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    if not _recognized(os.read(self._fd, BLOCK_SIZE)):
                        raise ValueError("unrecognized pre-existing bus file; refusing overwrite")
                self._map = mmap.mmap(self._fd, BLOCK_SIZE, access=mmap.ACCESS_WRITE)
                self._map[:] = _empty(self.session_id)
                self._map.flush()
        except Exception:
            self._release_resources()
            raise

    def _release_resources(self):
        try:
            if self._map is not None:
                self._map.close()
                self._map = None
        finally:
            try:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
            finally:
                for lock in (self._data, self._lease):
                    if lock is not None:
                        lock.close()

    def publish(self, packet):
        if self._closed:
            raise RuntimeError("publisher is closed")
        if not isinstance(packet, VisionTelemetry) or packet.session_id != self.session_id:
            raise ValueError("packet session must match publisher.session_id")
        entered_ns = time.monotonic_ns()
        with self._data.locked():
            if _validate_fd(self._fd, self.path).st_size != BLOCK_SIZE:
                raise ValueError("bus file size changed; refusing a mapped write")
            age = min(65535, packet.capture_age_ms +
                      int(math.ceil((time.monotonic_ns() - entered_ns) / 1000000.0)))
            reasons, permit = packet.block_reasons, packet.strike_permit
            if age >= packet.valid_for_ms:
                permit = False
                reasons = tuple(dict.fromkeys(packet.block_reasons + ("stale_frame",)))
            wire = replace(packet, capture_age_ms=age, strike_permit=permit,
                           block_reasons=reasons)
            payload = encode_can_fd(wire)
            if not self._sequences.accept(packet):
                raise ValueError("duplicate/backward packet sequence")
            self._counter = (self._counter + 1) & 0xFFFFFFFF
            prefix = _HEADER.pack(MAGIC, 1, HEADER_SIZE, 48, 1,
                                  self.session_id, self._counter,
                                  time.monotonic_ns(), os.getpid()) + payload
            self._map[:] = prefix + _CRC.pack(zlib.crc32(prefix) & 0xFFFFFFFF)
            self._map.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            with self._data.locked():
                if _validate_fd(self._fd, self.path).st_size != BLOCK_SIZE:
                    raise ValueError("bus file size changed; refusing a mapped write")
                self._map[:] = _empty(self.session_id)
                self._map.flush()
        finally:
            self._release_resources()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MappedSubscriber:
    def __init__(self, name=DEFAULT_NAME, directory=None):
        self.path, data_path, unused_writer_path = _paths(name, directory)
        self._closed = False
        self._map = self._fd = self._data = None
        try:
            self._data = _FileLock(data_path, "data")
            with self._data.locked():
                self._fd, unused_created = _open_checked(self.path, readonly=True)
                if _validate_fd(self._fd, self.path).st_size != BLOCK_SIZE:
                    raise ValueError("bus file must be 96 bytes")
                self._map = mmap.mmap(self._fd, BLOCK_SIZE, access=mmap.ACCESS_READ)
                if not _recognized(self._map[:]):
                    raise ValueError("unrecognized bus file")
        except Exception:
            self.close()
            raise

    def read(self, max_age_ms=100):
        if self._closed:
            raise RuntimeError("subscriber is closed")
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or not 1 <= max_age_ms <= 65535:
            raise ValueError("max_age_ms must be an integer in [1,65535]")
        try:
            with self._data.locked():
                if _validate_fd(self._fd, self.path).st_size != BLOCK_SIZE:
                    return None
                block = self._map[:]
            (magic, version, header, payload, flags, session,
             counter, published_ns, pid) = _HEADER.unpack(block[:HEADER_SIZE])
            if (magic != MAGIC or version != 1 or header != HEADER_SIZE or
                    payload != 48 or flags != 1 or block[36:44] != bytes(8)):
                return None
            if _CRC.unpack(block[92:])[0] != zlib.crc32(block[:92]) & 0xFFFFFFFF:
                return None
            age_ns = time.monotonic_ns() - published_ns
            if published_ns == 0 or age_ns < 0:
                return None
            packet = decode_can_fd(block[44:92])
            age_ms = int(math.ceil(age_ns / 1000000.0))
            if packet.session_id != session or packet.capture_age_ms + age_ms >= min(max_age_ms, packet.valid_for_ms):
                return None
            return MappedSnapshot(packet, age_ns / 1000000.0,
                                  is_actionable(packet, age_ms, local_max_age_ms=max_age_ms),
                                  counter, pid)
        except (ValueError, OSError, TimeoutError, struct.error):
            return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._map is not None:
                self._map.close()
                self._map = None
        finally:
            try:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
            finally:
                if self._data is not None:
                    self._data.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Package runtime can swap the import without changing telemetry construction.
SharedMemoryPublisher = MappedPublisher
SharedMemorySubscriber = MappedSubscriber
