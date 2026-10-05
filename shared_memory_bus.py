"""Coherent named local-PC telemetry, protected by actual OS interprocess locks.

Windows uses a named kernel mutex for each copy, plus a separate named object
claim for single-writer lifetime. POSIX uses flock and an in-process thread lock.
Neither the shared buffer nor a Python seqlock is treated as an atomic snapshot.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import inspect
import math
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
import secrets
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
    """Another publisher already owns this block; never silently steal it."""


class _AbandonedMutex(RuntimeError):
    pass


def _name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", name):
        raise ValueError("shared-memory name must be 1-120 ASCII letters/digits/_.-")
    return name


def mutex_names(name):
    """Names also used by native Windows readers in include/neurogrip_shm.h."""
    digest = hashlib.sha256(_name(name).encode("ascii")).hexdigest()
    return ("Local\\NeurogripData_" + digest, "Local\\NeurogripWriter_" + digest)


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel.CreateMutexW.restype = wintypes.HANDLE
    _kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel.WaitForSingleObject.restype = wintypes.DWORD
    _kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    _kernel.ReleaseMutex.restype = wintypes.BOOL
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL

    class _OSLock:
        def __init__(self, name, *, claim=False):
            ctypes.set_last_error(0)
            self.handle = _kernel.CreateMutexW(None, False, name)
            error = ctypes.get_last_error()
            if not self.handle:
                raise ctypes.WinError(error)
            if claim and error == 183:  # ERROR_ALREADY_EXISTS: atomic object claim.
                self.close()
                raise WriterBusyError("another publisher owns this shared-memory name")

        @contextmanager
        def locked(self, *, allow_abandoned=False, timeout_ms=50):
            result = _kernel.WaitForSingleObject(self.handle, timeout_ms)
            if result == 0x102:
                raise TimeoutError("shared-memory mutex timeout")
            if result not in (0, 0x80):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if result == 0x80 and not allow_abandoned:
                    raise _AbandonedMutex("writer died during a protected copy")
                yield
            finally:
                if not _kernel.ReleaseMutex(self.handle):
                    raise ctypes.WinError(ctypes.get_last_error())

        def close(self):
            if getattr(self, "handle", None):
                _kernel.CloseHandle(self.handle)
                self.handle = None
else:
    import fcntl

    class _OSLock:
        def __init__(self, name, *, claim=False):
            lock_root = Path(tempfile.gettempdir()) / ("neurogrip-locks-" + str(os.getuid()))
            lock_root.mkdir(mode=0o700, exist_ok=True)
            path = lock_root / (hashlib.sha256(name.encode("utf-8")).hexdigest() + ".lock")
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            self.fd = os.open(path, flags, 0o600)
            self.thread_lock = threading.RLock()
            if claim:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.close()
                    raise WriterBusyError("another publisher owns this shared-memory name")

        @contextmanager
        def locked(self, *, allow_abandoned=False, timeout_ms=50):
            del allow_abandoned
            with self.thread_lock:
                deadline = time.monotonic() + timeout_ms / 1000
                while True:
                    try:
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("shared-memory file-lock timeout")
                        time.sleep(0.001)
                try:
                    yield
                finally:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)

        def close(self):
            if getattr(self, "fd", None) is not None:
                os.close(self.fd)
                self.fd = None


def _open_memory(name, *, create=False):
    kwargs = dict(name=name, create=create, size=BLOCK_SIZE if create else 0)
    # Independent POSIX readers on >=3.13 must not unlink the owner's mapping
    # when their own resource tracker exits. Windows ignores this parameter.
    if "track" in inspect.signature(shared_memory.SharedMemory).parameters:
        kwargs["track"] = bool(create)
    return shared_memory.SharedMemory(**kwargs)


def _empty_block(session):
    header = _HEADER.pack(MAGIC, 1, HEADER_SIZE, 48, 0, session, 0, 0, os.getpid())
    prefix = header + bytes(48)
    return prefix + _CRC.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


@dataclass(frozen=True)
class SharedMemorySnapshot:
    packet: VisionTelemetry
    publisher_age_ms: float
    actionable: bool
    write_counter: int
    publisher_pid: int


class SharedMemoryPublisher:
    def __init__(self, name=DEFAULT_NAME, create=True):
        self.name = _name(name)
        if not isinstance(create, bool):
            raise ValueError("create must be boolean")
        self.session_id = secrets.randbits(32)
        self._memory = self._data_lock = self._claim = None
        self._created = False
        self._closed = False
        self._counter = 0
        self._sequences = SequenceTracker()
        self._sequences.reset(self.session_id)
        data_name, writer_name = mutex_names(self.name)
        try:
            self._claim = _OSLock(writer_name, claim=True)
            self._data_lock = _OSLock(data_name)
            with self._data_lock.locked(allow_abandoned=True):
                if create:
                    try:
                        self._memory = _open_memory(self.name, create=True)
                        self._created = True
                    except FileExistsError:
                        self._memory = _open_memory(self.name)
                else:
                    self._memory = _open_memory(self.name)
                if self._memory.size < BLOCK_SIZE:
                    raise ValueError("existing shared-memory block is too small")
                # New session invalidates the old permission before first publish.
                self._memory.buf[:BLOCK_SIZE] = _empty_block(self.session_id)
        except Exception:
            if self._memory is not None:
                self._memory.close()
                if self._created and os.name != "nt":
                    self._memory.unlink()
                self._memory = None
            for lock in (self._data_lock, self._claim):
                if lock is not None:
                    lock.close()
            raise

    def publish(self, packet):
        if self._closed:
            raise RuntimeError("publisher is closed")
        if not isinstance(packet, VisionTelemetry) or packet.session_id != self.session_id:
            raise ValueError("packet session_id must equal publisher.session_id")
        entered_ns = time.monotonic_ns()
        with self._data_lock.locked():
            delay_ms = math.ceil((time.monotonic_ns() - entered_ns) / 1_000_000)
            age = min(65535, packet.capture_age_ms + delay_ms)
            reasons = packet.block_reasons
            permit = packet.strike_permit
            if age >= packet.valid_for_ms:
                permit = False
                reasons = tuple(dict.fromkeys((*reasons, "stale_frame")))
            wire_packet = replace(packet, capture_age_ms=age, strike_permit=permit, block_reasons=reasons)
            payload = encode_can_fd(wire_packet)
            if not self._sequences.accept(packet):
                raise ValueError("publisher refuses duplicate or out-of-order packet sequence")
            self._counter = (self._counter + 1) & 0xFFFFFFFF
            header = _HEADER.pack(MAGIC, 1, HEADER_SIZE, 48, 1, self.session_id,
                                  self._counter, time.monotonic_ns(), os.getpid())
            prefix = header + payload
            self._memory.buf[:BLOCK_SIZE] = prefix + _CRC.pack(zlib.crc32(prefix) & 0xFFFFFFFF)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            with self._data_lock.locked(allow_abandoned=True):
                self._memory.buf[:BLOCK_SIZE] = _empty_block(self.session_id)
        finally:
            try:
                self._memory.close()
                # Only the creator unlinks POSIX memory. Windows removes a
                # mapping only after all publisher/subscriber handles close.
                if self._created and os.name != "nt":
                    self._memory.unlink()
            finally:
                self._data_lock.close()
                self._claim.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SharedMemorySubscriber:
    def __init__(self, name=DEFAULT_NAME):
        self.name = _name(name)
        self._closed = False
        self._lock = _OSLock(mutex_names(self.name)[0])
        try:
            with self._lock.locked(allow_abandoned=True):
                self._memory = _open_memory(self.name)
                if self._memory.size < BLOCK_SIZE:
                    self._memory.close()
                    raise ValueError("shared-memory block is too small")
        except Exception:
            self._lock.close()
            raise

    def read(self, max_age_ms=100):
        if self._closed:
            raise RuntimeError("subscriber is closed")
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or not 1 <= max_age_ms <= 65535:
            raise ValueError("max_age_ms must be an integer in [1,65535]")
        try:
            with self._lock.locked():
                block = bytes(self._memory.buf[:BLOCK_SIZE])
            (magic, version, header_size, payload_size, flags, session,
             counter, published_ns, pid) = _HEADER.unpack(block[:HEADER_SIZE])
            if (magic != MAGIC or version != 1 or header_size != HEADER_SIZE or
                    payload_size != 48 or flags != 1 or block[36:44] != bytes(8)):
                return None
            if _CRC.unpack(block[92:96])[0] != zlib.crc32(block[:92]) & 0xFFFFFFFF:
                return None
            elapsed_ns = time.monotonic_ns() - published_ns
            if published_ns == 0 or elapsed_ns < 0:
                return None
            packet = decode_can_fd(block[44:92])
            if packet.session_id != session:
                return None
            elapsed_ms = math.ceil(elapsed_ns / 1_000_000)
            if elapsed_ms >= max_age_ms or packet.capture_age_ms + elapsed_ms >= min(packet.valid_for_ms, max_age_ms):
                return None
            return SharedMemorySnapshot(packet, elapsed_ns / 1_000_000,
                                        is_actionable(packet, elapsed_ms, local_max_age_ms=max_age_ms),
                                        counter, pid)
        except (ValueError, struct.error, TimeoutError, _AbandonedMutex):
            return None

    def close(self):
        if not self._closed:
            self._closed = True
            self._memory.close()
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
