"""UDP transport for the existing 48-byte v1 vision payload; no motor I/O.

CRC detects corruption, not attackers. Receivers pin a peer and explicitly pair
a source boot session. End-to-end freshness needs a verified transport bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
import socket
import time

from protocol import (VisionTelemetry, SequenceTracker, encode_can_fd,
                      decode_can_fd, is_actionable)


def _port(port, *, zero=False):
    if isinstance(port, bool) or not isinstance(port, int) or not (0 if zero else 1) <= port <= 65535:
        raise ValueError("port must be an integer in the supported UDP range")
    return port


def _ipv4(host):
    if not isinstance(host, str):
        raise ValueError("host must be an IPv4 address or hostname")
    address = socket.gethostbyname(host)
    ipaddress.IPv4Address(address)
    return address


class UDPPublisher:
    def __init__(self, host, port, *, source_port=0):
        self.endpoint = (_ipv4(host), _port(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._closed = False
        try:
            if source_port:
                self._socket.bind(("0.0.0.0", _port(source_port)))
            else:
                _port(source_port, zero=True)
            # A connected UDP socket selects a route/source address and filters
            # its peer; UDP connect performs no remote handshake or motor I/O.
            self._socket.connect(self.endpoint)
            self.local_address = self._socket.getsockname()
        except Exception:
            self._socket.close()
            raise

    def publish(self, packet: VisionTelemetry):
        if self._closed:
            raise RuntimeError("UDP publisher is closed")
        payload = encode_can_fd(packet)
        sent = self._socket.send(payload)
        if sent != len(payload):
            raise OSError("incomplete UDP telemetry datagram")

    def close(self):
        if not self._closed:
            self._closed = True
            self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class NetworkSnapshot:
    packet: VisionTelemetry
    receiver_age_ms: float
    actionable: bool
    paired: bool
    peer: tuple[str, int]


class UDPReceiver:
    def __init__(self, bind_host="127.0.0.1", port=55050, *, expected_peer=None,
                 transport_delay_bound_ms=None, max_age_ms=100, paired_session_id=None):
        if isinstance(max_age_ms, bool) or not isinstance(max_age_ms, int) or not 1 <= max_age_ms <= 65535:
            raise ValueError("max_age_ms must be an integer in [1,65535]")
        if transport_delay_bound_ms is not None:
            if isinstance(transport_delay_bound_ms, bool) or not isinstance(transport_delay_bound_ms, int) or not 0 <= transport_delay_bound_ms <= 65535:
                raise ValueError("transport delay bound must be integer milliseconds")
        self.max_age_ms = max_age_ms
        self.transport_delay_bound_ms = transport_delay_bound_ms
        self.expected_peer = None
        if expected_peer is not None:
            if not isinstance(expected_peer, (tuple, list)) or len(expected_peer) != 2:
                raise ValueError("expected_peer must be (IPv4 address, port)")
            self.expected_peer = (_ipv4(expected_peer[0]), _port(expected_peer[1]))
        self._tracker = SequenceTracker()
        self._latest = None
        self._received_ns = None
        self._peer = None
        self._closed = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind((_ipv4(bind_host), _port(port, zero=True)))
            self.local_address = self._socket.getsockname()
            if paired_session_id is not None:
                self.pair_session(paired_session_id)
        except Exception:
            self._socket.close()
            raise

    @property
    def paired_session_id(self):
        return self._tracker.session_id

    def pair_session(self, session_id):
        """Explicit operator/controller fresh-session pairing; clears old data.

        Not a network handshake: establish the current source session through an
        operator or your controller's challenge/response before calling this.
        """
        self._tracker.reset(session_id)
        self._latest = self._received_ns = self._peer = None

    def _snapshot(self):
        if self._latest is None:
            return None
        elapsed_ns = time.monotonic_ns() - self._received_ns
        elapsed_ms = math.ceil(elapsed_ns / 1_000_000)
        bound = self.transport_delay_bound_ms
        total_age = self._latest.capture_age_ms + elapsed_ms + (bound or 0)
        if elapsed_ns < 0 or total_age >= min(self.max_age_ms, self._latest.valid_for_ms):
            return None
        paired = self._tracker.session_id == self._latest.session_id
        actionable = bool(
            paired and self.expected_peer is not None and bound is not None and
            is_actionable(self._latest, elapsed_ms,
                          transport_delay_budget_ms=bound, local_max_age_ms=self.max_age_ms)
        )
        return NetworkSnapshot(self._latest, elapsed_ns / 1_000_000,
                               actionable, paired, self._peer)

    def receive(self, timeout=0.05):
        """Receive one new valid packet. Invalid/duplicate/unexpected returns None.

        No rejected packet updates receipt time. A changed paired session revokes
        pairing and returns diagnostic telemetry with actionable=False.
        """
        if self._closed:
            raise RuntimeError("UDP receiver is closed")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        self._socket.settimeout(timeout)
        try:
            payload, peer = self._socket.recvfrom(65535)
        except (socket.timeout, BlockingIOError):
            return None
        if self.expected_peer is not None and peer != self.expected_peer:
            return None
        try:
            packet = decode_can_fd(payload)
        except ValueError:
            return None
        received_ns = time.monotonic_ns()
        if self._tracker.session_id is not None:
            if packet.session_id != self._tracker.session_id:
                # Explicit new-session event immediately revokes old permission.
                self._tracker = SequenceTracker()
                self._latest = self._received_ns = self._peer = None
            elif not self._tracker.accept(packet):
                return None
        # Unpaired packets may be inspected but can never be actionable.
        self._latest, self._received_ns, self._peer = packet, received_ns, peer
        return self._snapshot()

    def latest(self):
        """Re-evaluate cached freshness. It never refreshes the watchdog."""
        if self._closed:
            raise RuntimeError("UDP receiver is closed")
        return self._snapshot()

    def close(self):
        if not self._closed:
            self._closed = True
            self._latest = self._received_ns = self._peer = None
            self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
