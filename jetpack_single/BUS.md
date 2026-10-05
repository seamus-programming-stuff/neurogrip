# NEON local memory and Ethernet telemetry

This package targets **JetPack 5.1.2 / Python 3.8**. It uses one NEON camera. Telemetry recommends a finger direction and reports an estimated gap; it never drives a motor. BACKWARDS=0, UPRIGHT=1, FORWARDS=2, UNKNOWN=255 must be mapped to the hand's measured physical strokes. A true vision permit cannot initiate or repeat a strike without the electrical controller's separate local arm and one-shot request.

## Local Linux bus

`mapped_bus.py` maps a regular **96-byte** file at `/dev/shm/neurogrip_vision_v1`. It does not use `multiprocessing.shared_memory`, so an independent Python 3.8 reader's resource tracker cannot unlink the publisher's memory.

The publisher generates a new `session_id`, resets the ready flag on startup, and holds an exclusive `flock` lease on `/dev/shm/neurogrip_vision_v1.writer.lock` for its lifetime. A second publisher raises `WriterBusyError`. Both processes hold `flock` on `/dev/shm/neurogrip_vision_v1.data.lock` during each complete copy; lock acquisition times out after 50 ms. A CRC alone is not treated as atomic synchronization.

All three files use owner-only mode **0600** and must belong to the current user. Symlinks, hard links, other file types, and unrecognized pre-existing contents are refused. Run readers as the same account as the publisher. No unrelated file is overwritten. Clean shutdown clears permission but deliberately keeps the files and their lock inodes. A reader exit never removes the bus. After a publisher crash, the last frame expires against its source age and the reader's monotonic age cap; its kernel lease releases automatically. Restart reuses the mapping with a new session. The controller must disarm and explicitly re-pair on a session change.

Python reader, running from this package directory:

```python
from mapped_bus import MappedSubscriber

with MappedSubscriber() as reader:
    snapshot = reader.read(max_age_ms=100)
    if snapshot is None:
        direction, permit, gap_mm = 255, False, None
    else:
        direction = int(snapshot.packet.direction) if snapshot.actionable else 255
        permit = snapshot.actionable  # Vision condition; never a strike trigger.
        gap_mm = snapshot.packet.gap_mm
```

Publisher API: `MappedPublisher(name="neurogrip_vision_v1", directory=None, create=True)`, `.session_id`, `.publish(packet)`, `.close()`. Use the publisher's session in every `VisionTelemetry`. Subscriber API: `MappedSubscriber(name=..., directory=None)`, `.read(max_age_ms=100)`, `.close()`. Both support `with`. Unknown gap remains `None`, never zero. SIMULATION packets cannot provide a hardware permit. Duplicate/backward packet sequences are refused.

`snapshot.actionable` applies the existing protocol check to source capture age plus elapsed publisher monotonic age. Stale, invalid, future-timestamped, CRC-corrupted, or incomplete snapshots return `None`. Repeated reads cannot refresh the deadline. Metric depth is an estimate; a validated error margin, timing bound, calibrated stroke, actual finger feedback, current/travel limits, and the local watchdog are still required.

## Binary layout for a native Linux reader

All numbers are little endian. Open the data lock file without replacing it, take `flock(fd, LOCK_SH)` or `LOCK_EX`, copy all 96 mapped bytes, release the lock, then validate the envelope and the embedded payload. The publisher uses `LOCK_EX`. Use Linux `CLOCK_MONOTONIC` in nanoseconds for publisher age; do not use wall time or the MCU's clock. `include/neurogrip_shm.h` defines the same byte offsets and portable decoding; its Windows-specific mapping functions are not used on Linux.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | `NGSHM1` followed by two zero bytes |
| 8 | 2 | Envelope version = 1 |
| 10 | 2 | Header size = 44 |
| 12 | 2 | Payload size = 48 |
| 14 | 2 | Ready flag = 1; 0 means empty |
| 16 | 4 | uint32 publisher session, matching payload session |
| 20 | 4 | uint32 write counter, modulo 2^32 |
| 24 | 8 | uint64 publisher monotonic nanoseconds |
| 32 | 4 | uint32 publisher PID |
| 36 | 8 | Reserved zero bytes |
| 44 | 48 | Existing `NGV1` telemetry payload, including its CRC |
| 92 | 4 | CRC-32/ISO-HDLC of bytes 0–91 |

Reject unrecognized versions, reserved bytes, invalid flags, CRC errors, or mismatched sessions. Round elapsed time upward and require:

```text
capture_age_ms + ceil((now_monotonic_ns - published_ns) / 1_000_000)
    < min(packet.valid_for_ms, local_max_age_ms)
```

The C decoder is a helper, not actuator firmware. It checks a vision condition and must be combined with independent controller authorization.

## Ethernet controller

Shared memory is local to the NEON. It does not cross the gigabit switch. `network_bus.UDPPublisher(host, port, source_port=0)` sends the embedded **48-byte NGV1 payload** as one UDP datagram directly to a configured Ethernet controller. It sends neither the 96-byte local envelope nor a CAN frame. `source_port` can be fixed so the controller pins the expected NEON source IP and port.

Use the packaged `ethernet_bus_receiver.py` as a diagnostic reference. Its default binding is localhost; explicitly bind the controller interface for Ethernet. Pair the current source session, pin the expected source endpoint, and supply a measured worst-case transport delay bound before its `actionable` result can become true. Replace the placeholders below; no real controller address is assumed:

```bash
python3 ethernet_bus_receiver.py --bind CONTROLLER_IP --port 55050 --peer NEON_IP:SOURCE_PORT --session CURRENT_SESSION --transport-delay-bound-ms VERIFIED_BOUND --max-age-ms 100
```

Receiver freshness uses its own monotonic receipt time plus `capture_age_ms` and the verified network-delay bound. It must not subtract the NEON timestamp from a different machine's clock. Changed sessions revoke pairing; duplicates and backward counters cannot refresh the watchdog. CRC and peer filtering detect corruption/unexpected endpoints, not cryptographic authenticity. This receiver prints diagnostics and performs no motor I/O.

## Verification limits

The mapped-bus tests exercise spawned concurrent processes, writer exclusion, reader exit, clean/crash restart, killing a writer during a partial copy, freshness expiry, CRC checks, and refusal to overwrite unrelated files. They pass on the Windows mmap/byte-lock test fallback. Python 3.8 grammar is checked. **Linux `flock`, ARM execution, and NEON hardware behavior still require verification on the device.** No inference speed, physical accuracy, or safe whole-finger stroke is established by these protocol tests.
