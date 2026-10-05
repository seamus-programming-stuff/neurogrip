# Local shared memory and Ethernet controller telemetry

The PC publishes the same vision recommendation through two interfaces: a coherent named shared-memory block for **processes on that PC**, and optional UDP telemetry for an **electrical controller reached over Ethernet**. Shared memory does not cross a gigabit switch. The Ethernet controller receives one 48-byte v1 payload per UDP datagram. Neither interface performs motor I/O or transmits a CAN frame.

`direction` uses BACKWARDS=0, UPRIGHT=1, FORWARDS=2, UNKNOWN=255. These are calibrated physical directions, not motor polarity or image axes. A fresh `strike_permit` is a vision condition; the controller still owns separately armed one-shot strikes, actual position/current/travel checks, and its watchdog. A heartbeat cannot start or repeat a strike. See [electrical-interface.md](electrical-interface.md) for the complete payload and controller responsibilities.

## Ethernet UDP interface

`network_bus.UDPPublisher(host, port, source_port=0)` sends `protocol.encode_can_fd(packet)` as an exact **48-byte UDP datagram**. The function name describes its binary layout; this UDP transport never opens a CAN device. All fields are little endian, including the CRC-32 at offset 44. The existing `include/neurogrip_bus.h` decoder works on the received payload. No JSON or shared-memory envelope is added to the datagram.

The calibration command's `--controller IP:PORT` option saves the destination in `dual_config.json`; the runtime sends there when it loads that config. The sender's actual source address is exposed as `UDPPublisher.local_address`. `controller_source_port` in that config defaults to55051, which pins a stable source port. Choose both ports with the electrical engineer. No remote endpoint is used by the test suite.

The reference `ethernet_bus_receiver.py` prints JSON recommendations; it never controls an actuator. It defaults to loopback. To listen on Ethernet, explicitly name the controller's interface address or use `--bind 0.0.0.0`.

Example on the controller-side PC, using placeholder addresses and a **current** publisher session reported by the runtime:

```bat
python ethernet_bus_receiver.py --bind CONTROLLER_IP --port 55050 --peer PC_IP:55051 --session CURRENT_SESSION --transport-delay-bound-ms VERIFIED_BOUND --max-age-ms 100
```

Replace `VERIFIED_BOUND` with a measured and validated worst-case network delay in integer milliseconds; it is not an invented default. Without a bound, explicit current-session pairing, and a pinned peer, the receiver provides diagnostics with `actionable=false`. The expected peer includes **source port**, not the destination port 55050. The publisher may choose an ephemeral port unless a source port is configured.

Local inspection without permission:

```bat
python ethernet_bus_receiver.py --bind 127.0.0.1 --port 55050
```

`UDPReceiver` validates exact datagram length, protocol/version/flags, CRC, finite values, and permit consistency. It rejects an unexpected source endpoint, duplicate or backward sequences, and expired frames. Accepted sequence order uses modulo 2^32 deltas in 1..2^31-1. Rejecting a packet does not refresh the receipt timestamp. `latest()` recalculates age each time; repeatedly reading it cannot extend a permit.

The receiver uses its local monotonic receipt clock and checks:

```text
capture_age_ms + elapsed_since_receipt_ms + verified_transport_delay_bound_ms
    < min(packet.valid_for_ms, receiver.max_age_ms)
```

Sender `timestamp_ms` is diagnostic; clocks on the PC and controller are not inherently synchronized. A receipt timer cannot discover an already delayed packet before receipt. Therefore the transport bound and a fresh-session handshake/manual bench pairing are required before relying on this receiver's timing. The PC must include bounded sensor/stream/inference age in `capture_age_ms` as well.

On a changed session the receiver immediately revokes pairing, invalidates the previous permission, and continues showing unpaired diagnostics. It will not silently accept the new session on a later heartbeat. The reference CLI requires an operator to restart with the current `--session`; an embedded controller should implement its own explicit fresh-session pairing. CRC, source endpoint, and counters are not cryptographic authentication.

An MCU Ethernet implementation should receive UDP bytes, verify the expected peer, call `ng_decode_vision(bytes, 48, &packet)`, verify its explicitly paired session and forward counter, then store its local receipt time. When evaluating a separate locally armed one-shot request, call `ng_vision_is_actionable()` with elapsed local time, verified network-delay bound, and the local age cap. The C helper returns a vision condition only; it must be combined with local actuator checks. Duplicate or invalid packets must not update the watchdog. No whole-finger collision-free motion is guaranteed by visible-tip monocular/stereo estimates.

## Named shared-memory API

The standard-library API needs no additional Python package:

```python
from shared_memory_bus import SharedMemoryPublisher, SharedMemorySubscriber

publisher = SharedMemoryPublisher(name="neurogrip_vision_v1", create=True)
session_id = publisher.session_id  # Use this in every VisionTelemetry packet.
publisher.publish(packet)         # packet.session_id must match session_id.

subscriber = SharedMemorySubscriber(name="neurogrip_vision_v1")
snapshot = subscriber.read(max_age_ms=100)
if snapshot is not None:
    # snapshot.packet, publisher_age_ms, write_counter, publisher_pid, actionable
    pass  # No actuator action belongs here.

subscriber.close()
publisher.close()
```

`publish()` rejects a mismatched session, duplicate sequence, or backward sequence. It accounts for its own lock/serialization wait in source age. `read()` returns `None` for an uninitialized, invalid, torn, future-timestamped, or expired snapshot. A fresh blocked/SIMULATION packet can be returned for diagnostics but has `actionable=false`.

`SharedMemoryPublisher` generates a new random uint32 session on each start and clears any old ready flag before the first publish. A second live publisher with the same name raises `WriterBusyError`; no process steals an active writer. Clean shutdown clears the ready flag. After a crash, the last packet expires against the local age cap even if a reader keeps the mapping alive. Any controller using snapshots must compare sessions and explicitly disarm/re-pair on change.

Windows uses a real named kernel data mutex while copying the full block. Its separate writer-claim mutex object's creation/existence provides exclusive publisher lifetime without binding publication to the constructor thread. A reader must **never hold the writer-claim object**; it would keep the claim alive after a crash. Reader and publisher hold only the data mutex during copies, with a 50 ms timeout; an abandoned copy is rejected by the Python reader. These use [Microsoft's named mutex semantics](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createmutexw) and [wait/abandonment results](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject).

POSIX uses matching hashed-name `flock` files under a per-user temporary directory, plus a thread lock for each handle. Lock files remain to avoid replacing a lock inode during concurrent use. Only a publisher that created a POSIX mapping unlinks it on clean shutdown. Windows deletes a mapping only after all handles close, consistent with [Python's shared-memory lifetime documentation](https://docs.python.org/3/library/multiprocessing.shared_memory.html).

Windows Python 3.10+ is the primary runtime. Independent POSIX processes should use Python 3.13+ for `track=False` on attach; Python 3.10–3.12 resource tracking can unlink a mapping when an independent reader exits. Spawned children of a common Python multiprocessing parent are supported for the tests. On POSIX, after creator unlink/restart, an existing reader must close/reopen the name; the reader CLI retries this automatically. POSIX behavior has not been exercised on this Windows host.

Reader CLI, in a separate Windows cmd window:

```bat
python shared_memory_reader.py --name neurogrip_vision_v1 --max-age-ms 100
```

The CLI prints blocked UNKNOWN 255/null output when no valid fresh snapshot exists. It keeps a changed-session condition blocked until restarted; it does not silently update the paired session. Its `actionable` field is still only the vision condition, not a motor command.

## 96-byte local envelope

All integers are little endian. The entire block must be copied under the named data mutex before validation. A Python seqlock or CRC-only double read is not used as an atomicity guarantee.

| Offset | Bytes | Field |
|---:|---:|---|
|0 |8 |ASCII `NGSHM1`, followed by two zero bytes |
|8 |2 |Envelope version 1 |
|10 |2 |Header size 44 |
|12 |2 |Payload size 48 |
|14 |2 |Flags: 1 ready, 0 empty; other values invalid |
|16 |4 |uint32 publisher session, must match payload session |
|20 |4 |uint32 write counter, modulo 2^32 |
|24 |8 |uint64 publisher `time.monotonic_ns()` at copy |
|32 |4 |uint32 publisher process ID |
|36 |8 |Reserved, must be zero |
|44 |48 |Existing v1 telemetry payload, including its inner CRC |
|92 |4 |CRC-32/ISO-HDLC over bytes 0–91 |

The publisher monotonic clock is shared by processes on the same PC. Windows native readers use QueryPerformanceCounter/QueryPerformanceFrequency, matching [Python's Windows monotonic clock](https://docs.python.org/3/library/time.html#time.monotonic). Reject future/zero timestamps and round elapsed age upward when checking freshness.

For the default mapping:

```text
Mapping: neurogrip_vision_v1
Data mutex: Local\NeurogripData_6a19e1474facf1da4beb667e38374ac0384c3700034618a0ef1bf25f70952caa
```

For a custom mapping name, the mutex suffix is `SHA256(name.encode("ascii")).hexdigest()`; `shared_memory_bus.mutex_names(name)[0]` gives the data mutex name. `include/neurogrip_shm.h` provides constants, explicit byte decoding, and native Windows open/copy/close helpers. It uses `OpenFileMappingW`, `MapViewOfFile`, `WaitForSingleObject`, and `ReleaseMutex`. A native reader must not send a C struct as wire bytes or open the writer claim.

## Verification

```bat
python -m unittest discover -s tests -p test_shared_memory_bus.py -v
python -m unittest discover -s tests -p test_network_bus.py -v
```

Windows tests exercise actual spawned concurrent writer/reader processes, exclusive ownership, normal/crash restart, killing a writer inside a protected partial copy, CRC/length/header rejection, source and publisher age expiry, and SIMULATION blocking. UDP tests use only 127.0.0.1 and dynamically chosen ports; they check peer filtering, corruption, counter ordering/rollover, watchdog expiry, unknown delay bounds, and explicit session re-pairing. Native C helpers remain uncompiled on this host because no C compiler is available.
