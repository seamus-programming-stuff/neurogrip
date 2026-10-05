# Neurogrip vision interface v1

The PC estimates the finger-to-object gap and recommends a calibrated finger direction. It publishes **vision telemetry**, not a motor command. The electrical controller owns movement authorization, feedback, current limits, travel limits, watchdogs, and the separately armed strike action. A true `strike_permit` heartbeat must never initiate or repeat a strike by itself.

This is a bench prototype interface. A single camera observes visible surfaces and estimates depth; it cannot guarantee a collision-free stroke, reveal hidden obstacles, or establish reliable distance on every plain, shiny, or transparent surface. Hardware authorization requires measured model error and a conservatively validated stroke envelope.

## Files to give the electrical engineer

- `bus_schema.json`: strict JSON telemetry schema.
- `protocol.py`: standard-library JSON encoder/decoder and optional CAN-FD payload encoder/decoder.
- `include/neurogrip_bus.h`: C99-compatible payload decoder, receiver freshness check, sequence helper, and constant definitions. It has no hardware I/O.

The physical shared bus has not been selected. The JSON contract works independently of the transport. An optional 48-byte CAN-FD payload is specified below; it cannot be sent as one classic-CAN frame. No bus sender, arbitration identifier, bitrate, pinout, voltage, or transceiver is assumed. Choose those together with the electrical engineer after the actual controller is known. If using classic CAN, define a separate bounded reassembly protocol before use; never accept a partial telemetry message.

## Direction and strike semantics

| Direction | Value | Electrical interpretation |
|---|---:|---|
| BACKWARDS | 0 | The backward physical pose/stroke calibrated on this hand |
| UPRIGHT | 1 | The upright physical pose/stroke calibrated on this hand |
| FORWARDS | 2 | The forward physical pose/stroke calibrated on this hand |
| UNKNOWN | 255 | No accepted direction; block new movement |

The values do not specify motor polarity, servo angle, PWM duty cycle, or which image axis is forward. Agree the physical meanings on the hand, then measure reachable strokes in the same calibrated coordinates as the camera. The vision application must default to UNKNOWN until that mapping is configured. Direction is only a recommendation; changing it on a heartbeat must not immediately move the finger.

`strike_permit=true` means the vision application's conditions are currently satisfied. It is not an instruction to strike. The MCU combines a fresh accepted packet with its own **separate one-shot strike request**, local armed state, actual finger position, limit/current checks, and validated reachable stroke. Once accepted, consume that one-shot request. A repeating true permit cannot retrigger it. This interface intentionally contains no strike-trigger field.

| Controller state | Required behavior |
|---|---|
| DISARMED | Boot state and state after a session change; telemetry cannot move the finger |
| READY | Explicitly armed locally; accept telemetry only after a fresh-session handshake; wait for an independent one-shot request |
| EXECUTING | Enforce current, position, travel, and timing limits locally throughout motion; fresh telemetry does not bypass those limits |
| FAULT / STALE | Block new movement, invalidate pending requests, and execute the stop/hold/retract policy established for the actual hand |

Whether holding or retracting is safe during a fault depends on the mechanics; determine that locally rather than assuming one universal pose. A lost vision stream must not leave the previous permission latched.

## JSON fields

All fields are required. Unknown distance is JSON `null`, never zero; zero means estimated contact. Numeric values must be finite. Decoder rejects missing/extra fields, duplicate JSON keys, non-finite numbers, unsupported enums, and inconsistent permits.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer = 1 | Contract version; reject unsupported versions |
| `session_id` | uint32 | Random source boot/session identifier; changing it requires an explicit receiver handshake and local rearm |
| `sequence` | uint32 | Packet counter, wraps modulo 2^32 |
| `timestamp_ms` | uint64 | Source monotonic milliseconds at serialization; diagnostic, not a shared clock |
| `capture_age_ms` | uint16 | Time from frame capture through inference/queueing to serialization |
| `valid_for_ms` | uint16, > 0 | Maximum **total frame age**, not a new lifetime beginning at receipt |
| `mode` | `SIMULATION` / `LIVE` | Synthetic and replay data must use SIMULATION |
| `direction` | 0, 1, 2, 255 | Physical direction recommendation defined above |
| `strike_permit` | boolean | Time-limited vision condition, no actuation command |
| `gap_mm` | number / null | Estimated visible finger-to-target surface gap in millimetres; valid range 0–1,000,000 |
| `quality` | number, 0–1 | Heuristic score; not a probability, error bound, or proof of safe contact |
| `scale_valid` | boolean | Application's metric-calibration checks accepted the scale; local authorization still needs validated model error and reach |
| `block_reasons` | unique string array | All active reasons that prevent a vision permit |

An estimated metric model output alone does not establish `scale_valid` or permission. Relative depth has no millimetre scale. Validate the accepted calibration and measurement error against known distances or an RGB-D reference. If the camera moves, the physical stroke mapping must be revalidated before permission is restored.

Example blocked packet, suitable for a synthetic run:

```json
{"schema_version":1,"session_id":42,"sequence":7,"timestamp_ms":12000,"capture_age_ms":20,"valid_for_ms":100,"mode":"SIMULATION","direction":255,"strike_permit":false,"gap_mm":null,"quality":0.0,"scale_valid":false,"block_reasons":["simulation","unknown_depth","direction_unknown"]}
```

When `strike_permit` is true, the decoder additionally requires LIVE mode, known direction, non-null metric gap, `scale_valid=true`, positive quality, no block reasons, and `capture_age_ms < valid_for_ms`. These conditions make a packet internally consistent; they do not replace the MCU's independent checks. A `/present` proximity output or an object-detection result must not be wired directly to a strike input.

## Freshness, restarts, and watchdogs

The receiver's acceptance order is:

1. Validate length, schema, CRC where used, finite numbers, and permit consistency.
2. Require an explicitly paired, current session. A new session ID is a disarm event, not an automatic reset-and-accept event.
3. Reject duplicate/out-of-order sequences. For the same session, calculate `delta = (new - previous) mod 2^32`; accept only `1 <= delta < 2^31`. The first packet may be accepted only after the fresh-session handshake. Re-pair before a gap of 2^31 packets makes ordering ambiguous.
4. Store receiver monotonic receipt time for the newest accepted packet. A rejected/repeated packet must not refresh that time or the watchdog.
5. Check `capture_age_ms + elapsed_since_receipt_ms + verified_transport_delay_budget_ms < min(valid_for_ms, local_max_age_ms)` every time a request is evaluated, not just when a packet arrives.
6. Combine that fresh vision condition with local armed state, current measured position, calibrated stroke/error margin, and an independently armed one-shot strike request.

The Python freshness helper uses a 100 ms local age cap by default, a starting value to validate against observed inference time and permitted finger speed. It accepts an explicit transport-delay budget. A zero budget is convenient for bench inspection and does not establish a hardware end-to-end age guarantee.

Receiving a frame on the PC is not necessarily its sensor capture time. Camera encoding, MJPEG transport, and buffers can add age before the PC timestamp. If reliable capture timestamps are absent, measure and conservatively bound that upstream delay as well. If age cannot be bounded, block hardware permission. Independently enforce a local link watchdog; inference stalls, camera disconnects, replayed packets, and parser errors must expire permission.

Source and MCU monotonic timestamps cannot be subtracted without established clock alignment. Receive-time expiry cannot identify a packet that was already delayed in transport before receipt. Use a verified bounded local bus plus a fresh-session handshake, or design and validate synchronized clocks/challenge freshness before relying on the age budget. Session IDs, sequence numbers, and CRCs are not cryptographic authentication.

## Optional CAN-FD payload

Payload length is exactly **48 bytes**. All integer fields and the IEEE-754 binary32 gap are **little endian**. The C structure is a decoded representation and must not be transmitted by casting its address.

| Offset | Bytes | Encoding |
|---:|---:|---|
| 0 | 4 | ASCII `NGV1` |
| 4 | 1 | Schema version = 1 |
| 5 | 1 | Mode: 0 SIMULATION, 1 LIVE |
| 6 | 1 | Direction enum |
| 7 | 1 | Flags: bit 0 vision permit; bit 1 scale valid; others zero |
| 8 | 4 | uint32 session ID |
| 12 | 4 | uint32 sequence |
| 16 | 8 | uint64 source monotonic timestamp, milliseconds |
| 24 | 2 | uint16 capture age, milliseconds |
| 26 | 2 | uint16 maximum total frame age, milliseconds |
| 28 | 4 | float32 gap, millimetres; **-1.0 means unknown** |
| 32 | 1 | Quality: `floor(quality * 255)`; divide by 255 when decoding |
| 33 | 3 | Reserved, must be zero |
| 36 | 4 | uint32 block reason mask |
| 40 | 4 | Reserved, must be zero |
| 44 | 4 | CRC-32/ISO-HDLC of bytes 0–43, same as Python `zlib.crc32` |

Float32 changes gap precision slightly, and quality is quantized conservatively downward. If a positive permitted quality would quantize to zero, the binary encoder rejects that permitted packet. NaN and infinity are rejected; -1.0 is the only accepted negative gap and always blocks permission.

| Reason bit | JSON string |
|---:|---|
| 0 | `stale_frame` |
| 1 | `unknown_depth` |
| 2 | `uncalibrated` |
| 3 | `low_quality` |
| 4 | `target_missing` |
| 5 | `finger_missing` |
| 6 | `unreachable` |
| 7 | `simulation` |
| 8 | `direction_unknown` |
| 9 | `controller_unarmed` |
| 10 | `invalid_geometry` |
| 11 | `occluded` |
| 12 | `calibration_unvalidated` |
| 13 | `no_stroke_calibration` |
| 14 | `camera_moved` |
| 15 | `depth_out_of_range` |

Unknown/reserved flags, reason bits, or bytes are rejected. Allocate a CAN identifier and bus priority after the controller and other nodes are known; evaluate worst-case arbitration delay and set the receiver transport budget accordingly. The payload functions do not open or write a bus.

## Minimum bench validation before connecting movement

Test the receiver with actuator power isolated first: missing/NaN distances, UNKNOWN direction, SIMULATION mode, duplicate packets, backward sequences, sequence rollover, source reboot, delayed packets, camera loss, and inference stalls must block a new strike. Verify fresh heartbeats cannot create or repeat a strike without the separate local one-shot request. Then validate the actual stroke envelope, camera-to-output age, distance error across the working range, current/travel limits, and fault behavior on the hand.

Protocol test command on the Windows PC:

```bat
python -m unittest discover -s tests -p test_protocol.py -v
```
