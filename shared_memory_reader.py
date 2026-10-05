"""Print local shared-memory recommendations; never control an actuator."""
import argparse
import json
import time

from shared_memory_bus import DEFAULT_NAME, SharedMemorySubscriber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--max-age-ms", type=int, default=100)
    parser.add_argument("--interval-ms", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.max_age_ms <= 65535 or not 1 <= args.interval_ms <= 60000:
        parser.error("age must be 1..65535 ms; interval must be 1..60000 ms")
    reader = None
    next_reconnect = 0.
    previous_session = None
    try:
        while True:
            if reader is None:
                try:
                    reader = SharedMemorySubscriber(args.name)
                except (FileNotFoundError, TimeoutError, ValueError):
                    pass
            snapshot = None if reader is None else reader.read(args.max_age_ms)
            if snapshot is None:
                output = {"fresh": False, "actionable": False, "direction": 255,
                          "strike_permit": False, "gap_mm": None}
                # Reattach to a new POSIX mapping after unlink/restart, and
                # retry missing Windows publishers without retaining old data.
                if reader is not None and time.monotonic() >= next_reconnect:
                    reader.close()
                    reader = None
                    next_reconnect = time.monotonic() + .25
            else:
                changed = previous_session is not None and snapshot.packet.session_id != previous_session
                if previous_session is None:
                    previous_session = snapshot.packet.session_id
                output = {"fresh": True, "actionable": snapshot.actionable and not changed,
                          "direction": int(snapshot.packet.direction) if snapshot.actionable and not changed else 255,
                          "strike_permit": snapshot.actionable and not changed,
                          "gap_mm": snapshot.packet.gap_mm,
                          "publisher_age_ms": snapshot.publisher_age_ms,
                          "session_changed": changed, "telemetry": snapshot.packet.to_dict()}
            print(json.dumps(output, allow_nan=False), flush=True)
            time.sleep(args.interval_ms / 1000)
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    main()
