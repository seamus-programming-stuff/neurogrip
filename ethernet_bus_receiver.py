"""UDP reference receiver: JSON diagnostics only; no actuator outputs."""
import argparse
import json

from network_bus import UDPReceiver


def endpoint(value):
    try:
        host, port = value.rsplit(":", 1)
        port = int(port)
        if not host or not 1 <= port <= 65535:
            raise ValueError
        return host, port
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("use IPv4-address:port")


def session(value):
    try:
        value = int(value, 0)
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("session must be uint32 (decimal or 0x hex)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1", help="explicitly use 0.0.0.0 to receive from Ethernet")
    parser.add_argument("--port", type=int, default=55050)
    parser.add_argument("--peer", type=endpoint, help="expected PC source IP:UDP-port")
    parser.add_argument("--session", type=session, help="explicitly pair the current publisher boot/session")
    parser.add_argument("--transport-delay-bound-ms", type=int, default=None,
                        help="verified worst-case network delay; unspecified always blocks actionable")
    parser.add_argument("--max-age-ms", type=int, default=100)
    args = parser.parse_args()
    try:
        receiver = UDPReceiver(args.bind, args.port, expected_peer=args.peer,
                               paired_session_id=args.session,
                               transport_delay_bound_ms=args.transport_delay_bound_ms,
                               max_age_ms=args.max_age_ms)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        with receiver:
            print(json.dumps({"listening": receiver.local_address,
                              "expected_peer": receiver.expected_peer,
                              "paired_session_id": receiver.paired_session_id}), flush=True)
            while True:
                receiver.receive(timeout=.05)
                snapshot = receiver.latest()
                if snapshot is None:
                    output = {"fresh": False, "actionable": False, "direction": 255,
                              "strike_permit": False, "gap_mm": None}
                else:
                    output = {"fresh": True, "paired": snapshot.paired,
                              "actionable": snapshot.actionable,
                              "direction": int(snapshot.packet.direction) if snapshot.actionable else 255,
                              "strike_permit": snapshot.actionable,
                              "gap_mm": snapshot.packet.gap_mm,
                              "receiver_age_ms": snapshot.receiver_age_ms,
                              "telemetry": snapshot.packet.to_dict()}
                print(json.dumps(output, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
