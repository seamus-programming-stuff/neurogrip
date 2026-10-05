"""Validated PC-side settings for the two-camera feed and electrical bus."""
from dataclasses import dataclass, asdict
import ipaddress
import json
import math
from pathlib import Path
from urllib.parse import urlparse


def stream_address(value):
    value = str(value).strip()
    if not value:
        raise ValueError("A camera address is required")
    if "://" not in value:
        host = ipaddress.ip_address(value)
        if host.version != 4:
            raise ValueError("Use the NEON's IPv4 address")
        return "http://%s:8081/raw" % host
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use an HTTP camera stream without embedded credentials")
    return value


def controller_address(value):
    if value is None or value == "":
        return None
    try:
        host, port = str(value).rsplit(":", 1)
        address = ipaddress.ip_address(host)
        port = int(port)
        if address.version != 4 or address.is_multicast or address.is_unspecified or not 1 <= port <= 65535:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("Controller must be a unicast IPv4 address and port, e.g. 192.168.20.50:55050")
    return str(address), port


@dataclass
class DualConfig:
    schema_version: int = 1
    calibration: str = "stereo.json"
    controller: str | None = None
    controller_source_port: int = 55051
    shared_memory_name: str = "neurogrip_vision_v1"
    stroke_config: str = "config.example.json"
    recommendations_enabled: bool = False
    stereo_timing_validated: bool = False
    exposure_skew_bound_ms: float | None = None
    max_pair_skew_ms: float = 20.0
    stale_after_ms: int = 500
    heartbeat_ms: int = 50
    num_disparities: int = 128
    block_size: int = 5
    min_depth_mm: float = 100.0
    max_depth_mm: float = 2000.0
    display_host: str = "127.0.0.1"
    display_port: int = 8090

    def checked(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported dual-camera configuration version")
        for name in ("recommendations_enabled", "stereo_timing_validated"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be boolean")
        for name in ("stale_after_ms", "heartbeat_ms", "num_disparities", "block_size", "display_port", "controller_source_port"):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + " must be an integer")
        if not 1 <= self.stale_after_ms <= 65535 or not 10 <= self.heartbeat_ms <= 1000:
            raise ValueError("Invalid observation lifetime or heartbeat period")
        if self.num_disparities < 16 or self.num_disparities % 16 or self.num_disparities > 1024:
            raise ValueError("num_disparities must be a multiple of 16 in [16,1024]")
        if self.block_size < 3 or self.block_size > 21 or self.block_size % 2 != 1:
            raise ValueError("block_size must be odd in [3,21]")
        if not 1 <= self.display_port <= 65535 or not 0 <= self.controller_source_port <= 65535:
            raise ValueError("Invalid port")
        for name in ("max_pair_skew_ms", "min_depth_mm", "max_depth_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        if self.max_depth_mm <= self.min_depth_mm:
            raise ValueError("max_depth_mm must exceed min_depth_mm")
        if self.exposure_skew_bound_ms is not None:
            value = self.exposure_skew_bound_ms
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("exposure_skew_bound_ms must be nonnegative and finite")
        if self.stereo_timing_validated and self.exposure_skew_bound_ms is None:
            raise ValueError("Validated stereo timing requires a measured exposure_skew_bound_ms")
        for name in ("calibration", "stroke_config", "shared_memory_name", "display_host"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(name + " must be a nonempty string")
        if not self.shared_memory_name.isascii() or not self.shared_memory_name.replace("_", "").isalnum() or len(self.shared_memory_name) > 100:
            raise ValueError("Use a short alphanumeric shared memory name with underscores")
        controller_address(self.controller)
        display_address = ipaddress.ip_address(self.display_host)
        if display_address.version != 4 or display_address.is_loopback is False:
            raise ValueError("The video display binds only to a PC loopback address")
        return self

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError("Invalid or unknown dual-camera settings")
        result = cls(**values).checked()
        for field in ("calibration", "stroke_config"):
            candidate = Path(getattr(result, field))
            if not candidate.is_absolute():
                setattr(result, field, str(path.parent / candidate))
        return result

    def save(self, path):
        self.checked()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
