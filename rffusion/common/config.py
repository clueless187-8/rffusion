"""Config loading shared by all three builds.

One schema, one loader, environment-variable overrides for the handful of
things that change between machines (device paths, gpsd host). Keeps the
three build entrypoints from each inventing their own config format.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("missing dependency: pip install pyyaml") from exc


@dataclass
class SdrConfig:
    driver: str = "uhd"                 # uhd (B210), rtlsdr, airspy, soapy
    args: str = ""                     # e.g. "type=b200" for UHD
    sample_rate: float = 20e6          # Hz. B210 goes to 56e6; start modest.
    center_freq: float = 915e6         # Hz
    gain: float = 40.0                 # dB
    block_size: int = 1 << 18          # samples per capture block
    ring_depth: int = 16               # blocks buffered before drop-oldest


@dataclass
class GpsConfig:
    host: str = "localhost"
    port: int = 2947
    enabled: bool = True


@dataclass
class CameraConfig:
    enabled: bool = False
    sensor_id: int = 0                 # CSI sensor index
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class KismetConfig:
    enabled: bool = False
    host: str = "localhost"
    port: int = 2501
    api_token: str = ""                # from ~/.kismet/kismet_httpd.conf


@dataclass
class OutputConfig:
    log_dir: Path = Path("/var/lib/rffusion")
    log_max_seconds: int = 1800

    def user_state_dir(self) -> Path:
        """Per-user fallback log directory (XDG state dir)."""
        state = os.environ.get("XDG_STATE_HOME")
        base = Path(state) if state else Path.home() / ".local" / "state"
        return base / "rffusion"

    def resolve_log_dir(self) -> Path:
        """Return a writable, existing log directory.

        The /var/lib/rffusion default is right for the Jetson, where the
        survey runs as a root service. It is not creatable by an ordinary
        user, so on a dev laptop or a CI runner fall back to a per-user
        state dir instead of dying: the documented smoke command has to
        work unprivileged. An explicitly requested directory is not
        subject to this fallback — see survey.__main__.
        """
        fallback = self.user_state_dir()
        candidates = [self.log_dir] + ([fallback] if fallback != self.log_dir else [])
        last_exc: OSError | None = None
        for cand in candidates:
            try:
                cand.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                last_exc = exc
                continue
            if os.access(cand, os.W_OK):
                return cand
            last_exc = PermissionError(f"not writable: {cand}")
        raise last_exc  # type: ignore[misc]


@dataclass
class Config:
    sdr: SdrConfig = field(default_factory=SdrConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    kismet: KismetConfig = field(default_factory=KismetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        data: dict[str, Any] = {}
        if path is not None:
            data = yaml.safe_load(Path(path).read_text()) or {}

        cfg = cls(
            sdr=SdrConfig(**data.get("sdr", {})),
            gps=GpsConfig(**data.get("gps", {})),
            camera=CameraConfig(**data.get("camera", {})),
            kismet=KismetConfig(**data.get("kismet", {})),
            output=OutputConfig(**{
                **data.get("output", {}),
                # Path needs coercion from the YAML string.
                **({"log_dir": Path(data["output"]["log_dir"])}
                   if data.get("output", {}).get("log_dir") else {}),
            }),
        )
        cfg._apply_env()
        cfg.validate()
        return cfg

    def _apply_env(self) -> None:
        """Per-machine overrides without editing the YAML."""
        if v := os.environ.get("RFFUSION_SDR_ARGS"):
            self.sdr.args = v
        if v := os.environ.get("RFFUSION_CENTER_FREQ"):
            self.sdr.center_freq = float(v)
        if v := os.environ.get("RFFUSION_GPSD_HOST"):
            self.gps.host = v
        if v := os.environ.get("RFFUSION_KISMET_TOKEN"):
            self.kismet.api_token = v
        if v := os.environ.get("RFFUSION_LOG_DIR"):
            self.output.log_dir = Path(v)

    def validate(self) -> None:
        if self.sdr.sample_rate <= 0:
            raise ValueError("sdr.sample_rate must be positive")
        if self.sdr.block_size & (self.sdr.block_size - 1) != 0:
            raise ValueError("sdr.block_size should be a power of two (FFT-friendly)")
        if not (1e6 <= self.sdr.center_freq <= 8e9):
            raise ValueError("sdr.center_freq out of plausible range")
        if self.sdr.ring_depth < 2:
            raise ValueError("sdr.ring_depth must be >= 2")
