"""Unified time base for RF/vision fusion.

Everything in this monorepo — SDR samples, WiFi/BLE detections, camera
frames, GPS fixes — must be stampable against ONE monotonic clock, or
fusion (Build 3) is impossible and the survey/analytics logs (Builds 1/2)
can't be correlated after the fact. This module is that clock.

Design:
    * capture_ns() is the single source of truth. It reads
      CLOCK_MONOTONIC_RAW so it never jumps on NTP/GPS steps — critical,
      because a GPS lock mid-drive would otherwise warp every timestamp.
    * A GpsClock maps monotonic time -> UTC using observed (monotonic,
      utc) pairs from gpsd, so you get absolute time for logging WITHOUT
      contaminating the monotonic base you fuse on.

The split matters: fuse on monotonic, *label* with UTC. Never fuse on
UTC — it's not monotonic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock


def capture_ns() -> int:
    """Monotonic capture timestamp in nanoseconds.

    CLOCK_MONOTONIC_RAW: not subject to NTP slew or GPS step. This is the
    number every subsystem stamps its data with at the moment of capture.
    """
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


@dataclass
class _Anchor:
    mono_ns: int
    utc_ns: int


@dataclass
class GpsClock:
    """Maps monotonic capture time to UTC via gpsd time observations.

    Feed it (mono_ns, utc_ns) pairs as GPS fixes arrive; query it to turn
    any monotonic capture stamp into wall-clock UTC for logging. Uses a
    small rolling anchor set and linear fit so a single jittery fix does
    not skew the mapping.
    """

    _anchors: list[_Anchor] = field(default_factory=list)
    _max_anchors: int = 16
    _lock: Lock = field(default_factory=Lock)

    def observe(self, mono_ns: int, utc_ns: int) -> None:
        """Record a synchronized (monotonic, UTC) observation."""
        with self._lock:
            self._anchors.append(_Anchor(mono_ns, utc_ns))
            if len(self._anchors) > self._max_anchors:
                self._anchors.pop(0)

    def to_utc_ns(self, mono_ns: int) -> int | None:
        """Convert a monotonic capture stamp to UTC ns, or None if no fix yet.

        With one anchor: pure offset. With several: least-squares slope +
        offset, which also absorbs the small rate difference between the
        Pi/Jetson oscillator and GPS time.
        """
        with self._lock:
            anchors = list(self._anchors)

        if not anchors:
            return None
        if len(anchors) == 1:
            a = anchors[0]
            return a.utc_ns + (mono_ns - a.mono_ns)

        # Least-squares fit utc = slope * mono + intercept.
        n = len(anchors)
        sx = sum(a.mono_ns for a in anchors)
        sy = sum(a.utc_ns for a in anchors)
        sxx = sum(a.mono_ns * a.mono_ns for a in anchors)
        sxy = sum(a.mono_ns * a.utc_ns for a in anchors)
        denom = n * sxx - sx * sx
        if denom == 0:
            a = anchors[-1]
            return a.utc_ns + (mono_ns - a.mono_ns)
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return int(slope * mono_ns + intercept)

    @property
    def has_fix(self) -> bool:
        with self._lock:
            return bool(self._anchors)


@dataclass(frozen=True)
class Stamp:
    """A capture instant. mono_ns always present; utc_ns/lat/lon best-effort."""

    mono_ns: int
    utc_ns: int | None = None
    lat: float | None = None
    lon: float | None = None

    @classmethod
    def now(cls, gps: GpsClock | None = None,
            lat: float | None = None, lon: float | None = None) -> Stamp:
        m = capture_ns()
        u = gps.to_utc_ns(m) if gps is not None else None
        return cls(mono_ns=m, utc_ns=u, lat=lat, lon=lon)
