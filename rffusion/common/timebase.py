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
    * A DeviceClock maps SDR device time -> monotonic, so a GPSDO-
      disciplined USRP's sample-accurate timestamps can be used without
      introducing a third clock domain to fuse on.

Both are the same linear fit over rolling anchors (LinearClock); only the
pair of domains differs.

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
    x_ns: int
    y_ns: int


@dataclass
class LinearClock:
    """Maps one nanosecond time domain onto another by linear fit.

    Two different pairs of domains need exactly this: monotonic -> UTC
    (GpsClock, anchored by gpsd fixes) and SDR device time -> monotonic
    (DeviceClock, anchored by reading the USRP clock alongside the host
    clock). Same rolling-anchor least-squares in both cases, so it lives
    here once.

    The fit is computed relative to the first anchor rather than absolute
    epoch values. Absolute ns are large enough (~1e13 and up) that the
    normal-equation form loses precision in float64; differencing first
    keeps sub-ns resolution, which is the whole point when the caller is
    chasing sample-accurate time.
    """

    _anchors: list[_Anchor] = field(default_factory=list)
    _max_anchors: int = 16
    _lock: Lock = field(default_factory=Lock)

    def observe(self, x_ns: int, y_ns: int) -> None:
        """Record a synchronized observation of the two domains."""
        with self._lock:
            self._anchors.append(_Anchor(int(x_ns), int(y_ns)))
            if len(self._anchors) > self._max_anchors:
                self._anchors.pop(0)

    def map_ns(self, x_ns: int) -> int | None:
        """Map a stamp from the x domain to the y domain, None if no anchors.

        With one anchor: pure offset, unit rate. With several: slope and
        intercept, so the fit also absorbs the small rate difference
        between the two oscillators.
        """
        with self._lock:
            anchors = list(self._anchors)

        if not anchors:
            return None
        x0, y0 = anchors[0].x_ns, anchors[0].y_ns
        if len(anchors) == 1:
            return y0 + (x_ns - x0)

        # Fit dy = slope * dx + intercept on offsets from the first anchor.
        n = len(anchors)
        dxs = [a.x_ns - x0 for a in anchors]
        dys = [a.y_ns - y0 for a in anchors]
        sx = sum(dxs)
        sy = sum(dys)
        sxx = sum(dx * dx for dx in dxs)
        sxy = sum(dx * dy for dx, dy in zip(dxs, dys))
        denom = n * sxx - sx * sx
        if denom == 0:
            # Degenerate: every anchor at the same x. Use the newest offset.
            a = anchors[-1]
            return a.y_ns + (x_ns - a.x_ns)
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return y0 + round(slope * (x_ns - x0) + intercept)

    @property
    def has_fix(self) -> bool:
        with self._lock:
            return bool(self._anchors)


@dataclass
class GpsClock:
    """Maps monotonic capture time to UTC via gpsd time observations.

    Feed it (mono_ns, utc_ns) pairs as GPS fixes arrive; query it to turn
    any monotonic capture stamp into wall-clock UTC for logging. Uses a
    small rolling anchor set and linear fit so a single jittery fix does
    not skew the mapping.
    """

    _map: LinearClock = field(default_factory=LinearClock)

    def observe(self, mono_ns: int, utc_ns: int) -> None:
        """Record a synchronized (monotonic, UTC) observation."""
        self._map.observe(mono_ns, utc_ns)

    def to_utc_ns(self, mono_ns: int) -> int | None:
        """Convert a monotonic capture stamp to UTC ns, or None if no fix yet."""
        return self._map.map_ns(mono_ns)

    @property
    def has_fix(self) -> bool:
        return self._map.has_fix


@dataclass
class DeviceClock:
    """Maps SDR device time to host monotonic time.

    A GPSDO-disciplined USRP timestamps each received buffer in its OWN
    clock domain (UHD's `RXMetadata.time_spec`). That domain is far more
    stable than the host's view of when a buffer finished crossing USB,
    but it is not the domain the rest of this repo fuses on. So we keep
    the repo rule intact — everything still lands on `capture_ns()` — by
    anchoring the two domains against each other and converting.

    What this buys: the *spacing* between blocks becomes sample-accurate,
    driven by the disciplined oscillator instead of host scheduling
    jitter. The absolute offset still carries whatever error was in the
    anchor reads, but that error is a roughly constant bias rather than
    per-block noise, and more anchors average it down.
    """

    _map: LinearClock = field(default_factory=LinearClock)

    def observe(self, dev_ns: int, mono_ns: int) -> None:
        """Record a paired read of the device clock and the host clock."""
        self._map.observe(dev_ns, mono_ns)

    def to_mono_ns(self, dev_ns: int) -> int | None:
        """Convert a device timestamp to host monotonic ns, None if unanchored."""
        return self._map.map_ns(dev_ns)

    @property
    def has_fix(self) -> bool:
        return self._map.has_fix


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
