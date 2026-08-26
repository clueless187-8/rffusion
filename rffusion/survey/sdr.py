"""SDR sources feeding the IQRing.

The real path targets the Ettus B210 via UHD. Because UHD/CuPy aren't
present on CI or a dev laptop, a SyntheticSource generates test IQ so the
survey pipeline is runnable and demonstrable everywhere. select_source()
picks automatically.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..common.config import SdrConfig
from ..common.iobuf import IQRing, alloc_iq
from ..common.timebase import DeviceClock, capture_ns

LOG = logging.getLogger("rffusion.sdr")

NS_PER_S = 1_000_000_000


class SourceBase:
    """A source fills an IQRing from a background thread until stopped.

    Blocks are always pushed with a stamp in the ONE monotonic domain the
    repo fuses on. When the device carries a disciplined clock of its own
    (`pps_sync`), the stamp is derived from the device timestamp through
    `DeviceClock` rather than read from the host at recv completion — see
    the DeviceClock docstring for why that is more accurate and why it
    still lands in the monotonic domain.
    """

    def __init__(self, cfg: SdrConfig, ring: IQRing) -> None:
        self.cfg = cfg
        self.ring = ring
        self.dev_clock = DeviceClock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_anchor_mono = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _due_for_anchor(self) -> bool:
        """True when the device<->host anchor is stale (or missing)."""
        if not self.dev_clock.has_fix:
            return True
        interval_ns = int(self.cfg.time_anchor_interval_s * NS_PER_S)
        return capture_ns() - self._last_anchor_mono >= interval_ns

    def _stamp(self, dev_ns: int | None) -> int:
        """Monotonic stamp for a block.

        Falls back to host time whenever the device clock is unusable, so
        an un-anchored or non-PPS run still produces usable (just coarser)
        timestamps rather than failing.
        """
        if dev_ns is None:
            return capture_ns()
        mono = self.dev_clock.to_mono_ns(dev_ns)
        return mono if mono is not None else capture_ns()

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class SyntheticSource(SourceBase):
    """Generates noise + a couple of tones. For CI, dev, and demos.

    With `pps_sync` set it also simulates a disciplined device clock:
    "device time" advances exactly with the sample count, the way a
    GPSDO-locked USRP's does, and is anchored to host monotonic on the
    same cadence as the real path. That keeps the timing code in
    SourceBase on the tested CPU path instead of only in the hardware
    branch nobody can run on CI.

    Two ways this is IDEALISED, so nobody reads more into a synthetic run
    than it can support:
      * The simulated clock is perfect by construction — no drift, no
        holdover, no rate error — so it cannot exercise the slope term of
        the fit the way real hardware will.
      * A laptop cannot synthesise samples at the configured rate, so
        simulated device time advances slower than wall clock (~0.6 s of
        device time per 2 s of real time at 20 Msps). Block *spacing* is
        faithful; absolute pace is not. On the B210 the two converge
        because the radio really does deliver at the sample rate.
    Timing claims get verified on hardware, not from a synthetic run.
    """

    def _sim_dev_ns(self, sample_index: int) -> int:
        """Device time of a sample index, exact at the nominal sample rate."""
        return (sample_index * NS_PER_S) // int(self.cfg.sample_rate)

    def _run(self) -> None:
        fs = self.cfg.sample_rate
        n = self.cfg.block_size
        t0 = 0
        rng = np.random.default_rng(0)
        # two synthetic emitters at fixed offsets from center
        offsets = (0.15 * fs, -0.30 * fs)
        while not self._stop.is_set():
            t = (np.arange(n) + t0) / fs
            sig = np.zeros(n, dtype=np.complex64)
            for off in offsets:
                sig += np.exp(2j * np.pi * off * t).astype(np.complex64)
            noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
            buf = alloc_iq(n)
            buf[:] = sig * 0.5 + noise * 0.05

            if self.cfg.pps_sync:
                dev_ns = self._sim_dev_ns(t0)
                if self._due_for_anchor():
                    self.dev_clock.observe(dev_ns, capture_ns())
                    self._last_anchor_mono = capture_ns()
                mono_ns = self._stamp(dev_ns)
            else:
                mono_ns = capture_ns()

            self.ring.push(buf, mono_ns=mono_ns)
            t0 += n
            # pace to roughly real-time so the ring behaves like live capture
            time.sleep(n / fs)


def time_spec_to_ns(ts) -> int:
    """UHD TimeSpec -> integer nanoseconds.

    Kept off the float path: whole seconds stay integral and only the
    fractional part is scaled, so a large device uptime cannot eat the
    sub-microsecond resolution this whole exercise is about.
    """
    return int(ts.get_full_secs()) * NS_PER_S + round(ts.get_frac_secs() * NS_PER_S)


class UhdSource(SourceBase):
    """Ettus B210 via UHD. Only importable where UHD is installed."""

    def _discipline_time(self, usrp, uhd) -> bool:
        """Point the device at its reference and zero its clock on a PPS edge.

        Returns True if device time is now PPS-aligned. Device time is set
        to 0 at the edge rather than to GPS time-of-week: DeviceClock maps
        it onto host monotonic anyway, so only its *rate* and continuity
        matter, and 0 keeps the arithmetic small and exact.
        """
        if self.cfg.clock_source:
            usrp.set_clock_source(self.cfg.clock_source)
        if self.cfg.time_source:
            usrp.set_time_source(self.cfg.time_source)

        # Latch onto a real edge: wait for the last-PPS counter to tick
        # over, then arm the set for the NEXT edge. Setting without
        # observing an edge first can land the write across one, which
        # silently offsets device time by a whole second.
        deadline = time.monotonic() + self.cfg.pps_timeout_s
        last = usrp.get_time_last_pps().get_real_secs()
        while usrp.get_time_last_pps().get_real_secs() == last:
            if time.monotonic() > deadline:
                LOG.warning("no PPS edge within %.1fs — falling back to host "
                            "recv-completion stamps", self.cfg.pps_timeout_s)
                return False
            time.sleep(0.01)

        usrp.set_time_next_pps(uhd.types.TimeSpec(0.0))
        time.sleep(1.1)  # let that edge actually arrive before trusting it
        LOG.info("device time PPS-aligned (clock=%s time=%s)",
                 self.cfg.clock_source or "default",
                 self.cfg.time_source or "default")
        return True

    def _anchor(self, usrp) -> None:
        """Pair a device-clock read with a host-clock read.

        Sandwiching the device read between two host reads and taking the
        midpoint halves the bias from the USB round trip, which is the
        dominant error in this pairing.
        """
        before = capture_ns()
        dev_ns = time_spec_to_ns(usrp.get_time_now())
        after = capture_ns()
        self.dev_clock.observe(dev_ns, (before + after) // 2)
        self._last_anchor_mono = after

    def _run(self) -> None:  # pragma: no cover - hardware path
        import uhd  # type: ignore

        usrp = uhd.usrp.MultiUSRP(self.cfg.args or "type=b200")
        usrp.set_rx_rate(self.cfg.sample_rate)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.cfg.center_freq))
        usrp.set_rx_gain(self.cfg.gain)

        pps_ok = self._discipline_time(usrp, uhd) if self.cfg.pps_sync else False
        if pps_ok:
            self._anchor(usrp)

        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx = usrp.get_rx_stream(st_args)
        md = uhd.types.RXMetadata()

        n = self.cfg.block_size
        rx.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))
        try:
            while not self._stop.is_set():
                buf = alloc_iq(n)
                got = rx.recv(buf, md, 1.0)
                if md.error_code != uhd.types.RXMetadataErrorCode.none:
                    LOG.warning("UHD rx error: %s", md.error_code)
                    continue

                if pps_ok:
                    # md.time_spec is the device time of the FIRST sample in
                    # this buffer — the actual ADC instant, not when the
                    # buffer finished crossing USB.
                    if self._due_for_anchor():
                        self._anchor(usrp)
                    mono_ns = self._stamp(
                        time_spec_to_ns(md.time_spec) if md.has_time_spec else None)
                else:
                    mono_ns = capture_ns()

                self.ring.push(buf[:got], mono_ns=mono_ns)
        finally:
            rx.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))


def select_source(cfg: SdrConfig, ring: IQRing) -> SourceBase:
    """Pick the real source if its driver imports, else synthetic."""
    if cfg.driver == "uhd":
        try:
            import uhd  # noqa: F401
            return UhdSource(cfg, ring)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("UHD unavailable (%s) — using synthetic source", exc)
    return SyntheticSource(cfg, ring)
