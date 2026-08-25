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
from ..common.timebase import capture_ns

LOG = logging.getLogger("rffusion.sdr")


class SourceBase:
    """A source fills an IQRing from a background thread until stopped."""

    def __init__(self, cfg: SdrConfig, ring: IQRing) -> None:
        self.cfg = cfg
        self.ring = ring
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class SyntheticSource(SourceBase):
    """Generates noise + a couple of tones. For CI, dev, and demos."""

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
            self.ring.push(buf, mono_ns=capture_ns())
            t0 += n
            # pace to roughly real-time so the ring behaves like live capture
            time.sleep(n / fs)


class UhdSource(SourceBase):
    """Ettus B210 via UHD. Only importable where UHD is installed."""

    def _run(self) -> None:  # pragma: no cover - hardware path
        import uhd  # type: ignore

        usrp = uhd.usrp.MultiUSRP(self.cfg.args or "type=b200")
        usrp.set_rx_rate(self.cfg.sample_rate)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.cfg.center_freq))
        usrp.set_rx_gain(self.cfg.gain)

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
                # Stamp at recv completion. For sample-accurate time you'd
                # use md.time_spec against a GPSDO PPS; see docs/build1.md.
                self.ring.push(buf[:got], mono_ns=capture_ns())
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
