"""Core tests for rffusion. Run: pytest tests/ -v

These exercise the shared plumbing and each build's core logic on the
NumPy/CPU path, so the whole suite runs on x86 CI with no CUDA or SDR.
"""

import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from rffusion.analytics.engine import (
    AnalyticsEngine,
    DeviceObservation,
    WelfordAnomaly,
)
from rffusion.common.config import Config, OutputConfig, SdrConfig
from rffusion.common.iobuf import BACKEND_NAME, IQRing, alloc_iq
from rffusion.common.timebase import (
    DeviceClock,
    GpsClock,
    LinearClock,
    Stamp,
    capture_ns,
)
from rffusion.fusion.correlate import CorrelationEngine, RfEvent, Track
from rffusion.survey.sdr import SyntheticSource, time_spec_to_ns
from rffusion.survey.spectrum import SpectrumEngine

# ---------------------------------------------------------------- timebase

def test_capture_ns_monotonic():
    a = capture_ns()
    b = capture_ns()
    assert b >= a


def test_gpsclock_single_anchor_is_offset():
    gps = GpsClock()
    gps.observe(mono_ns=1_000, utc_ns=1_000_000)
    # 500 ns after the anchor in mono => 500 ns after in utc
    assert gps.to_utc_ns(1_500) == 1_000_500


def test_gpsclock_no_fix_returns_none():
    assert GpsClock().to_utc_ns(123) is None


def test_gpsclock_linear_fit_recovers_offset():
    gps = GpsClock()
    # utc = mono + 1_000_000, perfectly linear
    for m in range(0, 10_000, 1_000):
        gps.observe(m, m + 1_000_000)
    assert abs(gps.to_utc_ns(5_000) - (5_000 + 1_000_000)) < 10


def test_stamp_now_without_gps():
    s = Stamp.now()
    assert s.mono_ns > 0
    assert s.utc_ns is None


# ------------------------------------------- device time / GPSDO discipline

def test_device_clock_no_anchor_returns_none():
    assert DeviceClock().to_mono_ns(1) is None


def test_device_clock_single_anchor_is_offset():
    dc = DeviceClock()
    dc.observe(dev_ns=1_000, mono_ns=5_000_000)
    assert dc.to_mono_ns(2_000) == 5_001_000


def test_device_clock_fit_absorbs_oscillator_rate_offset():
    # Device oscillator runs 10 ppm slow against host monotonic; the fit's
    # slope term is what keeps a long capture from walking off.
    dc = DeviceClock()
    for i in range(8):
        dev = i * 1_000_000_000
        dc.observe(dev, 500 + round(dev * (1 - 10e-6)))
    want = 500 + round(8_000_000_000 * (1 - 10e-6))
    assert abs(dc.to_mono_ns(8_000_000_000) - want) < 50


def test_linear_clock_keeps_resolution_at_realistic_magnitudes():
    # Real stamps are ~1e13-1e15 ns. Fitting on absolute values in float64
    # loses exactly the sub-microsecond detail PPS sync exists to provide,
    # so the fit is done relative to the first anchor.
    lc = LinearClock()
    base_x, base_y = 1_700_000_000_000_000, 12_345_678_901_234
    for i in range(4):
        lc.observe(base_x + i * 1_000_000, base_y + i * 1_000_000)
    assert lc.map_ns(base_x + 4_000_000) == base_y + 4_000_000


def test_time_spec_to_ns_survives_long_uptime():
    class FakeTimeSpec:
        def __init__(self, full, frac):
            self._full, self._frac = full, frac

        def get_full_secs(self):
            return self._full

        def get_frac_secs(self):
            return self._frac

    # 4 days of device uptime plus 1.5 us must not lose the 1.5 us.
    assert time_spec_to_ns(FakeTimeSpec(345_600, 0.0000015)) == 345_600 * 10**9 + 1500


def test_stamp_falls_back_to_host_clock_when_unanchored():
    src = SyntheticSource(SdrConfig(pps_sync=True), IQRing(block_size=4, depth=2))
    before = capture_ns()
    got = src._stamp(None)
    assert before <= got <= capture_ns()


def test_synthetic_device_time_tracks_sample_count():
    src = SyntheticSource(SdrConfig(sample_rate=1e6, block_size=1024, pps_sync=True),
                          IQRing(block_size=1024, depth=4))
    assert src._sim_dev_ns(0) == 0
    assert src._sim_dev_ns(1024) == 1_024_000
    assert src._sim_dev_ns(2048) == 2_048_000


def test_pps_stamps_are_evenly_spaced_not_host_jittered():
    # The point of PPS sync: block spacing comes from the disciplined
    # oscillator, so it is exact rather than however the host was scheduled.
    n, fs = 512, 1e6
    ring = IQRing(block_size=n, depth=1024)
    src = SyntheticSource(SdrConfig(sample_rate=fs, block_size=n, pps_sync=True), ring)
    src.start()
    try:
        # Drain as we go and wait on a generous deadline rather than
        # assuming a fixed sleep yields a fixed block count — thread
        # spin-up and sleep granularity vary too much across machines.
        stamps: list[int] = []
        deadline = time.monotonic() + 5.0
        while len(stamps) < 8 and time.monotonic() < deadline:
            blk = ring.pop(timeout=0.5)
            if blk is None:
                continue
            if blk.dropped_before:
                stamps = []  # a gap makes spacing meaningless; restart
                continue
            stamps.append(blk.mono_ns)
    finally:
        src.stop()

    assert len(stamps) >= 8
    step = n * 1_000_000_000 // int(fs)
    assert {b - a for a, b in pairwise(stamps)} == {step}


def test_non_pps_run_still_stamps_from_host():
    src = SyntheticSource(SdrConfig(pps_sync=False), IQRing(block_size=4, depth=2))
    assert not src.dev_clock.has_fix
    before = capture_ns()
    assert before <= src._stamp(None) <= capture_ns()


# ------------------------------------------------------------------ iobuf

def test_alloc_iq_shape_dtype():
    buf = alloc_iq(1024)
    assert len(buf) == 1024
    assert buf.dtype == np.complex64


def test_ring_fifo_order():
    ring = IQRing(block_size=4, depth=4)
    for i in range(3):
        ring.push(np.full(4, i, dtype=np.complex64), mono_ns=i)
    assert ring.pop().seq == 0
    assert ring.pop().seq == 1
    assert ring.pop().seq == 2


def test_ring_drops_oldest_on_overrun():
    ring = IQRing(block_size=4, depth=2)
    for i in range(4):  # push 4 into depth-2 ring
        ring.push(np.zeros(4, dtype=np.complex64), mono_ns=i)
    # State: seq0,seq1 were dropped as seq2,seq3 arrived. The ring now holds
    # seq2 (1 drop before it: seq0) and seq3 (1 drop before it: seq1).
    blk = ring.pop()
    assert blk.seq == 2
    assert blk.dropped_before == 1
    assert ring.pop().seq == 3


def test_ring_accumulates_consecutive_drops():
    # Drops that happen with no intervening *pop* still attribute correctly:
    # fill, then push two more without reading => two blocks each carry the
    # single drop that displaced them.
    ring = IQRing(block_size=4, depth=2)
    for i in range(2):
        ring.push(np.zeros(4, dtype=np.complex64), mono_ns=i)
    ring.push(np.zeros(4, dtype=np.complex64), mono_ns=2)  # drops seq0
    ring.push(np.zeros(4, dtype=np.complex64), mono_ns=3)  # drops seq1
    a, b = ring.pop(), ring.pop()
    assert (a.seq, a.dropped_before) == (2, 1)
    assert (b.seq, b.dropped_before) == (3, 1)


def test_ring_pop_timeout_returns_none():
    ring = IQRing(block_size=4, depth=2)
    assert ring.pop(timeout=0.01) is None


def test_ring_close_unblocks():
    ring = IQRing(block_size=4, depth=2)
    ring.close()
    assert ring.pop(timeout=0.01) is None


# --------------------------------------------------------------- spectrum

def test_spectrum_detects_tone():
    fs = 1_000_000.0
    fc = 100_000_000.0
    nfft = 1024
    eng = SpectrumEngine(nfft=nfft, sample_rate=fs, center_freq=fc, threshold_db=6.0)

    # complex tone at +200 kHz from center, plus a little noise
    n = 1 << 15
    t = np.arange(n) / fs
    tone = np.exp(2j * np.pi * 200_000 * t).astype(np.complex64)
    noise = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 0.01
    iq = tone + noise

    psd = eng.psd(iq)
    dets = eng.detect(psd, mono_ns=42)

    assert dets, "expected at least one detection"
    # strongest detection should sit near fc + 200 kHz
    strongest = max(dets, key=lambda d: d.power_db)
    assert abs(strongest.freq_hz - (fc + 200_000)) < fs / nfft * 4
    assert strongest.mono_ns == 42


def test_spectrum_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        SpectrumEngine(nfft=1000, sample_rate=1e6, center_freq=1e8)


def test_spectrum_psd_length():
    eng = SpectrumEngine(nfft=512, sample_rate=1e6, center_freq=1e8)
    iq = np.zeros(1 << 14, dtype=np.complex64)
    assert len(eng.psd(iq)) == 512


# --------------------------------------------------------------- analytics

def test_welford_warmup_returns_zero():
    a = WelfordAnomaly(dim=2, warmup=10)
    # first observations during warmup score 0
    assert a.score([1.0, 1.0]) == 0.0


def test_welford_flags_outlier():
    import random
    a = WelfordAnomaly(dim=1, warmup=5)
    rng = random.Random(1)
    for _ in range(50):
        a.score([10.0 + rng.gauss(0, 1)])   # baseline ~N(10, 1), realistic
    out = a.score([1000.0], learn=False)     # wild outlier
    assert out > 5.0


def test_welford_dim_mismatch():
    a = WelfordAnomaly(dim=3)
    with pytest.raises(ValueError):
        a.score([1.0, 2.0])


def test_analytics_tracks_devices():
    eng = AnalyticsEngine()
    for i in range(10):
        eng.ingest(DeviceObservation(mac="AA:BB:CC:00:00:01",
                                     mono_ns=i * 1_000_000, rssi=-60, channel=6))
    assert eng.device_count == 1


def test_analytics_randomized_macs_grouped():
    eng = AnalyticsEngine()
    # two different random MACs, same phy+channel => one aggregate key
    eng.ingest(DeviceObservation(mac="DA:00:00:00:00:01", mono_ns=0, rssi=-50,
                                 channel=36, is_randomized_mac=True))
    eng.ingest(DeviceObservation(mac="DA:00:00:00:00:02", mono_ns=1_000_000, rssi=-50,
                                 channel=36, is_randomized_mac=True))
    assert eng.device_count == 1


# ----------------------------------------------------------------- fusion

def test_correlation_associates_cooccurring():
    eng = CorrelationEngine(window_ns=100_000_000)  # 100 ms
    ident = "AA:BB:CC:DD:EE:FF"

    # A car track and an RF ident that always appear together.
    for i in range(20):
        t_ns = i * 50_000_000
        eng.observe_tracks([Track(track_id=7, label="car", mono_ns=t_ns)])
        eng.observe_rf(RfEvent(ident=ident, mono_ns=t_ns + 1_000_000))

    assert eng.score(ident, 7) > 0.5


def test_correlation_ignores_noncooccurring():
    eng = CorrelationEngine(window_ns=100_000_000)
    # track early, RF ident much later — never within the window
    eng.observe_tracks([Track(track_id=1, label="car", mono_ns=0)])
    eng.observe_rf(RfEvent(ident="ZZ", mono_ns=10_000_000_000))
    assert eng.score("ZZ", 1) == 0.0


def test_correlation_top_associations_sorted():
    eng = CorrelationEngine(window_ns=100_000_000)
    for i in range(10):
        t = i * 10_000_000
        eng.observe_tracks([Track(track_id=1, label="car", mono_ns=t)])
        eng.observe_rf(RfEvent(ident="strong", mono_ns=t))
    eng.observe_tracks([Track(track_id=2, label="truck", mono_ns=0)])
    eng.observe_rf(RfEvent(ident="weak", mono_ns=0))

    top = eng.top_associations(min_score=0.0)
    assert top[0][2] >= top[-1][2]  # sorted descending


# ------------------------------------------------------------------ config

def test_config_defaults_valid():
    cfg = Config.load()
    assert cfg.sdr.sample_rate > 0


def test_config_rejects_non_pow2_block():
    import pathlib
    import tempfile
    p = pathlib.Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text("sdr:\n  block_size: 1000\n")
    with pytest.raises(ValueError):
        Config.load(p)


def test_shipped_example_config_loads():
    # The example is what users copy to /etc/rffusion; a key indented into
    # the wrong section fails only at load time, which no other test hits.
    cfg = Config.load(Path(__file__).resolve().parent.parent
                      / "configs" / "rffusion.example.yaml")
    assert cfg.sdr.driver == "uhd"
    assert cfg.sdr.pps_sync is False          # timing knobs live under sdr:
    assert cfg.sdr.pps_timeout_s > 0
    # Not just present — actually numeric. See below.
    assert cfg.sdr.sample_rate == 20e6
    assert cfg.sdr.center_freq == 915e6


def test_unsigned_exponent_yaml_still_yields_numbers(tmp_path):
    # PyYAML is YAML 1.1: "20.0e6" resolves to a str, only "20.0e+6" to a
    # float. That is exactly how people write sample rates, and a str
    # sample_rate does not fail at load — it fails later comparing str to
    # int, far from the cause. SdrConfig coerces instead.
    p = tmp_path / "unsigned.yaml"
    p.write_text("sdr:\n  sample_rate: 20.0e6\n  center_freq: 915.0e6\n")
    cfg = Config.load(p)
    assert isinstance(cfg.sdr.sample_rate, float)
    assert cfg.sdr.sample_rate == 20e6
    assert cfg.sdr.center_freq == 915e6


def test_non_numeric_sdr_value_names_the_field():
    with pytest.raises(ValueError, match="sdr.sample_rate"):
        SdrConfig(sample_rate="fast")


def test_config_rejects_typoed_time_source():
    # UHD ignores an unknown source, which would leave you with
    # undisciplined time that still *looks* PPS-synced. Fail loudly.
    import pathlib
    import tempfile
    p = pathlib.Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text("sdr:\n  time_source: gpsd\n")   # 'gpsdo' typoed
    with pytest.raises(ValueError):
        Config.load(p)


def test_config_accepts_gpsdo_sources():
    cfg = Config()
    cfg.sdr.clock_source = "gpsdo"
    cfg.sdr.time_source = "gpsdo"
    cfg.validate()   # must not raise


def test_log_dir_uses_configured_when_writable(tmp_path):
    out = OutputConfig(log_dir=tmp_path / "logs")
    assert out.resolve_log_dir() == tmp_path / "logs"
    assert (tmp_path / "logs").is_dir()


def test_log_dir_falls_back_when_unwritable(tmp_path, monkeypatch):
    # The /var/lib default is not creatable by an unprivileged user; the
    # documented survey smoke command must still run. Guards the CI runner.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    out = OutputConfig(log_dir=Path("/proc/nonexistent/rffusion"))
    resolved = out.resolve_log_dir()
    assert resolved == tmp_path / "state" / "rffusion"
    assert resolved.is_dir()


def test_backend_is_known():
    assert BACKEND_NAME in ("cupy", "numpy")
