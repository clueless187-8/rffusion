"""Core tests for rffusion. Run: pytest tests/ -v

These exercise the shared plumbing and each build's core logic on the
NumPy/CPU path, so the whole suite runs on x86 CI with no CUDA or SDR.
"""

import numpy as np
import pytest

from rffusion.analytics.engine import (
    AnalyticsEngine,
    DeviceObservation,
    WelfordAnomaly,
)
from rffusion.common.config import Config
from rffusion.common.iobuf import BACKEND_NAME, IQRing, alloc_iq
from rffusion.common.timebase import GpsClock, Stamp, capture_ns
from rffusion.fusion.correlate import CorrelationEngine, RfEvent, Track
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


def test_backend_is_known():
    assert BACKEND_NAME in ("cupy", "numpy")
