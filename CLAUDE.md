# CLAUDE.md

Standing context for Claude Code. Read this first every session.

## What this is

A monorepo for three related RF builds targeting the **NVIDIA Jetson Orin
Nano (Super)** with an **Ettus B210** SDR:

- **survey** (Build 1) — wideband GPU spectrum survey → GPS-stamped detections
- **analytics** (Build 2) — streaming analytics on a live Kismet device feed
- **fusion** (Build 3) — temporal correlation of RF idents to camera tracks

They share one time base, one sample-plumbing layer, and one config system
under `rffusion/common/`.

## Commands

```bash
pip install -e ".[dev,analytics]"     # CPU/dev install (no CUDA needed)
pytest -q                             # 27 tests, must stay green
ruff check rffusion tests             # must stay clean
python -m rffusion.survey --seconds 5 -v     # survey smoke (synthetic SDR)
python -m rffusion.analytics --events 500 -v # analytics smoke (synthetic feed)
python -m rffusion.fusion --demo -v          # fusion correlation demo
```

On the Jetson, add extras: `pip install -e ".[jetson,sdr,analytics]"`
(CuPy for cuFFT, UHD bindings for the B210).

## Non-negotiable design rules

These are load-bearing. Don't violate them without a very good reason, and
if you do, update this file.

1. **Fuse on monotonic time, label with UTC.** Everything stamps against
   `common/timebase.capture_ns()` (`CLOCK_MONOTONIC_RAW`). Wall-clock/UTC
   is derived separately via `GpsClock` and used only for logging. A GPS
   lock mid-drive must never warp the fusion clock. Never sort, window, or
   correlate on UTC.

2. **Every hardware dependency has a CPU/synthetic fallback.** The whole
   repo runs and tests on x86 with no CUDA, no SDR, no Kismet, no camera:
   - `common/iobuf` → CuPy if present, else NumPy (`BACKEND_NAME`)
   - `survey/sdr` → UHD B210 if importable, else `SyntheticSource`
   - `analytics/feed` → Kismet websocket if reachable, else `synthetic_feed`
   - `fusion` → `--demo` runs the core on a synthetic co-occurring stream
   Keep it that way. New hardware paths get a fallback in the same commit.

3. **A build must justify the GPU.** If a feature doesn't saturate CUDA or
   need sub-frame fusion, a Pi would have done it — question whether it
   belongs on this board. Analytics clustering/anomaly is CPU on purpose.

4. **Getting samples onto the GPU is the hard part, not the math.**
   `common/iobuf` uses pinned/unified memory on the Jetson and a
   drop-oldest ring. A slow consumer causes *gaps* (counted in
   `Block.dropped_before`), never blocking (which would USB-overflow) and
   never corruption. Preserve this contract.

## Scope boundaries (in code, not just docs)

- **analytics** fingerprints DEVICES and characterises the ambient
  environment. NOT for re-identifying or tracking specific people. The
  randomized-MAC grouping in `AnalyticsEngine._key_for` is deliberate —
  don't "fix" it into per-person persistent tracking.
- **fusion** associates RF idents to camera-detected OBJECTS (vehicles,
  etc.). NOT to identifiable people. `Track.label` is an object class,
  never a person identity. Don't add face recognition or person re-id.

These are design constraints, not TODOs. Keep them.

## Layout

```
rffusion/
  common/    timebase, iobuf (ring + backend), config      ← shared, careful here
  survey/    spectrum (FFT/detect), sdr (sources), __main__
  analytics/ engine (features/anomaly), feed (kismet), __main__
  fusion/    correlate (co-occurrence), __main__
tests/       test_core.py — CPU-path tests for all of the above
docs/        build1-survey.md, build2-analytics.md, build3-fusion.md
configs/     rffusion.example.yaml
```

## Gotchas learned the hard way (tests guard these — don't regress them)

- **Ring drop-count** attributes per surviving block, reset after each
  stamp. `test_ring_accumulates_consecutive_drops` guards it.
- **Welford anomaly** scores only over features with observed variance; a
  zero-variance feature must not dilute the score to zero.
  `test_welford_flags_outlier` guards it.
- **Correlation co-occurrence** counts at most one hit per RF event per
  distinct track_id — not per buffered track sample, which inflates the
  count past the event count and drives the Jaccard denominator negative.
  `test_correlation_associates_cooccurring` guards it.
- **Survey log dir** defaults to `/var/lib/rffusion`, which only root can
  create. `OutputConfig.resolve_log_dir` falls back to
  `$XDG_STATE_HOME/rffusion` so the documented smoke command works
  unprivileged (this is what CI runs). `--out DIR` overrides and is used
  as-is, with no fallback, so an explicit path fails loudly.
  `test_log_dir_falls_back_when_unwritable` guards it.

## Where to work next

- Build 1: GPSDO/PPS sample-accurate timing (`survey/sdr.UhdSource` has a
  note where `time_spec` replaces `capture_ns`). Prerequisite for real
  fusion timing.
- Build 3: DeepStream camera source feeding `Track` into
  `CorrelationEngine`; the correlation core is done, the camera wiring is
  the scaffold.

## Definition of done for any change

`pytest -q` green, `ruff check` clean, and if you touched a hardware path,
its synthetic fallback still runs the matching `python -m rffusion.<build>`
smoke command.
