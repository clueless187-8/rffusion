# 0001 — Fuse on monotonic time, label with UTC

- Status: Accepted
- Date: 2026-11-09
- Source: `CLAUDE.md` "Non-negotiable design rules" #1; `README.md` "Three rules the whole repo obeys" #1; `rffusion/common/timebase.py`

## Context

The three builds (survey, analytics, fusion) stamp events from different
sources — SDR blocks, Kismet observations, camera frames — and fusion
correlates them in time. Wall-clock time jumps when GPS acquires lock
mid-drive. A GPSDO-disciplined SDR has its own clock (`time_spec`), a
third domain.

## Decision

From CLAUDE.md, verbatim:

> **Fuse on monotonic time, label with UTC.** Everything stamps against
> `common/timebase.capture_ns()` (`CLOCK_MONOTONIC_RAW`). Wall-clock/UTC
> is derived separately via `GpsClock` and used only for logging. A GPS
> lock mid-drive must never warp the fusion clock. Never sort, window, or
> correlate on UTC. Other clock domains (a GPSDO-disciplined SDR's own
> time) are mapped INTO the monotonic domain via `DeviceClock` before
> anything downstream sees them — never fused on directly.

## Consequences

- UHD `time_spec` never reaches the ring raw; it goes through
  `timebase.DeviceClock`. Guarded by `test_device_clock_*`.
- The device↔host time fit is computed relative to the first anchor, not
  on absolute ns (float64 loses sub-microsecond detail at ~1e13). Guarded
  by `test_linear_clock_keeps_resolution_at_realistic_magnitudes`.
- Any new source (DeepStream camera, future SDRs) must stamp via
  `capture_ns()` or map through `DeviceClock` in the same commit.
