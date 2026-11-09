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
pytest -q                             # 41 tests, must stay green
ruff check rffusion tests             # must stay clean
python -m rffusion.survey --seconds 5 -v     # survey smoke (synthetic SDR)
python -m rffusion.analytics --events 500 -v # analytics smoke (synthetic feed)
python -m rffusion.fusion --demo -v          # fusion correlation demo
```

On the Jetson, add extras: `pip install -e ".[jetson,sdr,analytics]"`
(CuPy for cuFFT, UHD bindings for the B210).

The same commands via `make` (the Makefile is the canonical encoding of
the definition of done below; `make help` lists everything):

```bash
make install   # pip install -e ".[dev,analytics]" (PIP_FLAGS=--break-system-packages by default)
make check     # install + lint + test + smoke — the definition of done
make lint      # ruff check rffusion tests
make fmt       # ruff format + ruff check --fix (writes files)
make test      # pytest -q
make smoke     # the three synthetic smokes; survey logs to ./.smoke (gitignored)
make dev       # run survey against the synthetic SDR until Ctrl-C
```

There is no deploy step: on the Jetson, copy `configs/rffusion.example.yaml`
to `/etc/rffusion/rffusion.yaml` and run `python -m rffusion.survey -c ...`
(see README). Exact pins the toolchain was last verified with live in
`requirements-dev.txt`; `.python-version` is 3.12; `.devcontainer/` runs
`make check` on create.

## Verification loop

Run `make check` after every change. Do not report a task done until it
passes. If a check cannot run locally, say so explicitly. CUDA, the B210,
Kismet and the camera are never available in `make check`; the synthetic
fallbacks are what it exercises, and anything hardware-only is reported
as unverified, not as passing.

## Non-negotiable design rules

These are load-bearing. Don't violate them without a very good reason, and
if you do, update this file.

1. **Fuse on monotonic time, label with UTC.** Everything stamps against
   `common/timebase.capture_ns()` (`CLOCK_MONOTONIC_RAW`). Wall-clock/UTC
   is derived separately via `GpsClock` and used only for logging. A GPS
   lock mid-drive must never warp the fusion clock. Never sort, window, or
   correlate on UTC. Other clock domains (a GPSDO-disciplined SDR's own
   time) are mapped INTO the monotonic domain via `DeviceClock` before
   anything downstream sees them — never fused on directly.

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
- **Device time is a THIRD clock domain.** UHD's `time_spec` is the USRP's
  own clock, not the monotonic base. It never reaches the ring raw — it
  goes through `timebase.DeviceClock`, which anchors device↔host and
  converts, so rule 1 still holds. Don't "simplify" by stamping blocks
  with `time_spec` directly; that quietly reintroduces a clock nobody
  else in the repo shares. `test_device_clock_*` guards the mapping.
- **The time fit is computed relative to the first anchor**, not on
  absolute ns. Absolute stamps are ~1e13+, where float64 loses exactly
  the sub-microsecond detail PPS sync exists to provide.
  `test_linear_clock_keeps_resolution_at_realistic_magnitudes` guards it.
- **YAML 1.1 needs a SIGNED exponent.** PyYAML resolves `20.0e6` to a
  *string* and only `20.0e+6` to a float — and a str `sample_rate` doesn't
  fail at load, it fails much later comparing str to int. `SdrConfig.
  __post_init__` coerces the numerics, and the shipped example uses the
  signed form. `test_unsigned_exponent_yaml_still_yields_numbers` guards it.
- **`configs/rffusion.example.yaml` is load-tested.** It's what users copy
  to `/etc/rffusion`, and a key indented under the wrong section (or an
  unparseable scalar) surfaces nowhere else.
  `test_shipped_example_config_loads` guards it.

## Where to work next

- Build 1: GPSDO/PPS timing is IMPLEMENTED (`sdr.pps_sync`) but has only
  ever run against the synthetic device clock. Unverified on hardware:
  needs a B210 + GPSDO to confirm PPS alignment and real anchor error.
  Prerequisite for real fusion timing.
- Build 3: DeepStream camera source feeding `Track` into
  `CorrelationEngine`; the correlation core is done, the camera wiring is
  the scaffold.

## Definition of done for any change

`pytest -q` green, `ruff check` clean, and if you touched a hardware path,
its synthetic fallback still runs the matching `python -m rffusion.<build>`
smoke command.

`make check` runs exactly this (all three smokes, every time). The Stop
hook in `.claude/settings.json` runs it when a session ends.

## Hardware & constants (as found in the repo — don't invent)

- Board: NVIDIA Jetson Orin Nano (Super); extras `jetson` (`cupy-cuda12x`),
  `sdr` (`pyuhd`, aarch64 only).
- SDR: Ettus B210, UHD args `type=b200`; default `sample_rate` 20.0e+6
  (B210 max 56e6), `center_freq` 915.0e+6, `gain` 40 dB, `block_size`
  262144, `ring_depth` 16. Timing knobs under `sdr:`: `pps_sync`,
  `clock_source` / `time_source` ∈ {internal, external, gpsdo},
  `pps_timeout_s` 3.0, `time_anchor_interval_s` 10.0.
- GPS: gpsd at `localhost:2947`.
- Kismet: `ws://host:2501/devices/monitor.ws?KISMET=<token>`; token from
  `~/.kismet/kismet_httpd.conf`, configured as `kismet.api_token` (keep it
  out of git — the shipped example is empty).
- Camera (Build 3): CSI `sensor_id` 0, 1280x720 @ 30 fps, DeepStream.
- Output: `output.log_dir` `/var/lib/rffusion` → falls back to
  `$XDG_STATE_HOME/rffusion`; `--out DIR` overrides with no fallback.
- Env vars: `RFFUSION_LOG_DIR` (overrides `output.log_dir`),
  `XDG_STATE_HOME`.
- Config file: `configs/rffusion.example.yaml` → `/etc/rffusion/rffusion.yaml`.

## Standing rules

- No new dependencies without asking. `pyproject.toml` extras are
  deliberate (see the CUDA note there).
- Prefer small commits; one logical change each.
- Plan before multi-file changes, especially anything under
  `rffusion/common/`.
- Write or extend a test in `tests/test_core.py` with every behavior
  change; the gotchas list above is the pattern.
- Secrets (Kismet token, anything else) live in env vars or the local
  `/etc/rffusion/rffusion.yaml` only. Never commit credentials.
- Project hooks (`.claude/settings.json`): ruff-format on every edited
  `.py`; `.claude/hooks/guard.sh` blocks force-push, `rm -rf /`,
  `git reset --hard origin*`, and `--no-verify`.

## Decision log

Architecture decisions live in `docs/adr/`. ADRs 0001–0005 are the
non-negotiable rules and scope boundaries above, with rationale and the
tests that guard each. New load-bearing decisions get a new ADR (copy
`docs/adr/0000-template.md`) in the same PR. Reference material index:
`docs/reference/README.md`.

## Open backlog

No TODO/FIXME markers in code as of 2026-11-09. Open work is the "Where
to work next" section above (PPS/GPSDO hardware verification; DeepStream
camera source for fusion), plus: the tree is not `ruff format`-clean (13
files) — `make fmt-check` is advisory until a one-time format commit
lands, after which it should be added to `make check`.

## Before merging

Run the `agent-protocols:code-reviewer` agent (or `/review`) on the full
diff of the branch. For anything touching `rffusion/common/` (the shared
time base and ring), `survey/sdr.py` (hardware control), the Kismet token
path in `analytics/feed.py`, or auth of any kind, also run
`security-auditor`. Merge only with `make check` green in CI (`check.yml`)
and the reviewer findings addressed or explicitly deferred in the PR.
