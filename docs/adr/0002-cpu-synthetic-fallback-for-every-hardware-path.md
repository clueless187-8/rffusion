# 0002 — Every hardware dependency has a CPU/synthetic fallback

- Status: Accepted
- Date: 2026-11-09
- Source: `CLAUDE.md` "Non-negotiable design rules" #2; `README.md` "Why these run anywhere"; `pyproject.toml` optional extras

## Context

Target hardware is a Jetson Orin Nano with CUDA, an Ettus B210, Kismet and
a CSI camera. Development and CI happen on x86 with none of those. The
`jetson` and `sdr` extras are deliberately not default dependencies so the
package installs on CI without CUDA.

## Decision

From CLAUDE.md, verbatim:

> **Every hardware dependency has a CPU/synthetic fallback.** The whole
> repo runs and tests on x86 with no CUDA, no SDR, no Kismet, no camera:
> - `common/iobuf` → CuPy if present, else NumPy (`BACKEND_NAME`)
> - `survey/sdr` → UHD B210 if importable, else `SyntheticSource`
> - `analytics/feed` → Kismet websocket if reachable, else `synthetic_feed`
> - `fusion` → `--demo` runs the core on a synthetic co-occurring stream
> Keep it that way. New hardware paths get a fallback in the same commit.

## Consequences

- `make check` (and CI) run the three `python -m rffusion.<build>` smoke
  commands on the synthetic paths; they are part of the definition of done.
- A broken CUDA install fails loudly at import in `common/iobuf`, not
  silently on the NumPy path.
- PPS/GPSDO timing (`sdr.pps_sync`) has only been exercised against the
  synthetic device clock; hardware verification is still open.
