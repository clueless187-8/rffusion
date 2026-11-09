# 0003 — A build must justify the GPU

- Status: Accepted
- Date: 2026-11-09
- Source: `CLAUDE.md` "Non-negotiable design rules" #3; `README.md` design principle and "Three rules" #3

## Context

The Jetson is more expensive, hotter and more power-hungry than a Pi. Plain
wardriving is explicitly a Pi job (README). Analytics clustering/anomaly
detection is intentionally CPU-side.

## Decision

From CLAUDE.md, verbatim:

> **A build must justify the GPU.** If a feature doesn't saturate CUDA or
> need sub-frame fusion, a Pi would have done it — question whether it
> belongs on this board. Analytics clustering/anomaly is CPU on purpose.

## Consequences

- Survey (GPU FFT at 20–56 MS/s) and fusion (sub-frame correlation) are
  the GPU-justified builds; analytics rides along because it feeds fusion.
- A proposal to move Welford/anomaly scoring to CuPy needs a measured
  bottleneck first, not a hunch.
