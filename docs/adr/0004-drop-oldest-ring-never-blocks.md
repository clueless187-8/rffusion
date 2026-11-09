# 0004 — Getting samples onto the GPU is the hard part: drop-oldest ring, never block

- Status: Accepted
- Date: 2026-11-09
- Source: `CLAUDE.md` "Non-negotiable design rules" #4; `README.md` "Three rules" #2; `rffusion/common/iobuf.py`

## Context

The B210 streams over USB 3. If the host stops draining, UHD overflows and
samples are lost anyway, with the added failure of a stalled device. The
FFT itself is cheap relative to moving the samples.

## Decision

From CLAUDE.md, verbatim:

> **Getting samples onto the GPU is the hard part, not the math.**
> `common/iobuf` uses pinned/unified memory on the Jetson and a
> drop-oldest ring. A slow consumer causes *gaps* (counted in
> `Block.dropped_before`), never blocking (which would USB-overflow) and
> never corruption. Preserve this contract.

## Consequences

- `IQRing.push` never waits on a consumer; the oldest slot is overwritten.
- Drop counts attribute to the next surviving block and reset after each
  stamp. Guarded by `test_ring_accumulates_consecutive_drops`.
- Downstream code must treat `dropped_before > 0` as a gap in time, not a
  contiguous stream.
