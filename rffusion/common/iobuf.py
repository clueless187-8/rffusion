"""Host-side ring buffer for streaming IQ, with optional pinned memory.

The recurring lesson across all three builds: getting samples ONTO the
GPU efficiently is harder than the GPU math. A discrete-GPU laptop pays a
PCIe copy; the Jetson Orin shares physical RAM between CPU and GPU
(unified memory), so a correctly-allocated buffer can be handed to CUDA
with NO copy. Leaving that on the table wastes half the board's point.

This module abstracts that:
    * On a Jetson with CuPy present -> allocate pinned/managed memory so
      DMA to the GPU never stalls and (with managed memory) no explicit
      copy is needed.
    * Everywhere else (x86 CI, a dev laptop, a Pi) -> fall back to plain
      NumPy so the whole repo stays runnable and testable without CUDA.

The ring absorbs scheduling jitter between the USB SDR producer and the
GPU consumer. Size it for your worst-case stall, not your average.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Condition
from typing import Any

import numpy as np

LOG = logging.getLogger("rffusion.iobuf")


def _detect_backend() -> tuple[str, Any]:
    """Return (name, module). 'cupy' on Jetson/CUDA, else 'numpy'.

    Import is deferred and guarded so importing this module never fails on
    a machine without CUDA. CI runs the numpy path.
    """
    try:
        import cupy  # type: ignore
        # Touch the runtime so a broken CUDA install fails HERE, loudly,
        # not three layers deep in a capture callback.
        cupy.cuda.runtime.getDeviceCount()
        return "cupy", cupy
    except Exception as exc:  # noqa: BLE001 - any failure => fall back
        LOG.info("CUDA/CuPy unavailable (%s); using NumPy host buffers", exc)
        return "numpy", np


BACKEND_NAME, _xp = _detect_backend()


def on_gpu() -> bool:
    """True if the GPU (CuPy) backend is active."""
    return BACKEND_NAME == "cupy"


def alloc_iq(n: int, dtype=np.complex64):
    """Allocate an n-sample IQ buffer in the best available memory.

    On CuPy: pinned host memory, so producer writes are DMA-ready.
    On NumPy: a plain array.
    """
    if BACKEND_NAME == "cupy":
        # Pinned host allocation; the SDR fills this, CUDA DMAs from it.
        mem = _xp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
        return np.frombuffer(mem, dtype=dtype, count=n)
    return np.empty(n, dtype=dtype)


@dataclass
class Block:
    """One captured chunk handed from producer to consumer."""

    data: np.ndarray          # IQ samples (complex64)
    mono_ns: int              # capture timestamp (see common.timebase)
    seq: int                  # monotonically increasing block index
    dropped_before: int = 0   # blocks lost to overrun since last delivery


class IQRing:
    """Single-producer / single-consumer ring of fixed-size IQ blocks.

    Overrun policy is DROP-OLDEST: in a live RF capture, stale samples are
    worthless, so when the consumer (GPU) falls behind we discard the
    oldest block and count it rather than blocking the SDR, which would
    cause a USB overflow and a much larger gap.
    """

    def __init__(self, block_size: int, depth: int = 16,
                 dtype=np.complex64) -> None:
        self.block_size = block_size
        self.depth = depth
        self._dtype = dtype
        self._slots: list[Block | None] = [None] * depth
        self._head = 0          # next write
        self._tail = 0          # next read
        self._count = 0
        self._seq = 0
        self._dropped = 0
        self._closed = False
        self._cv = Condition()

    def push(self, data: np.ndarray, mono_ns: int) -> None:
        """Producer: enqueue a block. Never blocks; drops oldest on full."""
        with self._cv:
            if self._count == self.depth:
                # Overrun: advance tail, discard oldest. Accumulate the
                # drop count — it is only cleared once a block actually
                # carries it out to the consumer in pop().
                self._tail = (self._tail + 1) % self.depth
                self._count -= 1
                self._dropped += 1
            self._slots[self._head] = Block(
                data=data, mono_ns=mono_ns, seq=self._seq,
                dropped_before=self._dropped,
            )
            self._seq += 1
            self._dropped = 0          # attributed to this block; reset
            self._head = (self._head + 1) % self.depth
            self._count += 1
            self._cv.notify()

    def pop(self, timeout: float | None = 1.0) -> Block | None:
        """Consumer: dequeue the oldest block, or None on timeout/close."""
        with self._cv:
            if not self._cv.wait_for(
                lambda: self._count > 0 or self._closed, timeout=timeout
            ):
                return None
            if self._count == 0:
                return None
            block = self._slots[self._tail]
            self._slots[self._tail] = None
            self._tail = (self._tail + 1) % self.depth
            self._count -= 1
            return block

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def depth_used(self) -> int:
        with self._cv:
            return self._count
