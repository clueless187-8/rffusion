"""Build 1 — wideband spectrum survey: the FFT/detection core.

Takes IQ blocks, produces power spectra and energy detections. Runs the
FFT on the GPU via CuPy/cuFFT when available, transparently falling back
to NumPy so this is testable on CI and a laptop.

This is deliberately the *honest* build: samples in, FFT, average,
threshold, emit detections. It degrades gracefully — if the consumer
can't keep up, IQRing drops oldest and the survey simply has gaps, rather
than corrupting or stalling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..common.iobuf import BACKEND_NAME, _xp

LOG = logging.getLogger("rffusion.survey.spectrum")


@dataclass
class Detection:
    """A band of energy above threshold within one integrated spectrum."""

    freq_hz: float          # center of the detected band
    bandwidth_hz: float     # width above threshold
    power_db: float         # peak power in the band, dB (relative)
    mono_ns: int            # capture time of the source block


class SpectrumEngine:
    """Batched Welch-style power spectral density with energy detection.

    Welch averaging (overlapping windowed FFTs, averaged) trades frequency
    resolution for variance reduction, which is what you want for a survey:
    a stable noise floor to threshold against, not a single noisy snapshot.
    """

    def __init__(self, nfft: int, sample_rate: float, center_freq: float,
                 overlap: float = 0.5, threshold_db: float = 10.0) -> None:
        if nfft & (nfft - 1) != 0:
            raise ValueError("nfft must be a power of two")
        self.nfft = nfft
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.hop = int(nfft * (1.0 - overlap))
        self.threshold_db = threshold_db
        # Hann window on the active backend (GPU array on Jetson).
        self._window = _xp.hanning(nfft).astype(_xp.float32)
        self._win_power = float((self._window ** 2).sum())
        # Frequency axis (host-side, computed once).
        self._freqs = (
            np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
            + center_freq
        )

    def _to_backend(self, data: np.ndarray):
        """Move a host block to the compute backend (no-op on NumPy)."""
        if BACKEND_NAME == "cupy":
            return _xp.asarray(data)
        return data

    def psd(self, iq: np.ndarray) -> np.ndarray:
        """Return an averaged power spectrum (dB) for one IQ block.

        Host-returned (NumPy) regardless of backend, so downstream logging
        and detection are backend-agnostic.
        """
        x = self._to_backend(iq)
        n = (len(x) - self.nfft) // self.hop + 1
        if n <= 0:
            raise ValueError("IQ block shorter than one FFT frame")

        acc = _xp.zeros(self.nfft, dtype=_xp.float32)
        for i in range(n):
            start = i * self.hop
            frame = x[start:start + self.nfft] * self._window
            spec = _xp.fft.fftshift(_xp.fft.fft(frame))
            acc += (_xp.abs(spec) ** 2).astype(_xp.float32)

        acc /= (n * self._win_power * self.sample_rate)
        # to dB, guard against log(0)
        power_db = 10.0 * _xp.log10(acc + 1e-20)

        if BACKEND_NAME == "cupy":
            return _xp.asnumpy(power_db)
        return np.asarray(power_db)

    def detect(self, power_db: np.ndarray, mono_ns: int) -> list[Detection]:
        """Threshold a spectrum above its noise floor and group into bands.

        Noise floor estimated as the median (robust to the very signals
        we're trying to detect). Contiguous bins above floor+threshold
        become one Detection.
        """
        floor = float(np.median(power_db))
        mask = power_db > (floor + self.threshold_db)
        bin_hz = self.sample_rate / self.nfft

        detections: list[Detection] = []
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            band = power_db[i:j]
            peak_idx = i + int(np.argmax(band))
            detections.append(Detection(
                freq_hz=float(self._freqs[peak_idx]),
                bandwidth_hz=float((j - i) * bin_hz),
                power_db=float(power_db[peak_idx] - floor),
                mono_ns=mono_ns,
            ))
            i = j
        return detections

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs
