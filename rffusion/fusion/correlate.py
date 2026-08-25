"""Build 3 — camera + RF fusion: the temporal correlation core.

The hard part of fusion is not the ML, it is time synchronisation. Video
frames (~33 ms), RF/BLE detections (bursty, irregular), and GPS (1 Hz)
arrive on different clocks and cadences. Everything here fuses on the ONE
monotonic capture clock from common.timebase — never on wall-clock.

The correlation itself is the research surface: a RF identifier that
consistently co-occurs in time with a particular camera track is a
candidate association. This module provides the co-occurrence accumulator;
what you *do* with an association is your experiment.

Scope boundary, again in code: associate RF identifiers to camera-detected
OBJECTS (vehicles, etc.) in the ambient environment. Associating them to
identifiable PEOPLE is building a surveillance tool and is out of scope for
this project by design.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

LOG = logging.getLogger("rffusion.fusion")


@dataclass
class Track:
    """A camera-detected object track (from DeepStream / a tracker)."""

    track_id: int
    label: str                 # "car", "truck", ... never a person identity
    mono_ns: int               # frame capture time
    present: bool = True       # in-frame at this timestamp


@dataclass
class RfEvent:
    """An RF identifier observed at a capture instant."""

    ident: str                 # MAC, or a behavioural key from Build 2
    mono_ns: int
    rssi: float | None = None


@dataclass
class _Cooc:
    """Co-occurrence tally between one (ident, track_id) pair."""

    hits: int = 0
    ident_total: int = 0
    track_total: int = 0
    last_ns: int = 0


class CorrelationEngine:
    """Windowed temporal co-occurrence between RF idents and camera tracks.

    An RF event at t is considered co-present with any track seen within
    +/- window_ns of t. Over many observations, a genuine association
    accumulates hits while coincidental overlaps stay near the base rate.

    The score is a Jaccard-like ratio: hits / (ident_total + track_total -
    hits), so an ident that appears whenever a track does, and only then,
    approaches 1.0.
    """

    def __init__(self, window_ns: int = 500_000_000) -> None:  # 500 ms
        self.window_ns = window_ns
        self._recent_tracks: list[Track] = []
        self._cooc: dict[tuple[str, int], _Cooc] = defaultdict(_Cooc)
        self._ident_totals: dict[str, int] = defaultdict(int)
        self._track_totals: dict[int, int] = defaultdict(int)

    def observe_tracks(self, tracks: list[Track]) -> None:
        """Feed the current frame's tracks."""
        for t in tracks:
            if t.present:
                self._track_totals[t.track_id] += 1
        self._recent_tracks.extend(t for t in tracks if t.present)
        self._prune(tracks[-1].mono_ns if tracks else None)

    def observe_rf(self, event: RfEvent) -> list[tuple[int, float]]:
        """Feed one RF event; return current associations [(track_id, score)].

        Associations are updated against every track within the time
        window, and the ranked candidate list is returned so a caller can
        act on the strongest.
        """
        self._ident_totals[event.ident] += 1
        self._prune(event.mono_ns)

        # One hit per DISTINCT track within the window, not one per buffered
        # track sample — otherwise a track re-observed every frame inflates
        # the co-occurrence count far past the number of RF events and the
        # Jaccard denominator goes negative.
        matched: set[int] = set()
        for t in self._recent_tracks:
            if abs(t.mono_ns - event.mono_ns) <= self.window_ns:
                matched.add(t.track_id)

        results: list[tuple[int, float]] = []
        for tid in matched:
            c = self._cooc[(event.ident, tid)]
            c.hits += 1
            c.last_ns = event.mono_ns
            results.append((tid, self.score(event.ident, tid)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def score(self, ident: str, track_id: int) -> float:
        """Jaccard-like association strength in [0, 1]."""
        c = self._cooc.get((ident, track_id))
        if c is None:
            return 0.0
        it = self._ident_totals.get(ident, 0)
        tt = self._track_totals.get(track_id, 0)
        denom = it + tt - c.hits
        return c.hits / denom if denom > 0 else 0.0

    def _prune(self, now_ns: int | None) -> None:
        """Drop tracks older than the window to bound memory."""
        if now_ns is None:
            return
        cutoff = now_ns - self.window_ns
        self._recent_tracks = [t for t in self._recent_tracks if t.mono_ns >= cutoff]

    def top_associations(self, min_score: float = 0.3) -> list[tuple[str, int, float]]:
        """All (ident, track_id, score) above a threshold, best first."""
        out = [
            (ident, tid, self.score(ident, tid))
            for (ident, tid) in self._cooc
        ]
        out = [x for x in out if x[2] >= min_score]
        out.sort(key=lambda x: x[2], reverse=True)
        return out
