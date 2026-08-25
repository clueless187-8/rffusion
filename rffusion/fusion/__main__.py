"""rffusion-fusion — Build 3 entrypoint (scaffold).

Fuses camera tracks + RF events on the common monotonic clock via
CorrelationEngine. Real camera (DeepStream) and capture wiring live behind
interfaces; this scaffold runs the correlation core against a synthetic
co-occurring stream so the association logic is demonstrable now. See
docs/build3.md for the time-sync design that everything hangs off.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .correlate import CorrelationEngine, RfEvent, Track


def _demo(engine: CorrelationEngine, log) -> None:
    ident = "AA:BB:CC:DD:EE:FF"
    for i in range(30):
        t = i * 50_000_000
        engine.observe_tracks([Track(track_id=7, label="car", mono_ns=t)])
        engine.observe_rf(RfEvent(ident=ident, mono_ns=t + 2_000_000))
    for ident, tid, score in engine.top_associations():
        log.info("assoc ident=%s track=%d score=%.2f", ident, tid, score)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="camera + RF fusion")
    ap.add_argument("--demo", action="store_true", help="run synthetic correlation demo")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    log = logging.getLogger("rffusion.fusion")

    engine = CorrelationEngine()
    if args.demo:
        _demo(engine, log)
        return 0
    log.info("live fusion requires camera + capture wiring — see docs/build3.md")
    log.info("run with --demo to exercise the correlation core")
    return 0


if __name__ == "__main__":
    sys.exit(main())
