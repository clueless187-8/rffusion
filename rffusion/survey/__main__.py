"""rffusion-survey — Build 1 entrypoint.

Runs source -> IQRing -> SpectrumEngine -> newline-delimited JSON of
detections, GPS-stamped when a fix is available. Works on a laptop
(synthetic source, NumPy FFT) and on the Jetson+B210 (UHD source, cuFFT)
with no code change — only config and installed extras differ.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

from ..common.config import Config
from ..common.iobuf import BACKEND_NAME, IQRing
from ..common.timebase import GpsClock
from .sdr import select_source
from .spectrum import SpectrumEngine


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="wideband spectrum survey")
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("--nfft", type=int, default=4096)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until Ctrl-C")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    log = logging.getLogger("rffusion.survey")

    cfg = Config.load(args.config)
    log.info("backend=%s driver=%s fs=%.1f MHz fc=%.3f MHz",
             BACKEND_NAME, cfg.sdr.driver, cfg.sdr.sample_rate / 1e6,
             cfg.sdr.center_freq / 1e6)

    ring = IQRing(block_size=cfg.sdr.block_size, depth=cfg.sdr.ring_depth)
    engine = SpectrumEngine(nfft=args.nfft, sample_rate=cfg.sdr.sample_rate,
                            center_freq=cfg.sdr.center_freq)
    gps = GpsClock()  # feed from gpsd elsewhere; empty => utc_ns null

    cfg.output.log_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output.log_dir / f"survey-{int(time.time())}.jsonl"
    log.info("logging detections to %s", out_path)

    src = select_source(cfg.sdr, ring)
    src.start()

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    t_end = time.monotonic() + args.seconds if args.seconds else None
    n_dets = 0
    try:
        with out_path.open("w") as fout:
            while not stop["flag"]:
                if t_end and time.monotonic() >= t_end:
                    break
                blk = ring.pop(timeout=1.0)
                if blk is None:
                    continue
                if blk.dropped_before:
                    log.debug("dropped %d blocks before seq %d",
                              blk.dropped_before, blk.seq)
                psd = engine.psd(blk.data)
                for d in engine.detect(psd, blk.mono_ns):
                    rec = {
                        "mono_ns": d.mono_ns,
                        "utc_ns": gps.to_utc_ns(d.mono_ns),
                        "freq_hz": round(d.freq_hz, 1),
                        "bw_hz": round(d.bandwidth_hz, 1),
                        "power_db": round(d.power_db, 2),
                    }
                    fout.write(json.dumps(rec) + "\n")
                    n_dets += 1
                fout.flush()
    finally:
        src.stop()
        ring.close()
        log.info("done: %d detections", n_dets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
