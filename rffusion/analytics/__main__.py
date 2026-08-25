"""rffusion-analytics — Build 2 entrypoint.

Connects to Kismet's websocket device feed and runs the AnalyticsEngine.
Without a reachable Kismet (CI/dev), replays a synthetic device stream so
the pipeline is demonstrable. Real Kismet wiring is in feed.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..common.config import Config
from .engine import AnalyticsEngine
from .feed import select_feed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RF-environment analytics")
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("--events", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    log = logging.getLogger("rffusion.analytics")

    cfg = Config.load(args.config)
    engine = AnalyticsEngine()
    feed = select_feed(cfg.kismet)

    n = 0
    try:
        for obs in feed:
            engine.ingest(obs)
            n += 1
            if args.events and n >= args.events:
                break
    except KeyboardInterrupt:
        pass
    finally:
        log.info("processed %d observations, %d devices, %d anomaly alerts",
                 n, engine.device_count, engine.alert_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
