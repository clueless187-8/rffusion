"""Kismet device feed for Build 2.

Yields DeviceObservation from Kismet's device stream. Real path uses the
websocket eventbus; when Kismet isn't reachable (CI/dev) a synthetic
generator stands in so the analytics pipeline runs anywhere.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator

from ..common.config import KismetConfig
from ..common.timebase import capture_ns
from .engine import DeviceObservation

LOG = logging.getLogger("rffusion.analytics.feed")


def _is_randomized(mac: str) -> bool:
    """Locally-administered bit set in the first octet => randomized MAC."""
    try:
        first = int(mac.split(":")[0], 16)
    except (ValueError, IndexError):
        return False
    return bool(first & 0x02)


def synthetic_feed(count: int = 500) -> Iterator[DeviceObservation]:
    """Deterministic synthetic device stream with a planted anomaly.

    Most devices sit at a normal RSSI; one appears with an oddly strong,
    oddly regular signature so Build 2's detector has something to catch.
    """
    rng = random.Random(1234)
    base_macs = [f"AA:BB:CC:00:00:{i:02X}" for i in range(8)]
    for i in range(count):
        if i % 97 == 0:  # the planted oddball
            yield DeviceObservation(
                mac="DE:AD:BE:EF:00:01", mono_ns=capture_ns(),
                rssi=-20.0, channel=149, ssid="anomaly",
                is_randomized_mac=False,
            )
            continue
        mac = rng.choice(base_macs)
        yield DeviceObservation(
            mac=mac, mono_ns=capture_ns(),
            rssi=rng.gauss(-65, 6), channel=rng.choice([1, 6, 11, 36, 149]),
            ssid=None, is_randomized_mac=_is_randomized(mac),
        )


def kismet_websocket_feed(cfg: KismetConfig) -> Iterator[DeviceObservation]:  # pragma: no cover
    """Live Kismet feed over the device websocket.

    Requires `websockets` and a running Kismet with an API token. Kept out
    of the default test path because it needs a server.
    """
    import json

    import websockets.sync.client as ws  # type: ignore

    url = (f"ws://{cfg.host}:{cfg.port}/devices/monitor.ws"
           f"?KISMET={cfg.api_token}")
    LOG.info("connecting to Kismet at %s:%d", cfg.host, cfg.port)
    with ws.connect(url) as conn:
        conn.send(json.dumps({"monitor": "*", "request": 1, "rate": 1}))
        for message in conn:
            dev = json.loads(message)
            mac = dev.get("kismet.device.base.macaddr", "")
            if not mac:
                continue
            yield DeviceObservation(
                mac=mac,
                mono_ns=capture_ns(),
                rssi=float(dev.get("kismet.device.base.signal", {})
                           .get("kismet.common.signal.last_signal", -100)),
                channel=_safe_int(dev.get("kismet.device.base.channel")),
                ssid=dev.get("kismet.device.base.name") or None,
                phy=dev.get("kismet.device.base.phyname", "IEEE802.11"),
                is_randomized_mac=_is_randomized(mac),
            )


def _safe_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def select_feed(cfg: KismetConfig) -> Iterator[DeviceObservation]:
    """Live feed if Kismet is enabled and reachable, else synthetic."""
    if cfg.enabled:
        try:
            import websockets  # noqa: F401
            return kismet_websocket_feed(cfg)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Kismet feed unavailable (%s) — using synthetic", exc)
    return synthetic_feed()
