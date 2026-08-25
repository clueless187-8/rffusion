"""Build 2 — RF-environment analytics on the live Kismet feed.

No SDR required. Kismet exposes observed devices over REST/websocket; this
turns that stream into features and runs lightweight ONLINE models over it
(clustering, streaming anomaly detection). The GPU is optional here and
frankly unnecessary at wardriving data rates — this module stays on the
CPU on purpose and documents where a GPU would actually help (rich
behavioural fingerprinting, not this).

Scope boundary, stated in code because it matters: this fingerprints
DEVICES and characterises the ambient RF environment. It is not for
re-identifying or tracking specific PEOPLE across space. Keep it pointed
at your own gear and the aggregate environment.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

LOG = logging.getLogger("rffusion.analytics")


@dataclass
class DeviceObservation:
    """One sighting of a device from the Kismet feed."""

    mac: str
    mono_ns: int
    rssi: float
    channel: int | None = None
    ssid: str | None = None
    phy: str = "IEEE802.11"
    is_randomized_mac: bool = False


@dataclass
class DeviceFeatures:
    """Rolling behavioural features for one device key.

    Deliberately PHY/behaviour features, not identity: sighting cadence,
    RSSI statistics, channel spread. These are what survive MAC
    randomization and what make the environment characterisable.
    """

    key: str
    first_ns: int
    last_ns: int
    count: int = 0
    _rssi: deque = field(default_factory=lambda: deque(maxlen=64))
    _intervals: deque = field(default_factory=lambda: deque(maxlen=64))
    channels: set = field(default_factory=set)

    def update(self, obs: DeviceObservation) -> None:
        if self.count > 0:
            self._intervals.append(obs.mono_ns - self.last_ns)
        self.last_ns = obs.mono_ns
        self.count += 1
        self._rssi.append(obs.rssi)
        if obs.channel is not None:
            self.channels.add(obs.channel)

    @property
    def rssi_mean(self) -> float:
        return sum(self._rssi) / len(self._rssi) if self._rssi else 0.0

    @property
    def rssi_std(self) -> float:
        if len(self._rssi) < 2:
            return 0.0
        m = self.rssi_mean
        return math.sqrt(sum((r - m) ** 2 for r in self._rssi) / len(self._rssi))

    @property
    def mean_interval_ns(self) -> float:
        return sum(self._intervals) / len(self._intervals) if self._intervals else 0.0

    def vector(self) -> list[float]:
        """Feature vector for anomaly scoring."""
        return [
            self.rssi_mean,
            self.rssi_std,
            self.mean_interval_ns / 1e9,      # seconds
            float(len(self.channels)),
            float(self.count),
        ]


class WelfordAnomaly:
    """Streaming z-score anomaly detector (Welford's online mean/variance).

    Per-feature running mean and variance, single pass, O(1) memory. A
    device whose feature vector lands far from the running distribution of
    the environment scores high. This is intentionally simple and
    explainable — the point is a live, interpretable signal, not a
    black-box model. Swap in an autoencoder here if you later want the GPU
    to earn its keep.
    """

    def __init__(self, dim: int, warmup: int = 50) -> None:
        self.dim = dim
        self.warmup = warmup
        self._n = 0
        self._mean = [0.0] * dim
        self._m2 = [0.0] * dim

    def _update_stats(self, x: list[float]) -> None:
        self._n += 1
        for i in range(self.dim):
            delta = x[i] - self._mean[i]
            self._mean[i] += delta / self._n
            self._m2[i] += delta * (x[i] - self._mean[i])

    def score(self, x: list[float], learn: bool = True) -> float:
        """Return an anomaly score (mean abs z across features).

        During warmup, learns but returns 0.0 (no stable baseline yet).
        """
        if len(x) != self.dim:
            raise ValueError(f"expected {self.dim} features, got {len(x)}")

        score = 0.0
        if self._n >= self.warmup:
            total = 0.0
            scored = 0
            for i in range(self.dim):
                var = self._m2[i] / self._n if self._n > 0 else 0.0
                std = math.sqrt(var) if var > 0 else 0.0
                if std > 0:
                    total += abs(x[i] - self._mean[i]) / std
                    scored += 1
            # Average over features that actually have a scale to measure
            # against. A feature with zero observed variance carries no
            # information and must not dilute the score toward zero.
            score = total / scored if scored > 0 else 0.0

        if learn:
            self._update_stats(x)
        return score


class AnalyticsEngine:
    """Consumes DeviceObservations, maintains features, scores anomalies."""

    def __init__(self, feature_dim: int = 5, anomaly_threshold: float = 3.0) -> None:
        self._features: dict[str, DeviceFeatures] = {}
        self._anomaly = WelfordAnomaly(dim=feature_dim)
        self.anomaly_threshold = anomaly_threshold
        self._alert_count = 0

    def _key_for(self, obs: DeviceObservation) -> str:
        """Grouping key.

        Randomized MACs are grouped by (phy, channel-set-proxy) so we
        characterise them in aggregate rather than pretending each random
        MAC is a distinct persistent device. Non-randomized keep their MAC.
        """
        if obs.is_randomized_mac:
            return f"rand:{obs.phy}:{obs.channel}"
        return obs.mac

    def ingest(self, obs: DeviceObservation) -> float:
        """Process one observation; return its anomaly score."""
        key = self._key_for(obs)
        feat = self._features.get(key)
        if feat is None:
            feat = DeviceFeatures(key=key, first_ns=obs.mono_ns, last_ns=obs.mono_ns)
            self._features[key] = feat
        feat.update(obs)

        score = self._anomaly.score(feat.vector())
        if score >= self.anomaly_threshold:
            self._alert_count += 1
            LOG.info("anomaly key=%s score=%.2f rssi=%.0f chans=%d",
                     key, score, feat.rssi_mean, len(feat.channels))
        return score

    @property
    def device_count(self) -> int:
        return len(self._features)

    @property
    def alert_count(self) -> int:
        return self._alert_count
