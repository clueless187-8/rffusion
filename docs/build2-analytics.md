# Build 2 — RF-Environment Analytics

Streaming analytics on Kismet's live device feed. No SDR. The most novel
of the three, and the one that runs on hardware you already have.

## Kismet feed

`analytics/feed` connects to Kismet's device websocket with an API token
(from `~/.kismet/kismet_httpd.conf`). Set `kismet.enabled: true` and the
token in config. Without a reachable Kismet it replays a synthetic device
stream with a planted anomaly, so the pipeline is demonstrable anywhere.

## Features, not identity

`analytics/engine` extracts *behavioural* features — sighting cadence,
RSSI mean/std, channel spread — not identity. These survive MAC
randomization, which is the point: randomized MACs are grouped into an
aggregate key rather than pretending each random MAC is a new device.

## Where the GPU helps (and doesn't)

Honestly: at wardriving data rates, the clustering and anomaly detection
here run fine on the CPU, and the module stays there on purpose. The GPU
becomes worthwhile only when you enrich the feature vector into something
heavy — a behavioural autoencoder, or PHY-layer fingerprinting fed by the
SDR (clock offset, transient RF signature). Swap `WelfordAnomaly` for a
TensorRT model at that point.

## The anomaly detector

`WelfordAnomaly` is a streaming z-score over running mean/variance
(Welford's online algorithm) — O(1) memory, single pass, explainable. It
scores only over features that have observed variance, so a constant
feature can't dilute the score to zero. Deliberately simple: a live,
interpretable signal beats a black box you can't debug in a moving car.

## Scope

Fingerprints DEVICES and characterises the environment. Not for
re-identifying or tracking specific people. Keep it on your own gear and
the aggregate ambient RF.
