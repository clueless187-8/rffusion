# rffusion

GPU-accelerated RF survey, analytics, and camera+RF sensor fusion for the
**NVIDIA Jetson Orin Nano (Super)**, built around an **Ettus B210** wideband
SDR and, optionally, a live **Kismet** feed and a CSI camera.

This is a monorepo for three related builds that share one time base, one
sample-plumbing layer, and one config system:

| Build | What it does | Needs |
|---|---|---|
| **survey** | Wideband spectrum survey — GPU FFT, energy detection, GPS-stamped JSONL | B210 (or any Soapy/UHD SDR) |
| **analytics** | Streaming analytics on the live Kismet device feed — behavioural features, online anomaly detection | Kismet (no SDR) |
| **fusion** | Temporal correlation of RF identifiers to camera object-tracks on a common clock | camera + capture + the above |

The design principle throughout: **the Jetson earns its keep only when a
build saturates the GPU or needs sub-frame fusion.** Anything else a Pi
would have done. These three qualify; plain wardriving does not (use the
Pi builds for that).

---

## Why these run anywhere

Every hardware dependency has a clean software fallback, so the whole repo
is testable on x86 CI and a dev laptop with no CUDA, no SDR, no camera:

- **No CUDA?** `common/iobuf` detects CuPy; without it, everything runs on
  NumPy. Same code path, same results, slower.
- **No SDR?** `survey/sdr` falls back to a synthetic IQ source (noise +
  planted tones).
- **No Kismet?** `analytics/feed` replays a synthetic device stream with a
  planted anomaly.
- **No camera?** `fusion` runs the correlation core against a synthetic
  co-occurring stream (`--demo`).

On the Jetson you install the `jetson` and `sdr` extras and the same code
uses cuFFT and the B210 with no changes — only config differs.

---

## Quick start

```bash
# dev / CI (CPU path)
pip install -e ".[dev,analytics]"
pytest -q

# see each build run with synthetic inputs
python -m rffusion.survey    --seconds 5 -v      # synthetic SDR -> detections
python -m rffusion.analytics --events 500 -v     # synthetic Kismet -> anomalies
python -m rffusion.fusion    --demo -v           # correlation core
```

On the Jetson:

```bash
# install CuPy for your JetPack's CUDA, and UHD Python bindings for the B210
pip install -e ".[jetson,sdr,analytics]"
sudo install -Dm644 configs/rffusion.example.yaml /etc/rffusion/rffusion.yaml
# edit the config, then:
python -m rffusion.survey -c /etc/rffusion/rffusion.yaml -v
```

---

## Architecture

```
                         common/timebase  (ONE monotonic clock)
                                 │  every subsystem stamps here
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                           │
  SDR (B210) ─► common/iobuf ─► survey/spectrum          camera ─► fusion/
  UHD/synth    pinned ring       GPU cuFFT / CPU              DeepStream  correlate
                 drop-oldest      detection                        │        │
                                     │                             │        │
  Kismet ─► analytics/feed ─► analytics/engine ──────────────► RF events ───┘
  ws/synth                    features + anomaly            (common idents)
```

Three rules the whole repo obeys:

1. **Fuse on monotonic time, label with UTC.** `CLOCK_MONOTONIC_RAW` never
   jumps on a GPS lock; wall-clock does. `common/timebase` keeps them
   separate.
2. **Getting samples onto the GPU is the hard part, not the math.**
   `common/iobuf` uses pinned/unified memory on the Jetson so DMA never
   stalls, and a drop-oldest ring so a slow consumer causes gaps, not
   corruption or USB overflow.
3. **A build must justify the GPU.** If it doesn't saturate CUDA or need
   sub-frame fusion, it doesn't belong on this board.

---

## Per-build docs

- [`docs/build1-survey.md`](docs/build1-survey.md) — SDR choice, GPSDO/PPS
  timing, thermal/power, cuFFT vs VkFFT.
- [`docs/build2-analytics.md`](docs/build2-analytics.md) — Kismet feed
  wiring, feature design, where a GPU actually helps.
- [`docs/build3-fusion.md`](docs/build3-fusion.md) — the time-sync design
  everything hangs off, DeepStream integration.

---

## Scope and ethics

This is passive RF observation and ambient-environment analysis. Two lines
are drawn in the code and the docs, deliberately:

- **Analytics fingerprints devices and characterises the environment** — not
  for re-identifying or tracking specific people across space. Same
  technique, different target; the target is the line.
- **Fusion associates RF identifiers to camera-detected objects** (vehicles,
  etc.) — not to identifiable individuals. Correlating RF to faces or named
  people is building a surveillance tool and is out of scope by design.

Keep it pointed at your own gear and the aggregate spectrum. Passive
reception and WiGLE-style logging are broadly lawful in the US; associating
and tracking people is a different thing entirely.

---

## License

MIT. See [LICENSE](LICENSE).
