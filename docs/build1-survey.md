# Build 1 — Wideband Spectrum Survey

GPU FFT over a wide capture, energy detection, GPS-stamped detection log.
The cleanest of the three pipelines and the right place to start: it proves
out your sample-plumbing (pinned/unified memory, drop-oldest ring) before
you add the harder builds on top.

## SDR choice

The GPU only earns its keep at wide bandwidth. At RTL-SDR's ~2.4 MHz you
don't need CUDA at all.

| SDR | Usable BW | Notes |
|---|---|---|
| RTL-SDR | ~2.4 MHz | proof-of-concept only; GPU is overkill here |
| Airspy R2/Mini | ~10 MHz | decent middle ground |
| **Ettus B210** | **up to 56 MHz** | your target — this is where cuFFT wins |

Start at `sample_rate: 20e6` and raise toward 56e6 as thermals allow. The
B210 over USB 3 sustains high rates but the Orin's USB controller and your
FFT throughput both cap out somewhere — find it empirically with the drop
counter (`dropped_before` in the logs) as your signal.

## Timing accuracy

The default stamps each block with `capture_ns()` at recv completion —
good to a few ms, fine for a survey. That is the difference between
"roughly when" and "which sample," and Build 3 needs the latter.

Set `sdr.pps_sync: true` to get it. With a GPSDO (or an external 10 MHz +
PPS) the survey stamps each block from UHD's `time_spec` — the device
time of the buffer's FIRST sample, i.e. the actual ADC instant — instead
of whenever the host noticed the buffer had finished crossing USB.

```yaml
sdr:
  pps_sync: true
  clock_source: gpsdo      # internal | external | gpsdo
  time_source: gpsdo
```

### It still fuses on monotonic

`time_spec` lives in the USRP's own clock domain. Stamping blocks with it
directly would put a third clock into a repo whose whole premise is one
monotonic base, so it doesn't. `common/timebase.DeviceClock` anchors the
two domains against each other — device time read alongside host time,
sandwiched between two host reads so the USB round trip biases out — and
every block is converted to monotonic before it reaches the ring. The
repo rule is unchanged: **fuse on monotonic, label with UTC.**

What this buys is *spacing*. Block-to-block intervals become exact,
driven by the disciplined oscillator rather than by however the host
scheduler felt. The absolute offset still carries the anchor read's
error, but that is a roughly constant bias instead of per-block noise,
and re-anchoring (`sdr.time_anchor_interval_s`) averages it down while
the fit's slope absorbs oscillator rate difference over a long capture.

Measured on the synthetic source, host-stamped vs PPS-stamped:

| | distinct block deltas | stdev |
|---|---|---|
| host recv-completion | 45 of 45 | ~1.5 ms |
| PPS device time | 1 of 46 | 0 ns |

Caveat: the synthetic clock is ideal by construction, so that 0 is a
floor, not a hardware result. It proves the plumbing, not the GPSDO.
Real numbers need the B210.

### Failure behaviour

No PPS edge within `sdr.pps_timeout_s`? It logs a warning and falls back
to host stamps rather than refusing to run — a survey with coarse time
beats no survey. Check for that warning before trusting a capture you
intend to feed Build 3. A typo in `clock_source`/`time_source` is
rejected at config load, because UHD silently ignores an unknown source
and you would get undisciplined time that still looked synced.

## GPU FFT backend

`survey/spectrum` uses CuPy's cuFFT when present. If cuFFT versioning
fights your JetPack, VkFFT is the escape hatch (Vulkan-based, JetPack-
agnostic). The FFT is batched Welch-style — overlapping windowed frames,
averaged — for a stable noise floor to threshold against.

## Power & thermal

Put the Orin in MAXN: `sudo nvpmodel -m 0 && sudo jetson_clocks`. Expect
15–25 W under sustained cuFFT and a working fan. This is a 12V→19V
car-power build, not a battery build — and the ignition-shutdown logic
from the Pi `car-power.md` matters MORE here, because you're protecting an
NVMe mid-write under GPU load.

## Output

Newline-delimited JSON, one detection per line:
```json
{"mono_ns": 384811075943, "utc_ns": null, "freq_hz": 908999023.4, "bw_hz": 29296.9, "power_db": 51.13}
```
`utc_ns` is null until a GPS fix feeds the GpsClock. Post-process into a
GPS-tagged RF-energy map.
