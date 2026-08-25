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
good to a few ms, fine for a survey. For sample-accurate time (needed if
you later feed Build 3), discipline the B210 from a GPSDO and use the
UHD `time_spec` metadata against PPS instead. That's the difference
between "roughly when" and "which sample."

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
