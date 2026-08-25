# Build 3 — Camera + RF Fusion

Correlate RF identifiers to camera-detected object tracks. The flagship
"only a Jetson can do this" build, because the GPU runs vision inference
(DeepStream) concurrently with the RF work.

## The hard part is time, not ML

Three async streams: video frames (~33 ms), RF/BLE detections (bursty,
irregular), GPS (1 Hz). Fusing them is a **time-synchronization problem
before it is an ML problem.** Everything stamps against the ONE monotonic
clock in `common/timebase` at the moment of capture. Never fuse on
wall-clock — it isn't monotonic and a GPS lock warps it.

For this to work, Build 1's SDR timing must be sample-accurate (GPSDO +
PPS), and the camera frames must carry a capture stamp from the same
monotonic source. Getting that common stamp right is 80% of the build.

## The correlation core

`fusion/correlate` accumulates windowed co-occurrence between RF idents
and camera track IDs. An ident seen within ±window of a track scores one
hit against that track (once per event, not once per buffered sample — the
naive version inflates the count past the number of events and breaks the
metric). The score is Jaccard-like: an ident that appears whenever a track
does, and only then, approaches 1.0.

This is the research surface. The accumulator is provided; what you do
with a strong association is your experiment.

## DeepStream

Camera detection/tracking runs in DeepStream (NVIDIA's native pipeline,
excellent on Orin), emitting `Track(track_id, label, mono_ns)` into the
correlation engine. Labels are object classes — "car", "truck" — never
person identities.

## Contention

You are now sharing GPU, memory bandwidth, and USB across camera inference
AND RF capture. Power mode and thermals are no longer a footnote. Profile
with `tegrastats`; if the FFT starves the tracker or vice versa, drop SDR
bandwidth before frame rate.

## Scope — the sharp line

Associating RF identifiers to camera-detected OBJECTS in the ambient
environment is interesting systems work. Associating them to identifiable
PEOPLE — faces, named individuals — is building a surveillance tool, and
this project does not do that. Point it at traffic and spectrum, not
people.
