# Architecture Decision Records

One file per load-bearing decision. Status is one of Proposed, Accepted,
Deprecated, Superseded. Copy `0000-template.md` for a new one and number
it sequentially. ADRs 0001–0004 are the "Non-negotiable design rules" from
`CLAUDE.md`, and 0005 its "Scope boundaries", recorded verbatim; CLAUDE.md
stays the short-form source, these carry the rationale and consequences.

| # | Title | Status |
|---|---|---|
| [0001](0001-fuse-on-monotonic-time.md) | Fuse on monotonic time, label with UTC | Accepted |
| [0002](0002-cpu-synthetic-fallback-for-every-hardware-path.md) | Every hardware dependency has a CPU/synthetic fallback | Accepted |
| [0003](0003-a-build-must-justify-the-gpu.md) | A build must justify the GPU | Accepted |
| [0004](0004-drop-oldest-ring-never-blocks.md) | Getting samples onto the GPU is the hard part: drop-oldest ring, never block | Accepted |
| [0005](0005-scope-devices-and-objects-not-people.md) | Scope boundary: devices and objects, never people | Accepted |
