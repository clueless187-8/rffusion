# 0005 — Scope boundary: devices and objects, never people

- Status: Accepted
- Date: 2026-11-09
- Source: `CLAUDE.md` "Scope boundaries (in code, not just docs)"; `README.md` "Scope and ethics"

## Context

The same RF fingerprinting and RF↔camera correlation techniques can be
pointed at people. The repo draws the line in code, not only in docs.

## Decision

From CLAUDE.md, verbatim:

> - **analytics** fingerprints DEVICES and characterises the ambient
>   environment. NOT for re-identifying or tracking specific people. The
>   randomized-MAC grouping in `AnalyticsEngine._key_for` is deliberate —
>   don't "fix" it into per-person persistent tracking.
> - **fusion** associates RF idents to camera-detected OBJECTS (vehicles,
>   etc.). NOT to identifiable people. `Track.label` is an object class,
>   never a person identity. Don't add face recognition or person re-id.
>
> These are design constraints, not TODOs. Keep them.

## Consequences

- `AnalyticsEngine._key_for` aggregates randomized MACs by phy+channel;
  `test_*randomized*` in `tests/test_core.py` guards this.
- Pull requests adding face recognition, person re-identification, or
  per-person persistent keys are out of scope and should be declined.
