# Laser transport convergence

The first production attempt exposed a hidden terminal-ray loss: with one allowed
critical-surface reflection, roughly 44% of launched power was classified as
`remaining`.  That power was included in the old accounting residual but was neither
deposited nor escaped.  The run was stopped at `t=0.00152 ns`, cause-specific remainder
diagnostics were added, and any remainder above `1e-10` of launched power became fatal.

## Initial-condition cap sweep

The first sweep used eight V100 ranks, 4,096 Gaussian rays, the production CH/He tables,
straight transport, and one complete RK2 cycle.

| Mesh | Reflection cap | Maximum reflection remainder / launched | Result |
|---|---:|---:|---|
| 100 x 64 x 64 | 1 | 3.1580487e-2 | rejected |
| 100 x 64 x 64 | 2 | 7.6007352e-3 | rejected |
| 100 x 64 x 64 | 4 | 5.6346644e-4 | rejected |
| 100 x 64 x 64 | 8 | 0 | passed |
| 100 x 64 x 64 | 16 | 0 | identical to cap 8 |
| 500 x 256 x 256 | 8 | 1.3180711e-7 | rejected |
| 500 x 256 x 256 | 16 | 3.5710657e-8 | rejected |
| 500 x 256 x 256 | 32 | 0 | passed |
| 500 x 256 x 256 | 64 | 0 | identical to cap 32 |

This established the initial hard-limit requirement but did not prove evolved-state
convergence.  At `200 x 128 x 128`, a no-hysteresis 100-cycle smoke run later reached
64 reflections on one ray at cycle 46, leaving `5.6401e-5` of launched power.  Its
33,351 aggregate turns, versus about two turns per ray on the coarser mesh, and much
shorter distance per turn identify cell-local critical-surface chatter rather than a
physical 64-bounce trajectory.

## Turning-surface rearm

After a specular turn, a ray is disarmed and moved a small distance toward the underdense
side along the surface normal.  It rearms only after traversing one complete segment for
which the reconstructed electron density stays below the saved turning density by the
configured hysteresis fraction.  The armed state and saved cutoff migrate with the ray
across MPI ranks.  A forward fractional-cell lookup probe assigns a reflected ray to the
MeshBlock on its direction side, preventing zero-distance rank-boundary ping-pong.

The selected hysteresis is `1e-2`; the selected normal offset is `1e-5` cell widths.
Cap 64 remains a hard failure guard, not the mechanism used to terminate chatter.

## Evolved-state sweep

The matched runs below used the same final CUDA binary, `200 x 128 x 128` uniform cells,
eight V100s, and 100 RK2 cycles.  Every passing run contains 200 laser solves with zero
total, wave, and reflection remainder and maximum conservation residual below `6.1e-14`.
Values are compared at the common history time `t=0.002 ns`; incident laser energy by
that time is `3.636363636e-6` code units.

| Variant | Value | Result | Max turns/ray | Deposited energy at 0.002 ns | Difference / incident energy |
|---|---:|---|---:|---:|---:|
| no hysteresis | 0 | rejected at cycle 46 | 64 | n/a | n/a |
| hysteresis | 0.005 | passed | 17 | 3.116962219964e-6 | 7.17e-4 vs selected |
| hysteresis | **0.010** | **passed, selected** | 17 | 3.119567670020e-6 | 0 |
| hysteresis | 0.020 | passed | 18 | 3.119734162583e-6 | 4.58e-5 vs selected |
| reflection cap | 128 | passed | 17 | 3.119299138743e-6 | 7.38e-5 vs cap 64 |
| normal offset | 1e-6 | passed | 17 | 3.112486356308e-6 | 1.95e-3 vs selected |
| normal offset | **1e-5** | **passed, selected** | 17 | 3.119567670020e-6 | 0 |
| normal offset | 1e-4 | passed | 19 | 3.126739791916e-6 | 1.97e-3 vs selected |

The 0.5%, 1%, and 2% hysteresis band is converged below `1e-3` of incident energy.  Cap
64 and 128 agree well below that level and the selected run uses at most 17 turns, giving
more than a factor-three guard margin.  A decade offset change on either side perturbs
cumulative deposition by less than `2e-3` of incident energy, much less than the spatial
resolution uncertainty, while retaining zero terminal power.

## Production acceptance

Each laser solve prints total and cause-specific remainder power/counts, maximum turns
per ray, suppressed same-surface candidate segments, and rearm count.  Gate schema 7
parses every laser record from baseline, doubled-resolution, reduced-light-speed,
physical-light-speed, and calibration evidence.  It requires split accounting and
residuals at or below `1e-10`, zero terminal power, and an observed maximum no greater
than half the configured reflection cap (32 for production cap 64).

## Full-layout ownership regression

The first schema-6 production start exposed a separate floating-point ownership defect
on its seventh laser solve.  A negative-going ray reached the `x=-0.6 mm` boundary
between global blocks 448 and 393.  Rank-local MeshBlock bounds use AthenaK's symmetric
`LeftEdgeX` arithmetic, while the laser's replicated global-block table had used an
algebraically equivalent interpolation that differed by two ulps.  The local lookup
correctly left block 448, but the mismatched global lookup returned block 448 again and
failed one nearly absorbed ray.

The replicated laser table now uses the exact MeshBlock endpoint rules and `LeftEdgeX`
arithmetic on every active axis.  A two-rank asymmetric-domain regression crosses the
same non-binary face in the negative direction and compares its fields with a one-rank
reference.  The production-memory calibration now advances four full cycles (eight RK2
laser solves), rather than two, and must record cycle 4.  Gate schema 7 enforces that
coverage in addition to the eight-log remainder and reflection checks above.
