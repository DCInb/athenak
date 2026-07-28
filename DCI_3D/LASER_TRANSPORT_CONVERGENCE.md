# Laser transport convergence

The first production attempt exposed a hidden terminal-ray loss: with one allowed
critical-surface reflection, roughly 44% of launched power was classified as
`remaining`.  That quantity was included in the old accounting residual but was neither
deposited nor escaped.  The run was stopped at `t=0.00152 ns`, the causes were split in
the diagnostics, and non-negligible remainder was made fatal.

All sweeps below used eight V100 ranks, 4,096 Gaussian rays, the production CH/He tables,
straight transport, and one complete RK2 cycle.  Every reported wave remainder was zero.

## Reflection-cap sweep

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

The production deck retains the doubled converged cap of 64.  This is a maximum, so it
does not add iterations once all rays have deposited or escaped.  Although 64 MPI waves
already produced zero wave remainder in every sweep, production retains the code default
of 1,024 as a decomposition-safety margin.

## Turning-offset sweep

The production-layout cap-64 results around the selected offset were:

| Offset (cell widths) | Deposited power | Escaped power | Total path |
|---:|---:|---:|---:|
| 1e-6 | 1.818173737908249e-3 | 8.080273568309e-9 | 9.512684551505e3 |
| 1e-5 | 1.818173736470490e-3 | 8.081711326555e-9 | 9.512684405597e3 |
| 1e-4 | 1.818173722078494e-3 | 8.096103323855e-9 | 9.512702916225e3 |

The 1e-5 and 1e-6 results differ by less than `1e-8` of launched power and less than
`2e-8` in relative total path.  The selected `1e-5` offset is therefore in the converged
small-offset regime without relying on a machine-epsilon displacement.

## Production acceptance

The binary prints total, MPI-wave, and reflection-cap remainder power and ray counts for
every laser solve.  It exits nonzero when total remainder exceeds `1e-10` of launched
power.  Gate schema 5 independently parses every smoke-phase laser record, requires the
split diagnostics, and applies the same remainder, accounting, and conservation limit.
