# Benchmark references

## Laser ray tracing and deposition

- T. B. Kaiser, “Laser ray tracing and power deposition on an unstructured
  three-dimensional grid,” *Physical Review E* **61**, 895–905 (2000),
  [doi:10.1103/PhysRevE.61.895](https://doi.org/10.1103/PhysRevE.61.895).
  This is the numerical reference for Hamiltonian/refractive laser transport and power
  deposition. AthenaK's separate FLASH laser-tube benchmark tests the analytic quadratic
  profile from this paper; this solid-target case instead exercises reflection, deposition,
  electron coupling, and MPI ray handoff in an evolving target.

- FLASH User Guide, “Energy Deposition,”
  <https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node122.html>.
  The inverse-bremsstrahlung convention, ray-power accounting, and electron deposition
  path follow this model.

## Electron thermal conduction

- FLASH 4.6.2 User Guide, “General Implicit Diffusion Solver” and “Thermal
  Conductivity,”
  <https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node122.html>.
  AthenaK follows the centered variable-coefficient stencil and theta method described
  there.  Conductivity and flux limiting are time lagged over a solve; the electron EOS
  energy remains nonlinear.  The same reference supplies the harmonic, min/max, and
  Larsen saturated-flux limiters and the fixed-temperature/zero-gradient boundary choices.

- FLASH 4.6.2 User Guide, “Spitzer High-Z,”
  <https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node139.html>.
  The `SpitzerHighZ` electron conductivity, including the (1/(1+3.3/\bar Z))
  correction, is the dimensional conductivity used by this benchmark.

## Biermann discretization and radiation-MHD implementation

- C. Graziani, P. Tzeferacos, D. Lee, D. Q. Lamb, K. Weide, M. Fatenejad, and
  J. Miller, “The Biermann Catastrophe in Numerical Magnetohydrodynamics,”
  *The Astrophysical Journal* **802**, 43 (2015),
  [doi:10.1088/0004-637X/802/1/43](https://doi.org/10.1088/0004-637X/802/1/43).
  This motivates the flux-form source and shock suppression. The benchmark starts from a
  smooth interface and reports the resolved crossed-gradient source so generated field is
  not interpreted as a discontinuity artifact.

- P. Tzeferacos, M. Fatenejad, N. Flocke, C. Graziani, G. Gregori, D. Q. Lamb,
  D. Lee, J. Meinecke, A. Scopatz, and K. Weide, “FLASH MHD simulations of
  experiments that study shock-generated magnetic fields,” *High Energy Density Physics*
  **17**, 24–31 (2015),
  [doi:10.1016/j.hedp.2014.11.003](https://doi.org/10.1016/j.hedp.2014.11.003).
  This provides the three-temperature radiation-MHD benchmark context: separate ion,
  electron, and radiation energies, multigroup diffusion, and Biermann generation. Here
  the control comparison tests the laser-to-electron-to-radiation energy route directly.

## Laser-solid magnetic-field topology

- J. A. Stamper, K. Papadopoulos, R. N. Sudan, S. O. Dean, E. A. McLean, and
  J. M. Dawson, “Spontaneous Magnetic Fields in Laser-Produced Plasmas,”
  *Physical Review Letters* **26**, 1012–1015 (1971),
  [doi:10.1103/PhysRevLett.26.1012](https://doi.org/10.1103/PhysRevLett.26.1012).
  This is the classic experimental reference for self-generated fields from laser plasma.
  The relevant comparison here is an azimuthal/toroidal field around the laser-heated
  region, represented by an odd `B3(x2)` pair in the 2D plane.

- N. Shukla, K. Schoeffler, E. Boella, J. Vieira, R. Fonseca, and L. O. Silva,
  “Interplay between the Weibel instability and the Biermann battery in realistic
  laser-solid interactions,” *Physical Review Research* **2**, 023129 (2020),
  [doi:10.1103/PhysRevResearch.2.023129](https://doi.org/10.1103/PhysRevResearch.2.023129).
  This provides a modern laser-solid comparison for where Biermann fields arise relative
  to density and temperature gradients. AthenaK is an MHD model and does not contain the
  kinetic Weibel mechanism, so only the large-scale Biermann topology is in scope.

- L. Willingale et al., “Fast Advection of Magnetic Fields by Hot Electrons,”
  *Physical Review Letters* **105**, 095001 (2010),
  [doi:10.1103/PhysRevLett.105.095001](https://doi.org/10.1103/PhysRevLett.105.095001).
  This is a useful experimental warning for interpretation: hot-electron transport can
  move laser-generated fields rapidly. The present test has neither a nonlocal hot-electron
  model nor Nernst advection, so it should not be used to benchmark late-time field motion.

## What is benchmarked versus what is deferred

The automated checks benchmark:

1. ray power conservation and critical-surface reflection;
2. MPI invariance of ray deposition and all coupled fluid/radiation fields;
3. laser deposition into the electron energy component;
4. radiation energy gain that disappears when matter-radiation exchange is disabled;
5. an antisymmetric/toroidal Biermann topology that disappears with zero coefficient;
6. closure of deposited laser energy against matter plus radiation energy.

The automated checks deliberately do not assert megagauss amplitudes, experimental
ablation-front positions, or late-time magnetic transport. Those require the missing
material and transport physics listed in `README.md` and an experiment-specific
dimensionalization.
