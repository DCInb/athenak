# Benchmark sources and mapping

## Primary benchmark

- FLASH User Guide, Energy Deposition unit, section 18.4.12, Cartesian unit test I:
  https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node122.html
- Thomas B. Kaiser, Laser ray tracing and power deposition on an unstructured
  three-dimensional grid, Physical Review E 61, 895 (2000):
  https://doi.org/10.1103/PhysRevE.61.895
- FLASH laser-model tutorial containing the production LaserSlab example:
  https://flash.rochester.edu/site/flashcode/user_support/tutorial_talks/RAL_May2012/lasermgd.pdf

The production LaserSlab problem also requires cylindrical geometry, helium and aluminum
materials, tabulated multitemperature EOS, IONMIX opacity tables, and six-group radiation
diffusion. The current AthenaK implementation cannot reproduce that full configuration.
FLASH unit test I is therefore the reliable shared-physics benchmark: it tests refractive
ray paths and inverse-bremsstrahlung power deposition against a closed-form solution.

## Published parameters

- Critical-surface coordinates: xc = zc = 10 cm
- Tube center: xw = zw = 5 cm
- Ray launch radius: R = 3 cm
- Wavelength: 1.0e-4 cm
- Center electron temperature: 10 keV
- Center electron density: nw = nc/2
- Initial ray power: 1 erg/s
- Fixed Coulomb logarithm: 1
- Published analytic exit power: 0.7017811 erg/s per ray

Eight rays start at the four coordinate-axis points and four diagonal points on the
R = 3 cm circle. FLASH reports finite-resolution path and power errors for axis Ray 1
and diagonal Ray 5; the diagonal power errors are explicitly non-monotonic with refinement.

## AthenaK coordinate mapping

- FLASH propagation y maps to AthenaK x1.
- FLASH transverse x and z map to AthenaK x2 and x3.
- The normalized density is rho = 0.5 + 0.02*(x2^2 + x3^2).
- The electron temperature is Te = 10 keV*(rho/0.5)^(2/3).
- AthenaK initializes the dispersive axial wave vector consistently with
  ne/nc + |q|^2 = 1. This places the transverse focus at x1 = 2*pi cm while preserving
  FLASH's crossing time and inverse-bremsstrahlung attenuation integral.

## Benchmark-driven implementation changes

- Added an optional fixed inverse-bremsstrahlung Coulomb logarithm; the default remains
  the locally evaluated FLASH Debye-number expression.
- Added a density-power temperature profile to the existing laser-profile problem generator.
- Added second-order local density and force reconstruction in the refractive tracer.
  This prevents symmetry-axis rays launched on cell/block planes from stalling at a
  zero-length first step and is exact for the quadratic benchmark profile.

The failed pre-fix run that exposed the symmetry-plane stall is preserved in attempt1.
