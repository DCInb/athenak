# Historical DCI EOS restart scanner

This utility reads the preserved legacy two-material DCI checkpoint without converting
it.  The accepted artifact is fixed by SHA-256
`2a72f97fdd1c3608c57f8cd0b642755052135cff68b4d24c9db6fde2805fdb40`, time `1 ns`,
and cycle `48420`.  Its layout is derived from AthenaK commit `843f2a86`: five hydro
conserved fields, one `rho*X_CH` scalar, ion and electron internal-energy densities,
then twenty radiation-group energy densities.  Each 32-cubed MeshBlock includes two
ghost zones on every side in the restart payload.

The scanner uses the current `MaterialMixtureDevice` implementation and current CH/He
IONMIX tables twice: once with the producer's `clamp` bounds policy to identify and
measure endpoint violations, and once with `flash-extrapolate` to invert the stored
ion/electron energies.  The legacy `Y_CH/(1-Y_CH)` composition accessor is used so the
two-material arithmetic matches the producer path.  Only active cells are counted.

The required scanned EOS inputs are bulk density, `rho*Y_CH`, ion energy density, and
electron energy density.  Bulk density and both component energies must be finite and
positive; `rho*Y_CH` must be finite and its raw fraction must lie in `[0,1]`.  No claim is
made about the unscanned momentum, total-energy, or radiation arrays.  This is a
same-state recovery test of a real historical CH/He checkpoint, not a rerun of the
trajectory and not evidence for the new CH/Au/He layout.

Before any payload analysis, the scanner verifies SHA-256 for the restart and both EOS
tables.  It opens the restart only through binary input streams, then recomputes its size
and SHA-256 after the scan and refuses to write passing evidence if the identity changed.
The JSON records both restart digests, the read-only access mode, table SHA-256 values,
the Git HEAD present at build time, and hashes of the relevant scanner/EOS sources.

Build and run with the production CUDA/Kokkos environment:

```sh
source /home/mengqi/Research/bashrc_athenaK
cmake -S DCI_3D/historical_eos_restart_scan -B /tmp/dci-eos-scan \
  -DCMAKE_CXX_COMPILER=/home/mengqi/Research/athenak/kokkos/bin/nvcc_wrapper
cmake --build /tmp/dci-eos-scan --parallel 2
CUDA_VISIBLE_DEVICES=0 /tmp/dci-eos-scan/historical_eos_restart_scan \
  DCI_3D/build/rst/dci_3d.00001.rst \
  DCI_3D/material_tables/ch.2t_eos \
  DCI_3D/material_tables/he.2t_eos \
  DCI_3D/evidence/historical_eos_restart_t1.json
```

An optional fifth positional argument changes the number of MeshBlocks processed per
GPU chunk (default 32).  The scanner rejects a wrong artifact digest, time/cycle, required
deck parameter, uniform-root-grid layout, selected binary header field, stride,
precision, file size, table digest, or table fingerprint.
