#ifndef DCI_3D_DCI_AMR_HPP_
#define DCI_3D_DCI_AMR_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! Material-following refinement of the DCI target. Levels are relative to the root grid.

#include "athena.hpp"

struct DCIAMRPolicy {
  Real density_threshold;
  int dense_level;
  int ch_level;
  int gold_level;

  // Partial densities follow the advected material scalars. A dense Au/He cell
  // with only a trace of CH must not acquire the CH shell's extra refinement.
  KOKKOS_INLINE_FUNCTION
  int TargetLevel(const Real density, const Real ch_density,
                  const Real gold_density) const {
    int target = density > density_threshold ? dense_level : 0;
    if (ch_density > density_threshold && ch_level > target) target = ch_level;
    if (gold_density > density_threshold && gold_level > target) target = gold_level;
    return target;
  }
};

#endif  // DCI_3D_DCI_AMR_HPP_
