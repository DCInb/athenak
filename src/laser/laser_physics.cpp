//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_physics.cpp
//! \brief Host-side entry points for laser optical physics.

#include "laser/laser_physics.hpp"

namespace laser {

// Keep a translation unit for future tabulated collision/ionization models while the
// initial physical kernels remain header-defined and device callable.
static_assert(electron_charge_cgs > 0.0, "Laser physical constants must be positive");

} // namespace laser
