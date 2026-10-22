//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file asymmetric_collision.cpp
//! \brief Inward-moving CH caps with an optional hot rear CH corona in helium.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"

namespace {
constexpr Real kKelvinPerEV = 11604.518121550082;
constexpr Real kPi = 3.14159265358979323846;
constexpr int kHistoryFields = 21;

void CollisionHistory(HistoryData *pdata, Mesh *pm);
void VacuumRadiationBoundary(Mesh *pm);

[[noreturn]] void ProblemError(const char *message) {
  std::cerr << "### FATAL ERROR in asymmetric_collision: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// Compact smoothing: exactly zero outside [-width,width], exactly one inside.
// A width of zero selects sharp, cell-centred geometry. The nominal interface is at 0.
KOKKOS_INLINE_FUNCTION
Real Inside(const Real distance, const Real width) {
  if (width == 0.0) return distance >= 0.0 ? 1.0 : 0.0;
  const Real s = fmin(fmax(0.5*(distance/width+1.0), 0.0), 1.0);
  return s*s*(3.0-2.0*s);
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  static_assert(kHistoryFields <= NHISTORY_VARIABLES, "Too many history fields");
  user_hist_func = CollisionHistory;
  user_bcs_func = VacuumRadiationBoundary;
  user_bcs = true;

  auto *pmbp = pmy_mesh_->pmb_pack;
  auto *ph = pmbp->phydro;
  // Default off preserves the initial state and scalar layout of archived decks.
  const bool corona_enabled = pin->GetOrAddBoolean("problem", "corona_enabled", false);
  if (ph == nullptr || pmbp->pmhd != nullptr || pmbp->plaser != nullptr) {
    ProblemError("requires <hydro>, with no <mhd> or <laser> block");
  }
  if (!pmy_mesh_->three_d) {
    ProblemError("requires a three-dimensional mesh");
  }
  if (ph->ptwo_temp == nullptr || ph->ptwo_temp->pradiation == nullptr ||
      !ph->ptwo_temp->pradiation->IsImplicit() || !ph->use_dual_energy) {
    ProblemError("requires two_temperature, dual_energy, and implicit thermal radiation");
  }
  if (ph->pmaterials == nullptr || !ph->pmaterials->UsesTabularEOS() ||
      ph->pmaterials->NumberOfMaterials() != 2 ||
      (ph->nuser_scalars != 4 && ph->nuser_scalars != 5) ||
      (corona_enabled && ph->nuser_scalars != 5) ||
      ph->pmaterials->ScalarIndex() != ph->nhydro) {
    ProblemError("requires tabular CH/He, scalar_index=0, and four user scalars "
                 "(five with corona_enabled=true)");
  }
  if (pin->GetString("materials", "material0_name") != "CH" ||
      pin->GetString("materials", "material1_name") != "He" ||
      ph->ptwo_temp->pradiation->ngroups != 20 ||
      pin->GetString("thermal_radiation", "opacity_model") != "mixed-table") {
    ProblemError("requires CH then He and their matching 20-group opacity tables");
  }
  for (int face = 0; face < 6; ++face) {
    if (pmy_mesh_->mesh_bcs[face] != BoundaryFlag::outflow) {
      ProblemError("all fluid boundaries must be outflow");
    }
    const std::string boundary = "implicit_x" + std::to_string(face/2+1) +
        ((face%2 == 0) ? "_inner_boundary" : "_outer_boundary");
    if (pin->GetString("thermal_radiation", boundary) != "vacuum") {
      ProblemError("all implicit radiation boundaries must be vacuum");
    }
  }
  // Restart state already contains material tracers, thermal energies, and radiation.
  if (restart) return;

  // Problem parameters use physical units so changing code units cannot change geometry.
  const Real length = 1.0e-4/pmbp->punit->length_cgs();  // micrometre -> code length
  const Real velocity = 1.0e5/pmbp->punit->velocity_cgs();  // km/s -> code speed
  const Real rho_unit = pmbp->punit->density_cgs();
  const Real upper_radius = pin->GetReal("problem", "upper_radius_um")*length;
  const Real lower_radius = pin->GetReal("problem", "lower_radius_um")*length;
  const Real thickness = pin->GetReal("problem", "shell_thickness_um")*length;
  const Real upper_speed = pin->GetReal("problem", "upper_speed_kms")*velocity;
  const Real lower_speed = pin->GetReal("problem", "lower_speed_kms")*velocity;
  const Real shell_density = pin->GetReal("problem", "shell_density_gcc")/rho_unit;
  const Real ambient_density = pin->GetReal("problem", "ambient_density_gcc")/rho_unit;
  const Real shell_temperature = pin->GetReal("problem", "shell_temperature_ev");
  const Real ambient_temperature = pin->GetReal("problem", "ambient_temperature_ev");
  const Real corona_density = pin->GetOrAddReal(
      "problem", "corona_density_gcc", 0.01)/rho_unit;
  const Real corona_temperature = pin->GetOrAddReal(
      "problem", "corona_temperature_ev", 1000.0);
  const Real angle = pin->GetReal("problem", "opening_half_angle_deg");
  const Real radial_width = pin->GetReal("problem", "interface_half_width_um")*length;
  const Real angular_width = pin->GetReal("problem", "angular_half_width");
  const Real ratio = pin->GetReal("hydro", "initial_electron_temperature_ratio");
  for (const Real value : {upper_radius, lower_radius, thickness, upper_speed,
       lower_speed, shell_density, ambient_density, shell_temperature,
       ambient_temperature, corona_density, corona_temperature, angle,
       radial_width, angular_width, ratio}) {
    if (!std::isfinite(value)) ProblemError("non-finite initial condition parameter");
  }
  const Real cosine = std::cos(angle*kPi/180.0);
  if (!(thickness > 0.0) || radial_width < 0.0 || radial_width >= 0.5*thickness ||
      upper_radius <= 0.5*thickness+radial_width ||
      lower_radius <= 0.5*thickness+radial_width ||
      !(upper_speed > 0.0) || !(lower_speed > 0.0) ||
      !(ambient_density > 0.0) || !(shell_density > ambient_density) ||
      !(shell_temperature > 0.0) || !(ambient_temperature > 0.0) ||
      !(angle > 0.0 && angle < 90.0) || angular_width < 0.0 ||
      angular_width >= cosine || !(ratio > 0.0) ||
      (corona_enabled && (!(corona_density > ambient_density) ||
                          !(corona_density < shell_density) ||
                          !(corona_temperature > shell_temperature)))) {
    ProblemError("invalid shell radii, thickness, speeds, density, angle, or temperature");
  }
  const Real support = std::fmax(upper_radius, lower_radius)+0.5*thickness+radial_width;
  for (int axis = 1; axis <= 3; ++axis) {
    const std::string name = "x"+std::to_string(axis);
    if (pin->GetReal("mesh", name+"min") > -support ||
        pin->GetReal("mesh", name+"max") < support) {
      ProblemError("the domain must contain both complete caps about the origin");
    }
  }
  if (global_variable::my_rank == 0) {
    std::cout << "# Collision caps: upper +x1, lower -x1; radial velocities toward origin\n"
              << "# Ballistic midpoint arrival [ns]: upper="
              << upper_radius/upper_speed*pmbp->punit->time_cgs()*1.0e9
              << ", lower=" << lower_radius/lower_speed*pmbp->punit->time_cgs()*1.0e9
              << "; no laser source" << std::endl;
    if (corona_enabled) {
      std::cout << "# Rear CH corona: stationary, rho=" << corona_density*rho_unit
                << " g/cm^3, Ti=" << corona_temperature << " eV, Te="
                << ratio*corona_temperature << " eV; inside cap angles, "
                << "outside shell outer radii, extending to box boundaries" << std::endl;
    }
  }

  const auto mixture = ph->pmaterials->DeviceData();
  const Real tshell = mixture.CodeTemperatureFromKelvin(shell_temperature*kKelvinPerEV);
  const Real tambient = mixture.CodeTemperatureFromKelvin(ambient_temperature*kKelvinPerEV);
  const Real tcorona = mixture.CodeTemperatureFromKelvin(corona_temperature*kKelvinPerEV);
  const int scalar = ph->pmaterials->ScalarIndex();
  const bool corona_tracer = ph->nuser_scalars == 5;
  auto w0 = ph->w0;
  auto size = pmbp->pmb->mb_size;
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  Kokkos::deep_copy(w0, 0.0);
  par_for("pgen_asymmetric_collision", DevExeSpace(), 0, pmbp->nmb_thispack-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                              size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                              size.d_view(m).x3max);
    const Real r = sqrt(x*x+y*y+z*z);
    const Real mu = (r > 0.0) ? x/r : 0.0;
    const Real upper_angle = Inside(mu-cosine, angular_width);
    const Real lower_angle = Inside(-mu-cosine, angular_width);
    const Real upper_outer = Inside(upper_radius+0.5*thickness-r, radial_width);
    const Real lower_outer = Inside(lower_radius+0.5*thickness-r, radial_width);
    const Real upper = Inside(r-upper_radius+0.5*thickness, radial_width)*
        upper_outer*upper_angle;
    const Real lower = Inside(r-lower_radius+0.5*thickness, radial_width)*
        lower_outer*lower_angle;
    // Back means radially outward, opposite the inward shell velocity. Complement
    // the same outer-interface gate to avoid a gap or double-counted volume.
    const Real corona = corona_enabled ?
        upper_angle*(1.0-upper_outer)+lower_angle*(1.0-lower_outer) : 0.0;
    const Real rho_upper = upper*shell_density;
    const Real rho_lower = lower*shell_density;
    const Real rho_corona = corona*corona_density;
    const Real rho_he = fmax(1.0-upper-lower-corona, 0.0)*ambient_density;
    const Real rho = rho_upper+rho_lower+rho_corona+rho_he;
    Real fractions[2] = {(rho_upper+rho_lower+rho_corona)/rho, rho_he/rho};
    const auto composition = mixture.CompositionFromFractions(fractions);
    const Real ti = corona_enabled ?
        ((rho_upper+rho_lower)*tshell+rho_corona*tcorona+rho_he*tambient)/rho :
        fractions[0]*tshell+fractions[1]*tambient;
    const auto state = mixture.StateFromRhoTemperatures(rho, ti, ratio*ti, composition);
    // Only dense shells carry initial momentum; corona CH and He are stationary.
    const Real inward = (r > 0.0) ?
        -(rho_upper*upper_speed+rho_lower*lower_speed)/(rho*r) : 0.0;
    w0(m, IDN, k, j, i) = rho;
    w0(m, IVX, k, j, i) = inward*x;
    w0(m, IVY, k, j, i) = inward*y;
    w0(m, IVZ, k, j, i) = inward*z;
    w0(m, IEN, k, j, i) = rho*(state.ion_specific_internal_energy+
                                    state.electron_specific_internal_energy);
    w0(m, scalar, k, j, i) = fractions[0];
    w0(m, scalar+1, k, j, i) = fractions[1];
    // Origin tags are passive scalars, not separate copies of CH in the EOS mixture.
    w0(m, scalar+2, k, j, i) = rho_upper/rho;
    w0(m, scalar+3, k, j, i) = rho_lower/rho;
    if (corona_tracer) w0(m, scalar+4, k, j, i) = rho_corona/rho;
  });
  // PrimToCons adds the kinetic energy; Driver initializes separate Ti/Te from the EOS.
  ph->peos->PrimToCons(w0, ph->u0, is, ie, js, je, ks, ke);
}

namespace {
void VacuumRadiationBoundary(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *ph = pmbp->phydro;
  auto *prad = ph->ptwo_temp->pradiation;
  auto u0 = ph->u0;
  auto bcs = pmbp->pmb->mb_bcs;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng, n1 = indcs.nx1+2*ng, n2 = indcs.nx2+2*ng;
  const int n3 = indcs.nx3+2*ng, nmb1 = pmbp->nmb_thispack-1;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke, first = prad->ifirst, groups = prad->ngroups;
  par_for("collision_vacuum_x1", DevExeSpace(), 0, nmb1, 0, groups-1, 0, n3-1, 0, n2-1,
  KOKKOS_LAMBDA(int m, int g, int k, int j) {
    for (int n = 0; n < ng; ++n) {
      if (bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::outflow)
        u0(m, first+g, k, j, is-n-1) = 0.0;
      if (bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow)
        u0(m, first+g, k, j, ie+n+1) = 0.0;
    }
  });
  par_for("collision_vacuum_x2", DevExeSpace(), 0, nmb1, 0, groups-1, 0, n3-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int g, int k, int i) {
    for (int n = 0; n < ng; ++n) {
      if (bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow)
        u0(m, first+g, k, js-n-1, i) = 0.0;
      if (bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow)
        u0(m, first+g, k, je+n+1, i) = 0.0;
    }
  });
  par_for("collision_vacuum_x3", DevExeSpace(), 0, nmb1, 0, groups-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int g, int j, int i) {
    for (int n = 0; n < ng; ++n) {
      if (bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::outflow)
        u0(m, first+g, ks-n-1, j, i) = 0.0;
      if (bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::outflow)
        u0(m, first+g, ke+n+1, j, i) = 0.0;
    }
  });
}

void CollisionHistory(HistoryData *pdata, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *ph = pmbp->phydro;
  auto *ptwo = ph->ptwo_temp;
  auto *prad = ptwo->pradiation;
  const char *labels[kHistoryFields] = {
    "mass", "CH_mass", "He_mass", "upper_mass", "lower_mass", "mom1", "mom2", "mom3",
    "mat_E", "kin_E", "eion_E", "eele_E", "erad_E", "total_E", "upper_x1",
    "lower_x1", "rad_Pesc", "eos_bad", "eos_floor", "tracer_error", "coro_mass"};
  pdata->nhist = kHistoryFields;
  for (int n = 0; n < kHistoryFields; ++n) pdata->label[n] = labels[n];
  auto u0 = ph->u0;
  auto thermo = ptwo->thermodynamics;
  auto size = pmbp->pmb->mb_size;
  const int scalar = ph->pmaterials->ScalarIndex(), iion = ptwo->iion, iele = ptwo->iele;
  const bool corona_tracer = ph->nuser_scalars == 5;
  const int first = prad->ifirst, groups = prad->ngroups;
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nji = nx2*nx1, nkji = nx3*nji;
  constexpr int bad_flags = materials::ionmix_density_above_table |
      materials::ionmix_temperature_below_table | materials::ionmix_temperature_above_table |
      materials::ionmix_energy_above_table;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("collision_history",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, pmbp->nmb_thispack*nkji),
  KOKKOS_LAMBDA(int idx, array_sum::GlobalSum &total) {
    const int m = idx/nkji, k = (idx%nkji)/nji+ks;
    const int j = (idx%nji)/nx1+js, i = idx%nx1+is;
    const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real rho = u0(m, IDN, k, j, i);
    const Real kinetic = 0.5*(SQR(u0(m, IM1, k, j, i))+SQR(u0(m, IM2, k, j, i))+
                              SQR(u0(m, IM3, k, j, i)))/rho;
    Real erad = 0.0;
    for (int g = 0; g < groups; ++g) erad += u0(m, first+g, k, j, i);
    array_sum::GlobalSum cell;
    cell.the_array[0] = vol*rho;
    for (int n = 0; n < 4; ++n) cell.the_array[1+n] = vol*u0(m, scalar+n, k, j, i);
    for (int n = 0; n < 3; ++n) cell.the_array[5+n] = vol*u0(m, IM1+n, k, j, i);
    cell.the_array[8] = vol*u0(m, IEN, k, j, i);
    cell.the_array[9] = vol*kinetic;
    cell.the_array[10] = vol*u0(m, iion, k, j, i);
    cell.the_array[11] = vol*u0(m, iele, k, j, i);
    cell.the_array[12] = vol*erad;
    cell.the_array[13] = cell.the_array[8]+cell.the_array[12];
    cell.the_array[14] = x*cell.the_array[3];
    cell.the_array[15] = x*cell.the_array[4];
    const int flags = static_cast<int>(thermo(
        m, two_temperature::TwoTemperature::eos_query_flags, k, j, i));
    cell.the_array[17] = (flags & bad_flags) != 0 ? 1.0 : 0.0;
    cell.the_array[18] = (flags & materials::ionmix_energy_below_table) != 0 ? 1.0 : 0.0;
    cell.the_array[20] = corona_tracer ? vol*u0(m, scalar+4, k, j, i) : 0.0;
    cell.the_array[19] = fabs(cell.the_array[1]-cell.the_array[3]-cell.the_array[4]-
                            cell.the_array[20]);
    total += cell;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  Kokkos::fence();
  // Implicit transport exposes the rank-local surface flux; history does the MPI sum.
  sum.the_array[16] = prad->implicit_boundary_power;
  for (int n = 0; n < kHistoryFields; ++n) pdata->hdata[n] = sum.the_array[n];
}
}  // namespace
