//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file implicit_conduction.cpp
//! \brief Self-checking regression problem for implicit two-temperature conduction.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "diffusion/conduction.hpp"
#include "pgen/pgen.hpp"
#include "two_temperature/two_temperature.hpp"

namespace {

constexpr int kNumberOfModes = 3;
constexpr Real kPi = 3.141592653589793238462643383279502884;
constexpr int kPeriodicBoundary = 0;
constexpr int kDirichletBoundary = 1;

struct ImplicitConductionTestParameters {
  Real background;
  Real amplitude[kNumberOfModes];
  int mode[kNumberOfModes];
  Real x1min;
  Real x1max;
  Real theta;
  int boundary_kind;
};

ImplicitConductionTestParameters test_parameters;

[[noreturn]] void RegressionError(const std::string &message) {
  if (global_variable::my_rank == 0) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << std::endl
              << message << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

KOKKOS_INLINE_FUNCTION
Real InitialTemperature(const ImplicitConductionTestParameters &parameters,
                        const Real x1) {
  const Real phase_factor = (parameters.boundary_kind == kPeriodicBoundary)
      ? 2.0*kPi : kPi;
  const Real phase = phase_factor*(x1-parameters.x1min)/
                     (parameters.x1max-parameters.x1min);
  Real temperature = parameters.background;
  for (int n = 0; n < kNumberOfModes; ++n) {
    const Real basis = (parameters.boundary_kind == kPeriodicBoundary)
        ? cos(parameters.mode[n]*phase) : sin(parameters.mode[n]*phase);
    temperature += parameters.amplitude[n]*basis;
  }
  return temperature;
}

KOKKOS_INLINE_FUNCTION
Real ExactTemperature(const ImplicitConductionTestParameters &parameters,
                      const Real x1, const Real dt, const Real diffusivity,
                      const Real dx1, const int nx1) {
  const Real phase_factor = (parameters.boundary_kind == kPeriodicBoundary)
      ? 2.0*kPi : kPi;
  const Real phase = phase_factor*(x1-parameters.x1min)/
                     (parameters.x1max-parameters.x1min);
  Real temperature = parameters.background;
  for (int n = 0; n < kNumberOfModes; ++n) {
    const Real eigen_angle = (parameters.boundary_kind == kPeriodicBoundary)
        ? kPi*parameters.mode[n]/static_cast<Real>(nx1)
        : 0.5*kPi*parameters.mode[n]/static_cast<Real>(nx1);
    const Real sine = sin(eigen_angle);
    const Real eigenvalue = 4.0*sine*sine/(dx1*dx1);
    const Real numerator = 1.0-(1.0-parameters.theta)*dt*diffusivity*eigenvalue;
    const Real denominator = 1.0+parameters.theta*dt*diffusivity*eigenvalue;
    const Real basis = (parameters.boundary_kind == kPeriodicBoundary)
        ? cos(parameters.mode[n]*phase) : sin(parameters.mode[n]*phase);
    temperature += parameters.amplitude[n]*(numerator/denominator)*
                   basis;
  }
  return temperature;
}

void CheckImplicitConduction(ParameterInput *pin, Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! Initialize a constant-density, common-temperature state containing three Fourier modes.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  pgen_final_func = CheckImplicitConduction;
  if (restart) {
    RegressionError("The implicit-conduction regression does not support restarts");
  }

  auto *pmbp = pmy_mesh_->pmb_pack;
  auto *phydro = pmbp->phydro;
  if (phydro == nullptr || phydro->ptwo_temp == nullptr || phydro->pcond == nullptr ||
      !phydro->pcond->IsImplicit()) {
    RegressionError("The regression requires implicit two-temperature Hydro conduction");
  }
  if (!pmy_mesh_->one_d || pmy_mesh_->multilevel) {
    RegressionError("The regression requires a one-dimensional, uniform mesh");
  }

  test_parameters.background =
      pin->GetOrAddReal("problem", "temperature_background", 1.0);
  test_parameters.amplitude[0] =
      pin->GetOrAddReal("problem", "temperature_amplitude_1", 0.15);
  test_parameters.amplitude[1] =
      pin->GetOrAddReal("problem", "temperature_amplitude_2", 0.10);
  test_parameters.amplitude[2] =
      pin->GetOrAddReal("problem", "temperature_amplitude_3", 0.05);
  test_parameters.mode[0] = pin->GetOrAddInteger("problem", "mode_1", 1);
  test_parameters.mode[1] = pin->GetOrAddInteger("problem", "mode_2", 7);
  test_parameters.mode[2] = pin->GetOrAddInteger("problem", "mode_3", 13);
  test_parameters.x1min = pin->GetReal("mesh", "x1min");
  test_parameters.x1max = pin->GetReal("mesh", "x1max");
  test_parameters.theta = pin->GetReal("hydro", "conduction_theta");
  const std::string boundary = pin->GetOrAddString("problem", "boundary", "periodic");
  if (boundary == "periodic") {
    test_parameters.boundary_kind = kPeriodicBoundary;
    if (!pmy_mesh_->strictly_periodic) {
      RegressionError("The periodic regression requires periodic mesh boundaries");
    }
  } else if (boundary == "dirichlet") {
    test_parameters.boundary_kind = kDirichletBoundary;
    if (pmy_mesh_->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic ||
        pmy_mesh_->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic) {
      RegressionError("The Dirichlet regression requires physical x1 mesh boundaries");
    }
  } else {
    RegressionError("<problem>/boundary must be 'periodic' or 'dirichlet'");
  }

  Real minimum_temperature = test_parameters.background;
  for (int n = 0; n < kNumberOfModes; ++n) {
    minimum_temperature -= std::abs(test_parameters.amplitude[n]);
    const int global_nx1 = pmy_mesh_->mb_indcs.nx1*pmy_mesh_->nmb_rootx1;
    const bool above_nyquist = (test_parameters.boundary_kind == kPeriodicBoundary)
        ? 2*test_parameters.mode[n] >= global_nx1
        : test_parameters.mode[n] >= global_nx1;
    if (test_parameters.mode[n] <= 0 || above_nyquist) {
      RegressionError("Regression Fourier modes must lie below the global Nyquist mode");
    }
  }
  if (!(minimum_temperature > 0.0) ||
      !(test_parameters.x1max > test_parameters.x1min)) {
    RegressionError("Regression temperatures and x1 domain are invalid");
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack-1;
  const Real gamma_minus_one = phydro->peos->eos_data.gamma-1.0;
  auto size = pmbp->pmb->mb_size;
  auto w0 = phydro->w0;
  const auto parameters = test_parameters;

  Kokkos::deep_copy(w0, 0.0);
  par_for("pgen_implicit_conduction", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real temperature = InitialTemperature(parameters, x1);
    w0(m, IDN, k, j, i) = 1.0;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = temperature/gamma_minus_one;
  });
  phydro->peos->PrimToCons(w0, phydro->u0, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! Compare the result with the exact centered-space theta-method update and fail on error.

void CheckImplicitConduction(ParameterInput *pin, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  auto *ptwo = phydro->ptwo_temp;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int global_nx1 = indcs.nx1*pm->nmb_rootx1;
  const Real dt = pm->dt_last_completed;
  const Real gamma_minus_one = phydro->peos->eos_data.gamma-1.0;
  const Real electron_fraction = ptwo->ElectronHeatCapacityFraction();
  const Real electron_capacity = electron_fraction/gamma_minus_one;
  const Real ion_capacity = (1.0-electron_fraction)/gamma_minus_one;
  const Real diffusivity = phydro->pcond->alpha_iso/electron_capacity;
  const auto parameters = test_parameters;
  auto size = pmbp->pmb->mb_size;
  auto temperature = ptwo->temperature;
  auto u = phydro->u0;
  const int iele = ptwo->iele;
  const int iion = ptwo->iion;

  if (pm->ncycle != 1 || !(dt > 0.0)) {
    RegressionError("The implicit-conduction regression must complete exactly one step");
  }

  Real l1_error = 0.0;
  Kokkos::parallel_reduce(
      "implicit_conduction_l1",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmbp->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real exact = ExactTemperature(parameters, x1, dt, diffusivity,
                                        size.d_view(m).dx1, global_nx1);
    sum += fabs(temperature(m, 1, k, j, i)-exact);
  }, l1_error);

  Real linf_error = 0.0;
  Kokkos::parallel_reduce(
      "implicit_conduction_linf",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmbp->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real exact = ExactTemperature(parameters, x1, dt, diffusivity,
                                        size.d_view(m).dx1, global_nx1);
    maximum = fmax(maximum, fabs(temperature(m, 1, k, j, i)-exact));
  }, Kokkos::Max<Real>(linf_error));

  Real electron_energy = 0.0;
  Kokkos::parallel_reduce(
      "implicit_conduction_electron_energy",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmbp->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum) {
    const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    sum += volume*u(m, iele, k, j, i);
  }, electron_energy);

  Real expected_electron_energy = 0.0;
  Kokkos::parallel_reduce(
      "implicit_conduction_expected_electron_energy",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmbp->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    const Real exact = ExactTemperature(parameters, x1, dt, diffusivity,
                                        size.d_view(m).dx1, global_nx1);
    sum += volume*electron_capacity*exact;
  }, expected_electron_energy);

  Real total_energy = 0.0;
  Kokkos::parallel_reduce(
      "implicit_conduction_total_energy",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmbp->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum) {
    const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    sum += volume*u(m, IEN, k, j, i);
  }, total_energy);

  Real expected_total_energy = 0.0;
  Kokkos::parallel_reduce(
      "implicit_conduction_expected_total_energy",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmbp->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    const Real initial = InitialTemperature(parameters, x1);
    const Real exact = ExactTemperature(parameters, x1, dt, diffusivity,
                                        size.d_view(m).dx1, global_nx1);
    sum += volume*(ion_capacity*initial+electron_capacity*exact);
  }, expected_total_energy);

  Real ion_linf_error = 0.0;
  Kokkos::parallel_reduce(
      "implicit_conduction_ion_linf",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmbp->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real initial = ion_capacity*InitialTemperature(parameters, x1);
    maximum = fmax(maximum, fabs(u(m, iion, k, j, i)-initial));
  }, Kokkos::Max<Real>(ion_linf_error));

#if MPI_PARALLEL_ENABLED
  Real sums[5] = {l1_error, electron_energy, expected_electron_energy,
                  total_energy, expected_total_energy};
  MPI_Allreduce(MPI_IN_PLACE, sums, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  l1_error = sums[0];
  electron_energy = sums[1];
  expected_electron_energy = sums[2];
  total_energy = sums[3];
  expected_total_energy = sums[4];
  Real maxima[2] = {linf_error, ion_linf_error};
  MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  linf_error = maxima[0];
  ion_linf_error = maxima[1];
#endif

  const Real number_of_cells = static_cast<Real>(
      pm->nmb_total*indcs.nx1*indcs.nx2*indcs.nx3);
  l1_error /= number_of_cells;
  const Real electron_energy_error =
      std::abs(electron_energy-expected_electron_energy)/
      std::abs(expected_electron_energy);
  const Real total_energy_error =
      std::abs(total_energy-expected_total_energy)/std::abs(expected_total_energy);
  const Real explicit_limit = size.h_view(0).dx1*size.h_view(0).dx1/
                              (2.0*diffusivity);
  const Real stiffness_ratio = dt/explicit_limit;

  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(17)
              << "# implicit conduction regression: boundary="
              << ((parameters.boundary_kind == kPeriodicBoundary)
                  ? "periodic" : "dirichlet")
              << " dt=" << dt
              << " explicit_limit=" << explicit_limit
              << " stiffness_ratio=" << stiffness_ratio
              << " l1_error=" << l1_error
              << " linf_error=" << linf_error
              << " electron_energy_error=" << electron_energy_error
              << " total_energy_error=" << total_energy_error
              << " ion_linf_error=" << ion_linf_error
              << " nonlinear_iterations="
              << phydro->pcond->nonlinear_iterations_last_solve
              << " pcg_iterations=" << phydro->pcond->iterations_last_solve
              << std::endl;
  }

  const Real error_tolerance = pin->GetOrAddReal(
      "problem", "error_tolerance", 2.0e-10);
  const Real conservation_tolerance = pin->GetOrAddReal(
      "problem", "conservation_tolerance", 2.0e-12);
  if (!std::isfinite(l1_error) || !std::isfinite(linf_error) ||
      !std::isfinite(electron_energy_error) || !std::isfinite(total_energy_error) ||
      stiffness_ratio <= 10.0 || l1_error > error_tolerance ||
      linf_error > error_tolerance || ion_linf_error > error_tolerance ||
      electron_energy_error > conservation_tolerance ||
      total_energy_error > conservation_tolerance ||
      phydro->pcond->iterations_last_solve <= 0) {
    std::ostringstream message;
    message << "Implicit-conduction regression failed; see reported error metrics";
    RegressionError(message.str());
  }
}

} // namespace
