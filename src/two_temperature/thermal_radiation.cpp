//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation.cpp
//! \brief Explicit/implicit multigroup FLD and electron-radiation energy exchange.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "diffusion/native_diffusion_solver.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/nghbr_index.hpp"
#include "parameter_input.hpp"
#include "two_temperature/opacity_table.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"
#include "two_temperature/thermal_radiation_halo.hpp"
#include "two_temperature/thermal_radiation_hypre.hpp"
#include "two_temperature/thermal_radiation_planck.hpp"
#include "two_temperature/thermal_radiation_scratch.hpp"

namespace two_temperature {
void ObserveImplicitAMRIterations(ImplicitAMRAggregateState *, int, bool);
void PrepareImplicitAMRReuseGroup(ImplicitAMRAggregateState *, int, int, bool);
bool ImplicitAMRHierarchyReused(ImplicitAMRAggregateState *);

namespace {

using detail::PlanckGroupFraction;
using detail::PlanckSpectrum;

#if SINGLE_PRECISION_ENABLED
constexpr Real kRealEpsilon = FLT_EPSILON;
#else
constexpr Real kRealEpsilon = DBL_EPSILON;
#endif

[[noreturn]] void ImplicitRadiationError(const std::string &message) {
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
bool IsPhysicalBoundary(const BoundaryFlag flag) {
  return flag != BoundaryFlag::block && flag != BoundaryFlag::periodic &&
         flag != BoundaryFlag::shear_periodic && flag != BoundaryFlag::undef;
}

KOKKOS_INLINE_FUNCTION
Real ImplicitFrozenFaceCoefficient(const Real center, const Real neighbor,
                                   const bool vacuum_face,
                                   const Real vacuum_face_cap) {
  const Real arithmetic_face = 0.5*(center+neighbor);
  return vacuum_face ? fmin(arithmetic_face, vacuum_face_cap) : arithmetic_face;
}

// Contribution of a face conductance to the diagonal of the homogeneous operator.
// Internal and periodic ghosts are independent neighboring unknowns.  At a physical
// face, zero-gradient ghosts equal the boundary cell, fixed-value ghosts equal its
// negative, and vacuum ghosts vanish.
KOKKOS_INLINE_FUNCTION
Real ImplicitFaceDiagonalWeight(const bool physical_face, const int boundary_type) {
  if (!physical_face) return 1.0;
  if (boundary_type == 0) return 0.0;
  if (boundary_type == 1) return 2.0;
  return 1.0;
}

// Sum the fine-grid conductances crossing one face of a piecewise-constant aggregate.
// This is the corresponding off-diagonal magnitude of the Galerkin matrix P^T A P.
// direction is 0/1/2 for x1/x2/x3 and side is -1/+1.  The aggregate extents are in
// fine cells; inactive dimensions therefore have extent one.
KOKKOS_INLINE_FUNCTION
Real ImplicitAggregateFaceConductance(
    const DvceArray5D<Real> coefficient, const int m, const int direction,
    const int side, const int ci, const int cj, const int ck,
    const int aggregate1, const int aggregate2, const int aggregate3,
    const int is, const int js, const int ks, const Real dt, const Real dx,
    const bool vacuum_face, const Real vacuum_face_cap) {
  Real sum = 0.0;
  if (direction == 0) {
    const int i = is+ci*aggregate1+(side > 0 ? aggregate1-1 : 0);
    for (int ok = 0; ok < aggregate3; ++ok) {
      const int k = ks+ck*aggregate3+ok;
      for (int oj = 0; oj < aggregate2; ++oj) {
        const int j = js+cj*aggregate2+oj;
        sum += ImplicitFrozenFaceCoefficient(
            coefficient(m, 0, k, j, i), coefficient(m, 0, k, j, i+side),
            vacuum_face, vacuum_face_cap);
      }
    }
  } else if (direction == 1) {
    const int j = js+cj*aggregate2+(side > 0 ? aggregate2-1 : 0);
    for (int ok = 0; ok < aggregate3; ++ok) {
      const int k = ks+ck*aggregate3+ok;
      for (int oi = 0; oi < aggregate1; ++oi) {
        const int i = is+ci*aggregate1+oi;
        sum += ImplicitFrozenFaceCoefficient(
            coefficient(m, 0, k, j, i), coefficient(m, 0, k, j+side, i),
            vacuum_face, vacuum_face_cap);
      }
    }
  } else {
    const int k = ks+ck*aggregate3+(side > 0 ? aggregate3-1 : 0);
    for (int oj = 0; oj < aggregate2; ++oj) {
      const int j = js+cj*aggregate2+oj;
      for (int oi = 0; oi < aggregate1; ++oi) {
        const int i = is+ci*aggregate1+oi;
        sum += ImplicitFrozenFaceCoefficient(
            coefficient(m, 0, k, j, i), coefficient(m, 0, k+side, j, i),
            vacuum_face, vacuum_face_cap);
      }
    }
  }
  return dt*sum/(dx*dx);
}

// Sum -A_ij*x_j over the neighbors that remain inside one MeshBlock aggregate grid.
// Global multilevel callers add the cross-MeshBlock face terms after exchanging the
// corresponding aggregate field.
KOKKOS_INLINE_FUNCTION
Real ImplicitAggregateInternalNeighborSum(
    const DvceArray3D<Real> conductance, const DvceArray5D<Real> solution,
    const int m, const int packed, const int ci, const int cj, const int ck,
    const int cnx1, const int cnx2, const int cnx3,
    const int fine_nx1, const int fine_nx2, const int is, const int js, const int ks,
    const bool multi_d, const bool three_d) {
  Real sum = 0.0;
  if (ci > 0) {
    sum += conductance(m, 0, packed)*ImplicitPackedScratchValue(
        solution, m, packed-1, fine_nx1, fine_nx2, is, js, ks);
  }
  if (ci < cnx1-1) {
    sum += conductance(m, 1, packed)*ImplicitPackedScratchValue(
        solution, m, packed+1, fine_nx1, fine_nx2, is, js, ks);
  }
  if (multi_d) {
    if (cj > 0) {
      sum += conductance(m, 2, packed)*ImplicitPackedScratchValue(
          solution, m, packed-cnx1, fine_nx1, fine_nx2, is, js, ks);
    }
    if (cj < cnx2-1) {
      sum += conductance(m, 3, packed)*ImplicitPackedScratchValue(
          solution, m, packed+cnx1, fine_nx1, fine_nx2, is, js, ks);
    }
  }
  if (three_d) {
    const int plane = cnx1*cnx2;
    if (ck > 0) {
      sum += conductance(m, 4, packed)*ImplicitPackedScratchValue(
          solution, m, packed-plane, fine_nx1, fine_nx2, is, js, ks);
    }
    if (ck < cnx3-1) {
      sum += conductance(m, 5, packed)*ImplicitPackedScratchValue(
          solution, m, packed+plane, fine_nx1, fine_nx2, is, js, ks);
    }
  }
  return sum;
}

KOKKOS_INLINE_FUNCTION
Real ImplicitFineBlockInternalNeighborSum(
    const DvceArray5D<Real> coefficient, const DvceArray5D<Real> solution,
    const int m, const int k, const int j, const int i,
    const int is, const int ie, const int js, const int je, const int ks, const int ke,
    const Real dt, const Real dx1, const Real dx2, const Real dx3,
    const bool multi_d, const bool three_d) {
  const Real kc = coefficient(m, 0, k, j, i);
  Real sum = 0.0;
  const Real scale1 = dt/(dx1*dx1);
  if (i > is) {
    sum += scale1*0.5*(kc+coefficient(m, 0, k, j, i-1))*
        solution(m, 0, k, j, i-1);
  }
  if (i < ie) {
    sum += scale1*0.5*(kc+coefficient(m, 0, k, j, i+1))*
        solution(m, 0, k, j, i+1);
  }
  if (multi_d) {
    const Real scale2 = dt/(dx2*dx2);
    if (j > js) {
      sum += scale2*0.5*(kc+coefficient(m, 0, k, j-1, i))*
          solution(m, 0, k, j-1, i);
    }
    if (j < je) {
      sum += scale2*0.5*(kc+coefficient(m, 0, k, j+1, i))*
          solution(m, 0, k, j+1, i);
    }
  }
  if (three_d) {
    const Real scale3 = dt/(dx3*dx3);
    if (k > ks) {
      sum += scale3*0.5*(kc+coefficient(m, 0, k-1, j, i))*
          solution(m, 0, k-1, j, i);
    }
    if (k < ke) {
      sum += scale3*0.5*(kc+coefficient(m, 0, k+1, j, i))*
          solution(m, 0, k+1, j, i);
    }
  }
  return sum;
}

KOKKOS_INLINE_FUNCTION
Real ImplicitFineGlobalNeighborSum(
    const DvceArray5D<Real> coefficient, const DvceArray5D<Real> solution,
    const DvceArray3D<Real> remote_faces, const DvceArray2D<int> neighbor_gid,
    const DvceArray2D<int> neighbor_rank, const int m, const int gid,
    const int gid0, const int my_rank, const int k, const int j, const int i,
    const int is, const int ie, const int js, const int je, const int ks, const int ke,
    const int nx1, const int nx2, const Real dt, const Real dx1, const Real dx2,
    const Real dx3, const bool multi_d, const bool three_d) {
  Real sum = ImplicitFineBlockInternalNeighborSum(
      coefficient, solution, m, k, j, i, is, ie, js, je, ks, ke,
      dt, dx1, dx2, dx3, multi_d, three_d);
  const Real kc = coefficient(m, 0, k, j, i);
  if (i == is && neighbor_gid(gid, 0) >= 0) {
    const Real value = neighbor_rank(gid, 0) == my_rank ?
        solution(neighbor_gid(gid, 0)-gid0, 0, k, j, ie) :
        remote_faces(m, 0, (k-ks)*nx2+(j-js));
    sum += dt*0.5*(kc+coefficient(m, 0, k, j, i-1))*value/(dx1*dx1);
  }
  if (i == ie && neighbor_gid(gid, 1) >= 0) {
    const Real value = neighbor_rank(gid, 1) == my_rank ?
        solution(neighbor_gid(gid, 1)-gid0, 0, k, j, is) :
        remote_faces(m, 1, (k-ks)*nx2+(j-js));
    sum += dt*0.5*(kc+coefficient(m, 0, k, j, i+1))*value/(dx1*dx1);
  }
  if (multi_d && j == js && neighbor_gid(gid, 2) >= 0) {
    const Real value = neighbor_rank(gid, 2) == my_rank ?
        solution(neighbor_gid(gid, 2)-gid0, 0, k, je, i) :
        remote_faces(m, 2, (k-ks)*nx1+(i-is));
    sum += dt*0.5*(kc+coefficient(m, 0, k, j-1, i))*value/(dx2*dx2);
  }
  if (multi_d && j == je && neighbor_gid(gid, 3) >= 0) {
    const Real value = neighbor_rank(gid, 3) == my_rank ?
        solution(neighbor_gid(gid, 3)-gid0, 0, k, js, i) :
        remote_faces(m, 3, (k-ks)*nx1+(i-is));
    sum += dt*0.5*(kc+coefficient(m, 0, k, j+1, i))*value/(dx2*dx2);
  }
  if (three_d && k == ks && neighbor_gid(gid, 4) >= 0) {
    const Real value = neighbor_rank(gid, 4) == my_rank ?
        solution(neighbor_gid(gid, 4)-gid0, 0, ke, j, i) :
        remote_faces(m, 4, (j-js)*nx1+(i-is));
    sum += dt*0.5*(kc+coefficient(m, 0, k-1, j, i))*value/(dx3*dx3);
  }
  if (three_d && k == ke && neighbor_gid(gid, 5) >= 0) {
    const Real value = neighbor_rank(gid, 5) == my_rank ?
        solution(neighbor_gid(gid, 5)-gid0, 0, ks, j, i) :
        remote_faces(m, 5, (j-js)*nx1+(i-is));
    sum += dt*0.5*(kc+coefficient(m, 0, k+1, j, i))*value/(dx3*dx3);
  }
  return sum;
}

KOKKOS_INLINE_FUNCTION
Real ImplicitAggregateGlobalNeighborSum(
    const DvceArray3D<Real> conductance, const DvceArray5D<Real> solution,
    const DvceArray3D<Real> remote_faces, const DvceArray2D<int> neighbor_gid,
    const DvceArray2D<int> neighbor_rank, const int m, const int gid,
    const int gid0, const int my_rank, const int packed,
    const int level_offset, const int ci, const int cj, const int ck,
    const int cnx1, const int cnx2, const int cnx3,
    const int fine_nx1, const int fine_nx2, const int is, const int js, const int ks,
    const bool multi_d, const bool three_d) {
  Real sum = ImplicitAggregateInternalNeighborSum(
      conductance, solution, m, packed, ci, cj, ck, cnx1, cnx2, cnx3,
      fine_nx1, fine_nx2, is, js, ks, multi_d, three_d);
  if (ci == 0 && neighbor_gid(gid, 0) >= 0) {
    const int peer = level_offset+(ck*cnx2+cj)*cnx1+(cnx1-1);
    const Real value = neighbor_rank(gid, 0) == my_rank ?
        ImplicitPackedScratchValue(
            solution, neighbor_gid(gid, 0)-gid0, peer,
            fine_nx1, fine_nx2, is, js, ks) :
        remote_faces(m, 0, ck*cnx2+cj);
    sum += conductance(m, 0, packed)*value;
  }
  if (ci == cnx1-1 && neighbor_gid(gid, 1) >= 0) {
    const int peer = level_offset+(ck*cnx2+cj)*cnx1;
    const Real value = neighbor_rank(gid, 1) == my_rank ?
        ImplicitPackedScratchValue(
            solution, neighbor_gid(gid, 1)-gid0, peer,
            fine_nx1, fine_nx2, is, js, ks) :
        remote_faces(m, 1, ck*cnx2+cj);
    sum += conductance(m, 1, packed)*value;
  }
  if (multi_d && cj == 0 && neighbor_gid(gid, 2) >= 0) {
    const int peer = level_offset+(ck*cnx2+(cnx2-1))*cnx1+ci;
    const Real value = neighbor_rank(gid, 2) == my_rank ?
        ImplicitPackedScratchValue(
            solution, neighbor_gid(gid, 2)-gid0, peer,
            fine_nx1, fine_nx2, is, js, ks) :
        remote_faces(m, 2, ck*cnx1+ci);
    sum += conductance(m, 2, packed)*value;
  }
  if (multi_d && cj == cnx2-1 && neighbor_gid(gid, 3) >= 0) {
    const int peer = level_offset+(ck*cnx2)*cnx1+ci;
    const Real value = neighbor_rank(gid, 3) == my_rank ?
        ImplicitPackedScratchValue(
            solution, neighbor_gid(gid, 3)-gid0, peer,
            fine_nx1, fine_nx2, is, js, ks) :
        remote_faces(m, 3, ck*cnx1+ci);
    sum += conductance(m, 3, packed)*value;
  }
  if (three_d && ck == 0 && neighbor_gid(gid, 4) >= 0) {
    const int peer = level_offset+((cnx3-1)*cnx2+cj)*cnx1+ci;
    const Real value = neighbor_rank(gid, 4) == my_rank ?
        ImplicitPackedScratchValue(
            solution, neighbor_gid(gid, 4)-gid0, peer,
            fine_nx1, fine_nx2, is, js, ks) :
        remote_faces(m, 4, cj*cnx1+ci);
    sum += conductance(m, 4, packed)*value;
  }
  if (three_d && ck == cnx3-1 && neighbor_gid(gid, 5) >= 0) {
    const int peer = level_offset+cj*cnx1+ci;
    const Real value = neighbor_rank(gid, 5) == my_rank ?
        ImplicitPackedScratchValue(
            solution, neighbor_gid(gid, 5)-gid0, peer,
            fine_nx1, fine_nx2, is, js, ks) :
        remote_faces(m, 5, cj*cnx1+ci);
    sum += conductance(m, 5, packed)*value;
  }
  return sum;
}

// The face flux can be written as
//
//   F_n = -c_* D(E, |grad E|) grad_n(E).
//
// A timestep based on D itself is unnecessarily singular in the streaming limit:
// D -> alpha E/|grad E| even though the differential flux has a finite characteristic
// speed alpha*c_*.  These quantities are the two pieces of the face-flux Jacobian that
// enter a frozen-state explicit stability estimate.  ``normal_diffusivity`` multiplies
// a perturbation of the normal gradient and ``normal_speed`` multiplies a perturbation
// of the face-averaged energy.  Both are non-negative for every supported limiter.
struct FLDLinearization {
  Real diffusion_coefficient;
  Real normal_diffusivity;
  Real normal_speed;
  Real streaming_fraction;
};

struct CoupledSourceEvaluation {
  Real electron_energy = 0.0;
  Real radiation_energy = 0.0;
  Real radiation_delta = 0.0;
  Real residual = 0.0;
};

struct LaggedSourceResult {
  Real electron_energy = 0.0;
  Real radiation_energy = 0.0;
  Real electron_temperature = 0.0;
};

// Algebraically equivalent forms of the backward-Euler group elimination.  Dividing by
// the coupling depth in the stiff branch prevents an otherwise avoidable inf/inf when a
// source step is intentionally much longer than the microscopic exchange time.
KOKKOS_INLINE_FUNCTION
Real BackwardEulerGroupEnergy(const Real old_energy, const Real equilibrium,
                              const Real absorption_opacity,
                              const Real emission_opacity,
                              const Real coupling_depth) {
  if (!(absorption_opacity > 0.0)) {
    const Real emission = emission_opacity*equilibrium;
    return (emission > 0.0) ? old_energy+coupling_depth*emission : old_energy;
  }
  if (coupling_depth > 1.0) {
    const Real inverse_depth = 1.0/coupling_depth;
    return (inverse_depth*old_energy+emission_opacity*equilibrium)/
           (inverse_depth+absorption_opacity);
  }
  return (old_energy+coupling_depth*emission_opacity*equilibrium)/
         (1.0+coupling_depth*absorption_opacity);
}

KOKKOS_INLINE_FUNCTION
Real ElectronEnergyDensityFromTemperature(
    const Real density, const Real electron_temperature,
    const Real gamma_minus_one, const Real fixed_electron_fraction,
    const bool use_materials,
    const materials::MaterialMixtureDevice &mixture,
    const materials::MaterialComposition &composition) {
  if (use_materials) {
    return density*mixture.ElectronSpecificEnergyFromRhoTemperature(
        density, electron_temperature, composition);
  }
  return density*fixed_electron_fraction*electron_temperature/gamma_minus_one;
}

// Evaluate the scalar conservative residual after analytically eliminating all group
// energies.  The optional cache is used only once, after convergence; root iterations
// therefore need no per-group scratch storage.
KOKKOS_INLINE_FUNCTION
CoupledSourceEvaluation EvaluateCoupledSource(
    const Real electron_temperature, const Real old_electron_energy,
    const Real density, const Real coupling_depth, const Real arad,
    const Real gamma_minus_one, const Real fixed_electron_fraction,
    const int ngroups, const int first_group, const int m, const int k,
    const int j, const int i, const DvceArray5D<Real> &cons,
    const DvceArray5D<Real> &prim, const DvceArray1D<Real> &bounds,
    const DvceArray1D<Real> &constant_absorption,
    const DvceArray1D<Real> &constant_emission, const bool use_table,
    const OpacityTableDevice &opacity, const bool use_mixed_table,
    const MixedOpacityTableDevice &mixed_opacity, const bool use_materials,
    const materials::MaterialMixtureDevice &mixture,
    const materials::MaterialComposition &composition,
    const bool cache_groups) {
  CoupledSourceEvaluation result;
  result.electron_energy = ElectronEnergyDensityFromTemperature(
      density, electron_temperature, gamma_minus_one,
      fixed_electron_fraction, use_materials, mixture, composition);

  OpacityTableLocation opacity_location;
  if (use_table) {
    opacity_location = opacity.Locate(density, electron_temperature);
  }
  MixedOpacityTableLocation mixed_location;
  if (use_mixed_table) {
    mixed_location = mixed_opacity.Locate(
        density, electron_temperature, composition);
  }

  const Real temperature2 = electron_temperature*electron_temperature;
  const Real blackbody = arad*temperature2*temperature2;
  PlanckSpectrum planck(bounds(0), electron_temperature);
  for (int g = 0; g < ngroups; ++g) {
    const Real old = fmax(cons(m, first_group+g, k, j, i), 0.0);
    const Real kappaa = use_mixed_table ? mixed_opacity.Get(
        opacity_absorption, g, mixed_location) : (use_table ? opacity.Get(
        opacity_absorption, g, opacity_location) : constant_absorption(g));
    const Real kappae = use_mixed_table ? mixed_opacity.Get(
        opacity_emission, g, mixed_location) : (use_table ? opacity.Get(
        opacity_emission, g, opacity_location) : constant_emission(g));
    const Real fraction = planck.Fraction(bounds(g+1));
    const Real updated = BackwardEulerGroupEnergy(
        old, blackbody*fraction, kappaa, kappae, coupling_depth);
    if (cache_groups) prim(m, first_group+g, k, j, i) = updated;
    result.radiation_energy += updated;
    result.radiation_delta += updated-old;
  }
  // Uncoupled groups may dwarf the material energy. Subtract group changes before
  // summing, so a bright spectator cannot erase electrons from the energy invariant.
  result.residual = (result.electron_energy-old_electron_energy)+result.radiation_delta;
  return result;
}

// The compatibility source update and the nonlinear failure path share this bounded
// lagged substep implementation.  Radiation slots in prim are temporary energy-density
// scratch until each substep is committed as a specific energy.
KOKKOS_INLINE_FUNCTION
LaggedSourceResult ApplyLaggedSourceSubsteps(
    const int substeps, const Real dt, const Real density,
    const Real initial_electron_energy, const Real electron_energy_floor,
    const Real initial_electron_temperature, const Real chat, const Real arad,
    const Real gamma_minus_one, const Real fixed_electron_fraction,
    const int ngroups, const int first_group, const int m, const int k,
    const int j, const int i, const DvceArray5D<Real> &cons,
    const DvceArray5D<Real> &prim, const DvceArray1D<Real> &bounds,
    const DvceArray1D<Real> &constant_absorption,
    const DvceArray1D<Real> &constant_emission, const bool use_table,
    const OpacityTableDevice &opacity, const bool use_mixed_table,
    const MixedOpacityTableDevice &mixed_opacity, const bool use_materials,
    const materials::MaterialMixtureDevice &mixture,
    const materials::MaterialComposition &composition) {
  LaggedSourceResult result;
  result.electron_energy = initial_electron_energy;
  result.electron_temperature = initial_electron_temperature;
  const Real substep_dt = dt/static_cast<Real>(substeps);
  const Real coupling_depth = substep_dt*chat*density;

  for (int step = 0; step < substeps; ++step) {
    const Real tele = result.electron_temperature;
    OpacityTableLocation opacity_location;
    if (use_table) opacity_location = opacity.Locate(density, tele);
    MixedOpacityTableLocation mixed_location;
    if (use_mixed_table) {
      mixed_location = mixed_opacity.Locate(density, tele, composition);
    }
    const Real temperature2 = tele*tele;
    const Real blackbody = arad*temperature2*temperature2;
    Real positive = 0.0;
    Real negative = 0.0;
    PlanckSpectrum planck(bounds(0), tele);
    for (int g = 0; g < ngroups; ++g) {
      const Real old = fmax(cons(m, first_group+g, k, j, i), 0.0);
      const Real kappaa = use_mixed_table ? mixed_opacity.Get(
          opacity_absorption, g, mixed_location) : (use_table ? opacity.Get(
          opacity_absorption, g, opacity_location) : constant_absorption(g));
      const Real kappae = use_mixed_table ? mixed_opacity.Get(
          opacity_emission, g, mixed_location) : (use_table ? opacity.Get(
          opacity_emission, g, opacity_location) : constant_emission(g));
      const Real fraction = planck.Fraction(bounds(g+1));
      const Real updated = BackwardEulerGroupEnergy(
          old, blackbody*fraction, kappaa, kappae, coupling_depth);
      prim(m, first_group+g, k, j, i) = updated;
      const Real delta = updated-old;
      if (delta > 0.0) positive += delta;
      if (delta < 0.0) negative += delta;
    }

    // Absorbed energy is available during the same substep.  Scale only net-positive
    // group changes when unconstrained emission would cross the material floor.
    const Real available = fmax(
        result.electron_energy-electron_energy_floor-negative, 0.0);
    const Real emission_scale = (positive > available && positive > 0.0)
        ? available/positive : 1.0;
    Real total_delta = 0.0;
    result.radiation_energy = 0.0;
    for (int g = 0; g < ngroups; ++g) {
      const Real old = fmax(cons(m, first_group+g, k, j, i), 0.0);
      const Real raw = prim(m, first_group+g, k, j, i);
      Real delta = raw-old;
      if (delta > 0.0) delta *= emission_scale;
      const Real updated = old+delta;
      cons(m, first_group+g, k, j, i) = updated;
      prim(m, first_group+g, k, j, i) = updated/density;
      total_delta += delta;
      result.radiation_energy += updated;
    }
    result.electron_energy = fmax(
        result.electron_energy-total_delta, electron_energy_floor);
    if (use_materials) {
      result.electron_temperature = mixture.ElectronTemperature(
          density, result.electron_energy/density, composition);
    } else {
      result.electron_temperature = gamma_minus_one*result.electron_energy/
          (density*fixed_electron_fraction);
    }
  }
  return result;
}

// Preserve representable gradients in faint groups: squaring their components
// directly can underflow even when both the gradient and its FLD flux are finite.
KOKKOS_INLINE_FUNCTION
Real RadiationGradientMagnitude(Real x, Real y, Real z) {
  const Real scale = fmax(fabs(x), fmax(fabs(y), fabs(z)));
  if (scale == 0.0) return 0.0;
  x /= scale;
  y /= scale;
  z /= scale;
  return scale*sqrt(x*x+y*y+z*z);
}

// mode: 0=none, 1=FLASH harmonic, 2=FLASH Larsen, 3=FLASH min/max,
// 4=Levermore-Pomraning.  D has units of length and the physical diffusion coefficient
// multiplying grad(E) is c_hat*D.
KOKKOS_INLINE_FUNCTION
FLDLinearization FLDProperties(Real sigma, Real energy, Real grad,
                              Real normal_grad, Real alpha,
                              Real energy_floor, int mode) {
  sigma = fmax(sigma, 1.0e-30);
  Real effective_energy = fmax(energy, energy_floor);
  Real q = grad/(sigma*effective_energy*alpha);
  Real lambda;
  Real dlambda_dq;
  if (mode == 0) {
    lambda = ONE_3RD;
    dlambda_dq = 0.0;
  } else if (mode == 1) {
    Real denominator = 3.0 + q;
    lambda = 1.0/denominator;
    dlambda_dq = -1.0/(denominator*denominator);
  } else if (mode == 2) {
    Real denominator = 9.0 + q*q;
    lambda = 1.0/sqrt(denominator);
    dlambda_dq = -q/(denominator*sqrt(denominator));
  } else if (mode == 3) {
    if (q > 3.0) {
      lambda = 1.0/q;
      dlambda_dq = -1.0/(q*q);
    } else {
      lambda = ONE_3RD;
      dlambda_dq = 0.0;
    }
  } else {
    Real denominator = 6.0 + 3.0*q + q*q;
    lambda = (2.0 + q)/denominator;
    dlambda_dq = -(q*q + 4.0*q)/(denominator*denominator);
  }

  FLDLinearization result;
  result.diffusion_coefficient = lambda/sigma;

  Real normal_fraction = (grad > 0.0) ? normal_grad/grad : 0.0;
  Real normal_fraction_sq = normal_fraction*normal_fraction;
  // d(D grad_n)/d(grad_n), holding rho, opacity, Te, and transverse gradients fixed.
  result.normal_diffusivity =
      fmax((lambda + q*dlambda_dq*normal_fraction_sq)/sigma, 0.0);
  // |d(D grad_n)/d(E_face)|.  The energy floor is constant when it is active.
  result.normal_speed = (energy > energy_floor)
      ? fabs(dlambda_dq)*q*alpha*q*fabs(normal_fraction) : 0.0;

  // energy_floor regularizes R at vanishing E, but it must not become radiation that
  // can be transported.  Enforce the physical |F| <= alpha*c_*max(E_face,0) bound
  // against the actual face energy.  This matters at vacuum boundaries and for groups
  // whose Planck population is below the numerical floor.  When the extra cap is active,
  // its differential response is the free-streaming closure alpha*E*grad/|grad|.
  if (mode != 0 && grad > 0.0) {
    Real causal_coefficient = alpha*fmax(energy, 0.0)/grad;
    if (causal_coefficient < result.diffusion_coefficient) {
      result.diffusion_coefficient = causal_coefficient;
      result.normal_diffusivity =
          causal_coefficient*fmax(1.0-normal_fraction_sq, 0.0);
      result.normal_speed = alpha*fabs(normal_fraction);
    }
  }
  result.streaming_fraction = (mode != 0 && energy > 0.0)
      ? fmin(result.diffusion_coefficient*grad/(alpha*energy), 1.0) : 0.0;
  return result;
}

KOKKOS_INLINE_FUNCTION
Real FLDNumericalFlux(const FLDLinearization &properties, Real normal_grad,
                      Real energy_left, Real energy_right, Real chat,
                      bool use_ap_face) {
  Real flux = -chat*properties.diffusion_coefficient*normal_grad;
  if (use_ap_face) {
    // In the streaming asymptote the FLD flux is an advection flux with bounded
    // velocity F/E.  The centered face energy used by the differential form has no
    // numerical dissipation and retains a parabolic angular Jacobian in multiple
    // dimensions.  Its local Lax--Friedrichs correction is exactly the upwind flux for
    // a frozen streaming direction.  It is conservative, vanishes with resolution,
    // and preserves the target FLD flux to leading order.
    Real face_energy = 0.5*(energy_left+energy_right);
    if (face_energy > 0.0) {
      Real normal_velocity = flux/face_energy;
      flux -= 0.5*fabs(normal_velocity)*(energy_right-energy_left);
    }
  }
  return flux;
}

// Return the face contribution (without c_*) to the diagonal stability rate.  The
// factor 1/2 multiplying normal_speed is from E_face=(E_L+E_R)/2.  When a face is
// uniform to floating-point roundoff, its current nonlinear flux is identically zero.
// Limited FLD then uses the causal grid-scale bound D <= alpha*dx/2 for the stability
// estimate.  This avoids letting an irrelevant 1/(rho*kappa) at a uniform vacuum face
// control the entire calculation, while retaining the exact diffusion coefficient in
// optically thick cells and retaining legacy behavior when no limiter is requested.
KOKKOS_INLINE_FUNCTION
Real FLDFaceStabilityRate(const FLDLinearization &properties, Real energy,
                          Real normal_grad, Real dx_normal, Real dx_short,
                          Real alpha, Real energy_floor, int mode, bool use_ap_face) {
  Real normal_diffusivity = properties.normal_diffusivity;
  Real roundoff_gradient = 64.0*kRealEpsilon*
      fmax(fabs(energy), energy_floor)/dx_short;
  if (mode != 0 && fabs(normal_grad) <= roundoff_gradient) {
    normal_diffusivity = fmin(normal_diffusivity, 0.5*alpha*dx_normal);
  }
  if (use_ap_face) {
    // The matching face flux is upwind in this branch, so its stability condition is
    // hyperbolic.  Do not retain the transverse derivative of the normalized gradient;
    // that derivative is the spurious parabolic restriction the AP flux removes.
    normal_diffusivity = 0.0;
  }
  // When E_face is held at the configured floor, dF/dE is formally zero even though a
  // streaming face can still remove O(c_* E_floor) per crossing time.  The secant speed
  // below supplies the corresponding positivity bound.  It is also the appropriate
  // one-sided bound at a vacuum Dirichlet face.  In ordinary streaming cells it tends
  // to alpha and is identical in scale to the differential characteristic speed.
  Real flux_speed = properties.diffusion_coefficient*fabs(normal_grad)
                    /fmax(fabs(energy), energy_floor);
  Real normal_speed = fmax(properties.normal_speed, flux_speed);
  return normal_diffusivity/(dx_normal*dx_normal)
         + 0.5*normal_speed/dx_normal;
}

KOKKOS_INLINE_FUNCTION
Real RadiationEnergy(const DvceArray5D<Real> &w, int m, int n,
                     int k, int j, int i) {
  return w(m, IDN, k, j, i)*w(m, n, k, j, i);
}

// Group-independent material state used by the batched radiation transport kernels.
struct FLDFaceMaterialState {
  Real density;
  Real electron_temperature;
  Real material0_mass_fraction;
  //! Density-weighted face composition. Valid when the mixture is active; for
  //! nmaterials=2 its first entry equals material0_mass_fraction exactly.
  materials::MaterialComposition composition;
};

struct FLDRadiationFaceState {
  Real energy_left;
  Real energy_right;
  Real energy;
  Real gradient;
  Real normal_gradient;
};

KOKKOS_INLINE_FUNCTION
FLDFaceMaterialState X1FaceMaterialState(
    const DvceArray5D<Real> &w, const DvceArray5D<Real> &temperature,
    int m, int iele, int k, int j, int i, Real gm1, Real fe,
    bool use_materials, const materials::MaterialMixtureDevice &mixture) {
  FLDFaceMaterialState state;
  const Real density_left = w(m, IDN, k, j, i-1);
  const Real density_right = w(m, IDN, k, j, i);
  state.density = 0.5*(density_left+density_right);
  state.material0_mass_fraction = 0.0;
  if (use_materials) {
    state.composition = mixture.CompositionFromPrimitivePair(
        w, m, k, j, i-1, m, k, j, i, density_left, density_right);
    state.material0_mass_fraction = state.composition[0];
    state.electron_temperature = 0.5*(
        temperature(m, 1, k, j, i-1)+temperature(m, 1, k, j, i));
  } else {
    state.electron_temperature =
        0.5*gm1*(w(m, iele, k, j, i-1)+w(m, iele, k, j, i))/fe;
  }
  return state;
}

KOKKOS_INLINE_FUNCTION
FLDFaceMaterialState X2FaceMaterialState(
    const DvceArray5D<Real> &w, const DvceArray5D<Real> &temperature,
    int m, int iele, int k, int j, int i, Real gm1, Real fe,
    bool use_materials, const materials::MaterialMixtureDevice &mixture) {
  FLDFaceMaterialState state;
  const Real density_left = w(m, IDN, k, j-1, i);
  const Real density_right = w(m, IDN, k, j, i);
  state.density = 0.5*(density_left+density_right);
  state.material0_mass_fraction = 0.0;
  if (use_materials) {
    state.composition = mixture.CompositionFromPrimitivePair(
        w, m, k, j-1, i, m, k, j, i, density_left, density_right);
    state.material0_mass_fraction = state.composition[0];
    state.electron_temperature = 0.5*(
        temperature(m, 1, k, j-1, i)+temperature(m, 1, k, j, i));
  } else {
    state.electron_temperature =
        0.5*gm1*(w(m, iele, k, j-1, i)+w(m, iele, k, j, i))/fe;
  }
  return state;
}

KOKKOS_INLINE_FUNCTION
FLDFaceMaterialState X3FaceMaterialState(
    const DvceArray5D<Real> &w, const DvceArray5D<Real> &temperature,
    int m, int iele, int k, int j, int i, Real gm1, Real fe,
    bool use_materials, const materials::MaterialMixtureDevice &mixture) {
  FLDFaceMaterialState state;
  const Real density_left = w(m, IDN, k-1, j, i);
  const Real density_right = w(m, IDN, k, j, i);
  state.density = 0.5*(density_left+density_right);
  state.material0_mass_fraction = 0.0;
  if (use_materials) {
    state.composition = mixture.CompositionFromPrimitivePair(
        w, m, k-1, j, i, m, k, j, i, density_left, density_right);
    state.material0_mass_fraction = state.composition[0];
    state.electron_temperature = 0.5*(
        temperature(m, 1, k-1, j, i)+temperature(m, 1, k, j, i));
  } else {
    state.electron_temperature =
        0.5*gm1*(w(m, iele, k-1, j, i)+w(m, iele, k, j, i))/fe;
  }
  return state;
}

KOKKOS_INLINE_FUNCTION
FLDRadiationFaceState X1RadiationFaceState(
    const DvceArray5D<Real> &w, int m, int n, int k, int j, int i,
    bool multi_d, bool three_d, Real dx1, Real dx2, Real dx3) {
  Real el = RadiationEnergy(w, m, n, k, j, i-1);
  Real er = RadiationEnergy(w, m, n, k, j, i);
  Real grad1 = (er-el)/dx1;
  Real grad2 = 0.0;
  Real grad3 = 0.0;
  if (multi_d) {
    Real ell = RadiationEnergy(w, m, n, k, j-1, i-1);
    Real elu = RadiationEnergy(w, m, n, k, j+1, i-1);
    Real erl = RadiationEnergy(w, m, n, k, j-1, i);
    Real eru = RadiationEnergy(w, m, n, k, j+1, i);
    grad2 = (elu-ell+eru-erl)/(4.0*dx2);
  }
  if (three_d) {
    Real ell = RadiationEnergy(w, m, n, k-1, j, i-1);
    Real elu = RadiationEnergy(w, m, n, k+1, j, i-1);
    Real erl = RadiationEnergy(w, m, n, k-1, j, i);
    Real eru = RadiationEnergy(w, m, n, k+1, j, i);
    grad3 = (elu-ell+eru-erl)/(4.0*dx3);
  }

  FLDRadiationFaceState state;
  state.energy_left = el;
  state.energy_right = er;
  state.energy = 0.5*(el+er);
  state.gradient = RadiationGradientMagnitude(grad1, grad2, grad3);
  state.normal_gradient = grad1;
  return state;
}

KOKKOS_INLINE_FUNCTION
FLDRadiationFaceState X2RadiationFaceState(
    const DvceArray5D<Real> &w, int m, int n, int k, int j, int i,
    bool three_d, Real dx1, Real dx2, Real dx3) {
  Real el = RadiationEnergy(w, m, n, k, j-1, i);
  Real er = RadiationEnergy(w, m, n, k, j, i);
  Real ell = RadiationEnergy(w, m, n, k, j-1, i-1);
  Real elu = RadiationEnergy(w, m, n, k, j-1, i+1);
  Real erl = RadiationEnergy(w, m, n, k, j, i-1);
  Real eru = RadiationEnergy(w, m, n, k, j, i+1);
  Real grad1 = (elu-ell+eru-erl)/(4.0*dx1);
  Real grad2 = (er-el)/dx2;
  Real grad3 = 0.0;
  if (three_d) {
    ell = RadiationEnergy(w, m, n, k-1, j-1, i);
    elu = RadiationEnergy(w, m, n, k+1, j-1, i);
    erl = RadiationEnergy(w, m, n, k-1, j, i);
    eru = RadiationEnergy(w, m, n, k+1, j, i);
    grad3 = (elu-ell+eru-erl)/(4.0*dx3);
  }

  FLDRadiationFaceState state;
  state.energy_left = el;
  state.energy_right = er;
  state.energy = 0.5*(el+er);
  state.gradient = RadiationGradientMagnitude(grad1, grad2, grad3);
  state.normal_gradient = grad2;
  return state;
}

KOKKOS_INLINE_FUNCTION
FLDRadiationFaceState X3RadiationFaceState(
    const DvceArray5D<Real> &w, int m, int n, int k, int j, int i,
    Real dx1, Real dx2, Real dx3) {
  Real el = RadiationEnergy(w, m, n, k-1, j, i);
  Real er = RadiationEnergy(w, m, n, k, j, i);
  Real ell = RadiationEnergy(w, m, n, k-1, j, i-1);
  Real elu = RadiationEnergy(w, m, n, k-1, j, i+1);
  Real erl = RadiationEnergy(w, m, n, k, j, i-1);
  Real eru = RadiationEnergy(w, m, n, k, j, i+1);
  Real grad1 = (elu-ell+eru-erl)/(4.0*dx1);
  ell = RadiationEnergy(w, m, n, k-1, j-1, i);
  elu = RadiationEnergy(w, m, n, k-1, j+1, i);
  erl = RadiationEnergy(w, m, n, k, j-1, i);
  eru = RadiationEnergy(w, m, n, k, j+1, i);
  Real grad2 = (elu-ell+eru-erl)/(4.0*dx2);
  Real grad3 = (er-el)/dx3;

  FLDRadiationFaceState state;
  state.energy_left = el;
  state.energy_right = er;
  state.energy = 0.5*(el+er);
  state.gradient = RadiationGradientMagnitude(grad1, grad2, grad3);
  state.normal_gradient = grad3;
  return state;
}

} // namespace

//----------------------------------------------------------------------------------------
// Constructor.  Group boundaries are photon energies h*nu/k_B in code-temperature units;
// constant and tabulated models both return mass opacities, so sigma=rho*kappa.

ThermalRadiation::ThermalRadiation(MeshBlockPack *ppack, ParameterInput *pin,
    int first_group_index, int electron_index, Real gamma_minus_one,
    Real electron_heat_capacity_fraction,
    materials::MaterialMixture *material_mixture) :
    ngroups(pin->GetInteger("thermal_radiation", "n_groups")),
    ifirst(first_group_index),
    dtnew(FLT_MAX),
    diagnostics("thermal-radiation-diagnostics", 1, 1, 1, 1, 1),
    pmy_pack_(ppack),
    iele_(electron_index),
    gamma_minus_one_(gamma_minus_one),
    cv_e_fraction_(electron_heat_capacity_fraction),
    use_material_mixture_(material_mixture != nullptr),
    group_bounds_("thermal-radiation-bounds", 1),
    kappa_transport_("thermal-radiation-kappa-transport", 1),
    kappa_absorption_("thermal-radiation-kappa-absorption", 1),
    kappa_emission_("thermal-radiation-kappa-emission", 1),
    source_integer_stats_("thermal-radiation-source-integer-stats", 1),
    source_real_stats_("thermal-radiation-source-real-stats", 1) {
  if (use_material_mixture_) material_mixture_ = material_mixture->DeviceData();
  if (ngroups < 1 || ngroups > 100) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<thermal_radiation>/n_groups must be between 1 and 100"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  arad_ = pin->GetReal("thermal_radiation", "arad");
  chat_ = pin->GetReal("thermal_radiation", "c_light");
  flux_limit_coefficient_ =
      pin->GetOrAddReal("thermal_radiation", "flux_limit_coefficient", 1.0);
  initial_radiation_temperature_ =
      pin->GetOrAddReal("thermal_radiation", "initial_radiation_temperature", 0.0);
  initial_radiation_temperature_right_ = initial_radiation_temperature_;
  initial_radiation_x1_ = 0.0;
  energy_floor_ = pin->GetOrAddReal("thermal_radiation", "energy_floor", 1.0e-30);
  source_cfl_ = pin->GetOrAddReal("thermal_radiation", "source_cfl", 0.1);
  couple_matter_ = pin->GetOrAddBoolean("thermal_radiation", "couple_matter", true);
  const std::string source_integrator = pin->GetOrAddString(
      "thermal_radiation", "source_integrator", "nonlinear");
  if (source_integrator == "nonlinear" || source_integrator == "coupled") {
    nonlinear_source_ = true;
  } else if (source_integrator == "lagged" || source_integrator == "frozen" ||
             source_integrator == "time-lagged") {
    nonlinear_source_ = false;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/source_integrator='"
              << source_integrator << "'; expected nonlinear or lagged" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real minimum_source_tolerance = 64.0*kRealEpsilon;
  const Real default_source_tolerance =
      std::max(static_cast<Real>(1.0e-10), minimum_source_tolerance);
  source_nonlinear_tolerance_ = pin->GetOrAddReal(
      "thermal_radiation", "source_nonlinear_tolerance",
      default_source_tolerance);
  source_nonlinear_absolute_tolerance_ = pin->GetOrAddReal(
      "thermal_radiation", "source_nonlinear_absolute_tolerance", 0.0);
  source_max_iterations_ = pin->GetOrAddInteger(
      "thermal_radiation", "source_max_iterations", 80);
  source_fallback_substeps_ = pin->GetOrAddInteger(
      "thermal_radiation", "source_fallback_substeps", 8);
  source_report_ = pin->GetOrAddBoolean(
      "thermal_radiation", "source_report", false);
  if (!std::isfinite(source_nonlinear_tolerance_) ||
      source_nonlinear_tolerance_ < minimum_source_tolerance ||
      source_nonlinear_tolerance_ >= 1.0 ||
      !std::isfinite(source_nonlinear_absolute_tolerance_) ||
      source_nonlinear_absolute_tolerance_ < 0.0 ||
      source_max_iterations_ <= 0 || source_fallback_substeps_ <= 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Nonlinear radiation-source tolerance must be finite, "
              << "at least " << minimum_source_tolerance
              << ", and less than one; the absolute tolerance must be finite and "
              << "non-negative; iteration and fallback-substep counts must be positive"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string transport_integrator = pin->GetOrAddString(
      "thermal_radiation", "transport_integrator", "explicit");
  if (transport_integrator == "explicit") {
    implicit_transport_ = false;
  } else if (transport_integrator == "implicit") {
    implicit_transport_ = true;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/transport_integrator='"
              << transport_integrator << "'; expected explicit or implicit" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real minimum_implicit_tolerance = 64.0*kRealEpsilon;
  const Real default_implicit_tolerance =
      std::max(static_cast<Real>(1.0e-10), minimum_implicit_tolerance);
  implicit_tolerance_ = pin->GetOrAddReal(
      "thermal_radiation", "implicit_tolerance", default_implicit_tolerance);
  implicit_max_iterations_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_max_iterations", 400);
  implicit_residual_check_interval_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_residual_check_interval", 50);
  implicit_amr_order_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_amr_order", 2);
  implicit_gmres_restart_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_gmres_restart", 30);
  const std::string orthogonalization = pin->GetOrAddString(
      "thermal_radiation", "implicit_gmres_orthogonalization", "mgs2");
  if (orthogonalization != "mgs2" && orthogonalization != "cgs2") {
    ImplicitRadiationError("implicit_gmres_orthogonalization must be mgs2 or cgs2");
  }
  implicit_gmres_cgs2_ = orthogonalization == "cgs2";
  implicit_amr_inner_iterations_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_amr_inner_iterations", 0);
  const std::string implicit_amr_inner_norm = pin->GetOrAddString(
      "thermal_radiation", "implicit_amr_inner_norm", "volume");
  if (implicit_amr_inner_norm == "preconditioner") {
    if (!implicit_transport_ || !ppack->pmesh->multilevel) {
      ImplicitRadiationError("implicit_amr_inner_norm=preconditioner requires "
                             "implicit transport on an SMR/AMR mesh");
    }
    implicit_amr_inner_preconditioner_norm_ = true;
  } else if (implicit_amr_inner_norm != "volume") {
    ImplicitRadiationError("implicit_amr_inner_norm must be volume or preconditioner");
  }
  const std::string implicit_gmres_metric = pin->GetOrAddString(
      "thermal_radiation", "implicit_gmres_metric", "volume");
  if (implicit_gmres_metric == "preconditioner") {
    if (!implicit_transport_ || !ppack->pmesh->multilevel) {
      ImplicitRadiationError("implicit_gmres_metric=preconditioner requires "
                             "implicit transport on an SMR/AMR mesh");
    }
    implicit_gmres_preconditioner_metric_ = true;
  } else if (implicit_gmres_metric != "volume") {
    ImplicitRadiationError("implicit_gmres_metric must be volume or preconditioner");
  }
  if ((implicit_amr_order_ != 1 && implicit_amr_order_ != 2) ||
      implicit_gmres_restart_ < 2 || implicit_amr_inner_iterations_ < 0) {
    ImplicitRadiationError("implicit_amr_order must be 1 or 2 and "
                           "implicit_gmres_restart must be at least 2; "
                           "implicit_amr_inner_iterations must be nonnegative");
  }
  implicit_report_ = pin->GetOrAddBoolean(
      "thermal_radiation", "implicit_report", false);
  solver_timers_.enabled = pin->GetOrAddBoolean(
      "thermal_radiation", "solver_timers", false);
  solver_summary_ = pin->GetOrAddBoolean(
      "thermal_radiation", "solver_summary", false);
  const std::string halo_exchange = pin->GetOrAddString(
      "thermal_radiation", "implicit_halo_exchange", "packed");
  if (halo_exchange == "legacy") implicit_halo_exchange_ = 0;
  else if (halo_exchange == "packed") implicit_halo_exchange_ = 1;
  else if (halo_exchange == "host") implicit_halo_exchange_ = 2;
  else ImplicitRadiationError("implicit_halo_exchange must be legacy, packed, or host");
  implicit_halo_faces_ = pin->GetOrAddBoolean(
      "thermal_radiation", "implicit_halo_faces", true);
  implicit_halo_compact_ = pin->GetOrAddBoolean(
      "thermal_radiation", "implicit_halo_compact", true);
  const std::string implicit_preconditioner = pin->GetOrAddString(
      "thermal_radiation", "implicit_preconditioner", "jacobi");
  int default_group_batch_size = 1;
#ifdef KOKKOS_ENABLE_CUDA
  if (implicit_preconditioner == "amr-amg" && global_variable::nranks > 1 &&
      implicit_halo_exchange_ != 0 && RadiationGroupBatch::Supported()) {
    default_group_batch_size = 4;
  }
#endif
  implicit_group_batch_size_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_group_batch_size", default_group_batch_size);
  if (implicit_group_batch_size_ < 1 || implicit_group_batch_size_ > 4) {
    ImplicitRadiationError("implicit_group_batch_size must be between 1 and 4");
  }
  if (implicit_group_batch_size_ > 1 &&
      (!RadiationGroupBatch::Supported() ||
       (implicit_preconditioner != "amr-amg" && implicit_preconditioner != "cell-amg" &&
        implicit_preconditioner != "native-mg") ||
       implicit_halo_exchange_ == 0)) {
    ImplicitRadiationError("Group batching requires Linux, amr-amg, cell-amg or native-mg, "
                           "and packed or host halos");
  }
  if (implicit_preconditioner == "jacobi") {
    implicit_preconditioner_mode_ = 0;
  } else if (implicit_preconditioner == "block-coarse") {
    implicit_preconditioner_mode_ = 1;
  } else if (implicit_preconditioner == "native-mg") {
    if (!implicit_transport_) ImplicitRadiationError("native-mg requires implicit transport");
    implicit_preconditioner_mode_ = 5;
    const int max_mib = pin->GetOrAddInteger(
        "thermal_radiation", "native_mg_max_mib", 1024);
    if (max_mib <= 0) ImplicitRadiationError("native_mg_max_mib must be positive");
    implicit_native_max_bytes_ = static_cast<std::size_t>(max_mib)*1024*1024;
    implicit_native_device_mpi_ = pin->GetOrAddBoolean(
        "thermal_radiation", "native_mg_device_mpi", false);
  } else if (implicit_preconditioner == "amr-aggregate" ||
             implicit_preconditioner == "amr-amg" ||
             implicit_preconditioner == "cell-amg") {
    const bool cell_amg = implicit_preconditioner == "cell-amg";
    const bool use_amg = implicit_preconditioner != "amr-aggregate";
    implicit_preconditioner_mode_ = cell_amg ? 4 : (use_amg ? 3 : 2);
    if (use_amg && !HasImplicitAMRAMGSupport()) {
      ImplicitRadiationError(
          implicit_preconditioner+" requires Athena_ENABLE_CHOLMOD=ON and "
          "Athena_ENABLE_HYPRE=ON");
    } else if (!HasImplicitAMRAggregateSupport()) {
      ImplicitRadiationError(
          "amr-aggregate requires a build with Athena_ENABLE_CHOLMOD=ON");
    }
    const auto &indices = ppack->pmesh->mb_indcs;
    const auto supported = [](int n) { return n == 1 || (n >= 9 && n%9 == 0); };
    if (!implicit_transport_ || !ppack->pmesh->multilevel) {
      ImplicitRadiationError(implicit_preconditioner+
          " requires implicit transport with mesh_refinement/refinement=static or adaptive");
    }
    if (!cell_amg && (indices.nx1 < 9 || !supported(indices.nx1) ||
        !supported(indices.nx2) || !supported(indices.nx3))) {
      ImplicitRadiationError(implicit_preconditioner+
          " requires every active MeshBlock dimension divisible by nine");
    }
    if (use_amg) {
      const int max_mib = pin->GetOrAddInteger(
          "thermal_radiation", "implicit_amr_amg_max_mib", 1024);
      constexpr std::size_t mib = 1024*1024;
      if (max_mib <= 0 || static_cast<std::size_t>(max_mib) >
          std::numeric_limits<std::size_t>::max()/mib) {
        ImplicitRadiationError("implicit_amr_amg_max_mib must be a positive "
                               "representable memory budget");
      }
      implicit_amr_amg_max_bytes_ = static_cast<std::size_t>(max_mib)*mib;
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/implicit_preconditioner='"
              << implicit_preconditioner
              << "'; expected jacobi, block-coarse, amr-aggregate, amr-amg, cell-amg, or native-mg"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  implicit_gmres_restart_max_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_gmres_restart_max", implicit_preconditioner_mode_ == 5 ?
      std::max(100, implicit_gmres_restart_) : implicit_gmres_restart_);
  if (implicit_gmres_restart_max_ < implicit_gmres_restart_) {
    ImplicitRadiationError("implicit_gmres_restart_max must be at least implicit_gmres_restart");
  }
  implicit_amg_reuse_ = pin->GetOrAddBoolean(
      "thermal_radiation", "implicit_amg_reuse", false);
  implicit_amg_reuse_steps_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_amg_reuse_steps", 4);
  implicit_amg_reuse_slots_ = pin->GetOrAddInteger(
      "thermal_radiation", "implicit_amg_reuse_slots", 5);
  implicit_amg_reuse_drift_ = pin->GetOrAddReal(
      "thermal_radiation", "implicit_amg_reuse_drift", 64.0);
  if (implicit_amg_reuse_steps_ < 1 || implicit_amg_reuse_slots_ < 1 ||
      !std::isfinite(implicit_amg_reuse_drift_) || implicit_amg_reuse_drift_ < 0.0) {
    ImplicitRadiationError("implicit_amg_reuse_steps and implicit_amg_reuse_slots "
                           "must be positive; implicit_amg_reuse_drift must be "
                           "finite and nonnegative");
  }
  if (implicit_amg_reuse_) {
    if (implicit_preconditioner_mode_ != 4) {
      ImplicitRadiationError("implicit_amg_reuse requires implicit_preconditioner=cell-amg");
    }
    if (!ImplicitAMGHypre::ReuseAvailable()) {
      ImplicitRadiationError("implicit_amg_reuse requires Athena_ENABLE_HYPRE_REUSE=ON "
                             "with static HYPRE 2.33.0 CUDA");
    }
    if (implicit_group_batch_size_ < 2 || ngroups < 2) {
      ImplicitRadiationError("implicit_amg_reuse requires implicit_group_batch_size >= 2 "
                             "and n_groups >= 2");
    }
    if (global_variable::my_rank == 0) {
      std::cout << "# cell-amg reuse configuration: enabled=1 steps="
                << implicit_amg_reuse_steps_ << " slots=" << implicit_amg_reuse_slots_
                << " drift=" << implicit_amg_reuse_drift_ << std::endl;
    }
  }
  const std::string coarse_solver = pin->GetOrAddString(
      "thermal_radiation", "implicit_amr_coarse_solver", "auto");
  if (implicit_preconditioner_mode_ >= 3 && coarse_solver != "auto") {
    ImplicitRadiationError(implicit_preconditioner+
                           " requires implicit_amr_coarse_solver=auto; "
                           "hypre executes on the build's CPU or CUDA backend");
  }
  if (coarse_solver == "host") {
    implicit_amr_coarse_solver_ = 1;
  } else if (coarse_solver == "device") {
    implicit_amr_coarse_solver_ = 2;
    if (implicit_preconditioner_mode_ != 2 ||
        !HasImplicitAMRAggregateDeviceSupport()) {
      ImplicitRadiationError("implicit_amr_coarse_solver=device requires amr-aggregate "
          "and a CUDA build with Athena_ENABLE_CUSPARSE=ON");
    }
  } else if (coarse_solver != "auto") {
    ImplicitRadiationError("implicit_amr_coarse_solver must be auto, host, or device");
  }
  if (!std::isfinite(implicit_tolerance_) ||
      implicit_tolerance_ < minimum_implicit_tolerance ||
      implicit_tolerance_ >= 1.0 ||
      implicit_max_iterations_ <= 0 || implicit_residual_check_interval_ <= 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Implicit radiation tolerance must be finite, at least "
              << minimum_implicit_tolerance << ", and less than one; iteration limit "
              << "and residual-check interval must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (implicit_transport_) {
    for (int face = BoundaryFace::inner_x1; face <= BoundaryFace::outer_x3; ++face) {
      if (ppack->pmesh->mesh_bcs[face] == BoundaryFlag::shear_periodic) {
        ImplicitRadiationError(
            "Implicit thermal-radiation transport does not support shear-periodic "
            "boundaries");
      }
    }
  }

  const char *boundary_names[6] = {
      "implicit_x1_inner_boundary", "implicit_x1_outer_boundary",
      "implicit_x2_inner_boundary", "implicit_x2_outer_boundary",
      "implicit_x3_inner_boundary", "implicit_x3_outer_boundary"};
  const char *value_names[6] = {
      "implicit_x1_inner_value", "implicit_x1_outer_value",
      "implicit_x2_inner_value", "implicit_x2_outer_value",
      "implicit_x3_inner_value", "implicit_x3_outer_value"};
  const char *temperature_names[6] = {
      "implicit_x1_inner_temperature", "implicit_x1_outer_temperature",
      "implicit_x2_inner_temperature", "implicit_x2_outer_temperature",
      "implicit_x3_inner_temperature", "implicit_x3_outer_temperature"};
  Real boundary_temperature[6] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
  for (int face = 0; face < 6; ++face) {
    const std::string boundary = pin->GetOrAddString(
        "thermal_radiation", boundary_names[face], "neumann");
    if (boundary == "neumann" || boundary == "zero-gradient" ||
        boundary == "reflecting") {
      implicit_boundary_type_[face] = 0;
    } else if (boundary == "dirichlet") {
      implicit_boundary_type_[face] = 1;
    } else if (boundary == "vacuum") {
      implicit_boundary_type_[face] = 2;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unknown <thermal_radiation>/"
                << boundary_names[face] << "='" << boundary
                << "'; expected neumann, dirichlet, or vacuum" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pin->DoesParameterExist("thermal_radiation", temperature_names[face])) {
      if (!implicit_transport_ || implicit_boundary_type_[face] != 1) {
        ImplicitRadiationError(
            std::string(temperature_names[face])+" requires implicit transport "
            "and a dirichlet boundary");
      }
      if (pin->DoesParameterExist("thermal_radiation", value_names[face])) {
        ImplicitRadiationError(
            std::string("Specify only one of ")+temperature_names[face]+" and "+
            value_names[face]);
      }
      boundary_temperature[face] = pin->GetReal(
          "thermal_radiation", temperature_names[face]);
      if (!std::isfinite(boundary_temperature[face]) ||
          boundary_temperature[face] < 0.0) {
        ImplicitRadiationError("Radiation boundary temperature must be finite "
                               "and non-negative");
      }
    } else {
      implicit_boundary_value_[face] = pin->GetOrAddReal(
          "thermal_radiation", value_names[face], 0.0);
    }
    if (!std::isfinite(implicit_boundary_value_[face]) ||
        implicit_boundary_value_[face] < 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Implicit radiation boundary values must be finite "
                << "and non-negative" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // These AP/upwind controls apply only to the explicit face-flux path.  The implicit
  // integrator uses a centered frozen-coefficient diffusion matrix.
  std::string transport_discretization = pin->GetOrAddString(
      "thermal_radiation", "transport_discretization", "asymptotic-preserving");
  if (transport_discretization == "asymptotic-preserving" ||
      transport_discretization == "ap") {
    use_ap_transport_ = true;
  } else if (transport_discretization == "face-jacobian" ||
             transport_discretization == "legacy") {
    use_ap_transport_ = false;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/transport_discretization='"
              << transport_discretization << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ap_streaming_threshold_ = pin->GetOrAddReal(
      "thermal_radiation", "ap_streaming_threshold", 0.5);
  ap_optical_depth_threshold_ = pin->GetOrAddReal(
      "thermal_radiation", "ap_optical_depth_threshold", 1.0);

  std::string initial_profile =
      pin->GetOrAddString("thermal_radiation", "initial_profile", "uniform");
  if (initial_profile == "uniform") {
    initial_profile_mode_ = 0;
  } else if (initial_profile == "step") {
    initial_profile_mode_ = 1;
    initial_radiation_temperature_right_ = pin->GetReal(
        "thermal_radiation", "initial_radiation_temperature_right");
    initial_radiation_x1_ =
        pin->GetOrAddReal("thermal_radiation", "initial_radiation_x1", 0.0);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/initial_profile='"
              << initial_profile << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (arad_ <= 0.0 || chat_ <= 0.0 || flux_limit_coefficient_ <= 0.0 ||
      initial_radiation_temperature_ < 0.0 ||
      initial_radiation_temperature_right_ < 0.0 || energy_floor_ <= 0.0 ||
      ap_streaming_threshold_ <= 0.0 || ap_streaming_threshold_ > 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Thermal-radiation constants must be positive and the "
              << "initial radiation temperature must be non-negative; the AP streaming "
              << "threshold must lie in (0,1]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ap_optical_depth_threshold_ <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<thermal_radiation>/ap_optical_depth_threshold "
              << "must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string limiter =
      pin->GetOrAddString("thermal_radiation", "flux_limiter", "levermore-pomraning");
  if (limiter == "none") {
    limiter_mode_ = 0;
  } else if (limiter == "harmonic") {
    limiter_mode_ = 1;
  } else if (limiter == "larsen") {
    limiter_mode_ = 2;
  } else if (limiter == "minmax" || limiter == "min/max") {
    limiter_mode_ = 3;
  } else if (limiter == "levermore-pomraning" || limiter == "levermore") {
    limiter_mode_ = 4;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/flux_limiter='" << limiter
              << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Kokkos::realloc(group_bounds_, ngroups + 1);
  Kokkos::realloc(kappa_transport_, ngroups);
  Kokkos::realloc(kappa_absorption_, ngroups);
  Kokkos::realloc(kappa_emission_, ngroups);
  Kokkos::realloc(source_integer_stats_, 2);
  Kokkos::realloc(source_real_stats_, 1);

  for (int g = 0; g <= ngroups; ++g) {
    group_bounds_.h_view(g) = pin->GetReal(
        "thermal_radiation", "group_bound_" + std::to_string(g));
    if (group_bounds_.h_view(g) < 0.0 ||
        (g > 0 && group_bounds_.h_view(g) <= group_bounds_.h_view(g-1))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Thermal-radiation group boundaries must be "
                << "non-negative and strictly increasing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // Match the initialized/emitted spectrum, including energy outside the configured
  // group interval: do not renormalize a truncated spectrum to a full blackbody.
  implicit_boundary_group_values_.resize(6*ngroups);
  for (int face = 0; face < 6; ++face) {
    const Real temperature = boundary_temperature[face];
    const Real blackbody = temperature >= 0.0 ? arad_*SQR(SQR(temperature)) : 0.0;
    if (!std::isfinite(blackbody)) {
      ImplicitRadiationError("Radiation boundary blackbody energy overflows");
    }
    for (int g = 0; g < ngroups; ++g) {
      implicit_boundary_group_values_[face*ngroups+g] = temperature >= 0.0 ?
          blackbody*PlanckGroupFraction(group_bounds_.h_view(g),
              group_bounds_.h_view(g+1), temperature) : implicit_boundary_value_[face];
    }
  }
  std::string opacity_model =
      pin->GetOrAddString("thermal_radiation", "opacity_model", "constant");
  if (opacity_model == "constant") {
    for (int g = 0; g < ngroups; ++g) {
      std::string suffix = std::to_string(g);
      kappa_transport_.h_view(g) = pin->GetReal(
          "thermal_radiation", "kappa_transport_" + suffix);
      kappa_absorption_.h_view(g) = pin->GetOrAddReal(
          "thermal_radiation", "kappa_absorption_" + suffix, 0.0);
      kappa_emission_.h_view(g) = pin->GetOrAddReal(
          "thermal_radiation", "kappa_emission_" + suffix,
          kappa_absorption_.h_view(g));
      if (kappa_transport_.h_view(g) <= 0.0 ||
          kappa_absorption_.h_view(g) < 0.0 || kappa_emission_.h_view(g) < 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Transport opacities must be positive and absorption/"
                  << "emission opacities must be non-negative" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  } else if (opacity_model == "table" || opacity_model == "tabulated" ||
             opacity_model == "mixed-table" ||
             opacity_model == "mixed_tabulated") {
    // Every active component needs its own opacity table; the mixture decides how many.
    const int opacity_materials =
        use_material_mixture_ ? material_mixture->NumberOfMaterials() : 2;
    const bool material0_table = pin->DoesParameterExist(
        "materials", "material0_opacity_table_file");
    bool all_material_tables = material0_table;
    for (int n = 1; n < opacity_materials; ++n) {
      const bool present = pin->DoesParameterExist(
          "materials", "material"+std::to_string(n)+"_opacity_table_file");
      all_material_tables = all_material_tables && present;
      if (present != material0_table) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Mixed opacity requires a "
                  << "material*_opacity_table_file for all "
                  << opacity_materials << " materials" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    const bool explicitly_mixed =
        (opacity_model == "mixed-table" || opacity_model == "mixed_tabulated");
    if (explicitly_mixed && !all_material_tables) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<thermal_radiation>/opacity_model="
                << opacity_model << " requires every material opacity table"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (all_material_tables) {
      if (!use_material_mixture_) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Mixed material opacity tables require an active "
                  << "<materials> mixture" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      use_mixed_opacity_table_ = true;
      mixed_opacity_table_ = new MixedOpacityTable(
          pin, ngroups, group_bounds_, opacity_materials);
    } else {
      use_opacity_table_ = true;
      opacity_table_ = new OpacityTable(pin, ngroups, group_bounds_);
    }
    for (int g = 0; g < ngroups; ++g) {
      kappa_transport_.h_view(g) = 1.0;
      kappa_absorption_.h_view(g) = 0.0;
      kappa_emission_.h_view(g) = 0.0;
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/opacity_model='"
              << opacity_model << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  group_bounds_.modify_host();
  kappa_transport_.modify_host();
  kappa_absorption_.modify_host();
  kappa_emission_.modify_host();
  group_bounds_.sync_device();
  kappa_transport_.sync_device();
  kappa_absorption_.sync_device();
  kappa_emission_.sync_device();

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(diagnostics, nmb, 2, ncells3, ncells2, ncells1);
  if (implicit_transport_) {
    if (indcs.ng < 2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Implicit thermal-radiation transport requires at "
                << "least two ghost cells" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto allocate = [&](DvceArray5D<Real> &view) {
      Kokkos::realloc(view, nmb, 1, ncells3, ncells2, ncells1);
    };
    allocate(implicit_old_);
    allocate(implicit_solution_);
    allocate(implicit_coefficient_);
    allocate(implicit_residual_);
    allocate(implicit_direction_);
    allocate(implicit_preconditioned_);
    allocate(implicit_operator_);
    allocate(implicit_coarse_scratch_);
    if (ppack->pmesh->multilevel) {
      implicit_amr_face_origin_ = {indcs.ks, indcs.js, indcs.is};
      const int nf1 = indcs.nx1, nf2 = indcs.nx2, nf3 = indcs.nx3;
      implicit_amr_diagonal_ = DvceArray5D<Real>(
          "rad_amr_diagonal", nmb, 1, nf3, nf2, nf1);
      if (implicit_amr_order_ == 2) {
        implicit_amr_gradient_ = DvceArray5D<Real>(
            "rad_amr_gradient", nmb, 3, nf3, nf2, nf1);
        if (implicit_amr_inner_iterations_ > 0) allocate(implicit_amr_inner_z_);
      }
      // The composite operator and reflux only use faces surrounding active cells.
      // Keep the mesh indices in kernels and translate into these compact registers.
      const int nf2f = ppack->pmesh->multi_d ? nf2+1 : 1;
      const int nf3f = ppack->pmesh->three_d ? nf3+1 : 1;
      for (auto *faces : {&implicit_amr_conductance_, &implicit_amr_flux_}) {
        Kokkos::realloc(faces->x1f, nmb, 1, nf3, nf2, nf1+1);
        Kokkos::realloc(faces->x2f, nmb, 1, nf3, nf2f, nf1);
        Kokkos::realloc(faces->x3f, nmb, 1, nf3f, nf2, nf1);
      }
    }

    // AMR uses local aggregate-diagonal corrections and a rebuilt global graph.
    // The uniform hierarchy additionally exchanges aggregate faces between blocks.
    if (implicit_preconditioner_mode_ >= 1 && implicit_preconditioner_mode_ < 4) {
      // Two factor-three aggregate levels are available whenever every active
      // MeshBlock dimension is divisible by nine.  The final (possibly non-factor-three)
      // collapse is the existing one-value-per-MeshBlock root; for DCI this gives
      // 45^3 -> 15^3 -> 5^3 -> 1^3.  Other block sizes retain point Jacobi.
      const auto supports_two_levels = [](const int n) {
        return n == 1 || (n >= 9 && n%9 == 0);
      };
      const bool supports_multilevel = supports_two_levels(indcs.nx1) &&
          supports_two_levels(indcs.nx2) && supports_two_levels(indcs.nx3);
      // Red/black smoothing is not a two-coloring across an odd periodic wrap.  Reject
      // that combination explicitly instead of silently applying a non-SPD smoother
      // inside ordinary conjugate gradient.
      const bool odd_periodic_x1 =
          ppack->pmesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic &&
          ppack->pmesh->mesh_indcs.nx1%2 != 0;
      const bool odd_periodic_x2 = ppack->pmesh->multi_d &&
          ppack->pmesh->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic &&
          ppack->pmesh->mesh_indcs.nx2%2 != 0;
      const bool odd_periodic_x3 = ppack->pmesh->three_d &&
          ppack->pmesh->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic &&
          ppack->pmesh->mesh_indcs.nx3%2 != 0;
      if (!ppack->pmesh->multilevel && supports_multilevel &&
          (odd_periodic_x1 || odd_periodic_x2 || odd_periodic_x3)) {
        ImplicitRadiationError(
            "The block-coarse implicit-radiation multilevel preconditioner does not "
            "support an odd number of cells in a periodic direction");
      }
      implicit_multilevel_enabled_ = supports_multilevel;
      if (implicit_multilevel_enabled_) {
        implicit_multilevel_nx1_[0] = indcs.nx1 == 1 ? 1 : indcs.nx1/3;
        implicit_multilevel_nx2_[0] = indcs.nx2 == 1 ? 1 : indcs.nx2/3;
        implicit_multilevel_nx3_[0] = indcs.nx3 == 1 ? 1 : indcs.nx3/3;
        implicit_multilevel_nx1_[1] = indcs.nx1 == 1 ? 1 : indcs.nx1/9;
        implicit_multilevel_nx2_[1] = indcs.nx2 == 1 ? 1 : indcs.nx2/9;
        implicit_multilevel_nx3_[1] = indcs.nx3 == 1 ? 1 : indcs.nx3/9;
        const int level1_cells = implicit_multilevel_nx1_[0]*
            implicit_multilevel_nx2_[0]*implicit_multilevel_nx3_[0];
        const int level2_cells = implicit_multilevel_nx1_[1]*
            implicit_multilevel_nx2_[1]*implicit_multilevel_nx3_[1];
        implicit_multilevel_offset_[0] = 0;
        implicit_multilevel_offset_[1] = level1_cells;
        Kokkos::realloc(
            implicit_multilevel_vector_, nmb, level1_cells+level2_cells);
      }
      if (implicit_multilevel_enabled_ && !ppack->pmesh->multilevel) {
        Kokkos::realloc(
            implicit_multilevel_conductance_, nmb, 6,
            implicit_multilevel_vector_.extent_int(1));
        const int max_face_cells = std::max(
            std::max(indcs.nx2*indcs.nx3, indcs.nx1*indcs.nx3),
            indcs.nx1*indcs.nx2);
        Kokkos::realloc(implicit_multilevel_send_faces_, nmb, 6, max_face_cells);
        Kokkos::realloc(implicit_multilevel_recv_faces_, nmb, 6, max_face_cells);
#if MPI_PARALLEL_ENABLED
        if (MPI_Comm_dup(MPI_COMM_WORLD, &implicit_multilevel_comm_) != MPI_SUCCESS) {
          ImplicitRadiationError(
              "Could not duplicate the implicit-radiation multilevel communicator");
        }
#endif
      }

      if (implicit_multilevel_enabled_ && !ppack->pmesh->multilevel) {
        const int ncoarse = ppack->pmesh->nmb_total;
        if (ncoarse > 1024) {
          ImplicitRadiationError(
              "The dense implicit-radiation root solve supports at most 1024 MeshBlocks");
        }
        Kokkos::realloc(implicit_coarse_faces_, ncoarse, 6);
        Kokkos::realloc(implicit_coarse_vector_, ncoarse);
        Kokkos::realloc(implicit_coarse_neighbor_gid_device_, ncoarse, 6);
        Kokkos::realloc(implicit_coarse_neighbor_rank_device_, ncoarse, 6);
        Kokkos::realloc(implicit_multilevel_block_parity_, ncoarse, 3);
        implicit_coarse_neighbor_gid_.assign(6*ncoarse, -1);
        implicit_coarse_scaling_.resize(ncoarse);
        implicit_coarse_cholesky_.resize(
            static_cast<std::size_t>(ncoarse)*static_cast<std::size_t>(ncoarse));

        // Each face in the uniform-mesh hierarchy has one neighbor.  Gather that
        // fixed global topology once; subsequent groups need
        // communicate only their six scalar face-conductance sums per MeshBlock.
        const int offsets[6][3] = {
            {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
            {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
        const int active_faces = ppack->pmesh->three_d ? 6 :
            (ppack->pmesh->multi_d ? 4 : 2);
        for (int m = 0; m < ppack->nmb_thispack; ++m) {
          const int gid = ppack->pmb->mb_gid.h_view(m);
          for (int face = 0; face < active_faces; ++face) {
            const int neighbor_index = NeighborIndex(
                offsets[face][0], offsets[face][1], offsets[face][2], 0, 0);
            implicit_coarse_neighbor_gid_[6*gid+face] =
                ppack->pmb->nghbr.h_view(m, neighbor_index).gid;
          }
        }
  #if MPI_PARALLEL_ENABLED
        RadiationBatchAllreduce(MPI_IN_PLACE, implicit_coarse_neighbor_gid_.data(), 6*ncoarse,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  #endif
        for (int gid = 0; gid < ncoarse; ++gid) {
          for (int face = 0; face < active_faces; ++face) {
            const int neighbor = implicit_coarse_neighbor_gid_[6*gid+face];
            if (neighbor >= ncoarse) {
              ImplicitRadiationError(
                  "Invalid MeshBlock neighbor while constructing the implicit-radiation "
                  "coarse topology");
            }
            if (neighbor >= 0 &&
                implicit_coarse_neighbor_gid_[6*neighbor+(face^1)] != gid) {
              ImplicitRadiationError(
                  "Non-reciprocal MeshBlock face in the implicit-radiation coarse "
                  "topology");
            }
            const int neighbor_rank = neighbor < 0 ? -1 :
                ppack->pmesh->rank_eachmb[neighbor];
            if (neighbor_rank >= global_variable::nranks || neighbor_rank < -1 ||
                (neighbor >= 0 && (neighbor <
                    ppack->pmesh->gids_eachrank[neighbor_rank] || neighbor >=
                    ppack->pmesh->gids_eachrank[neighbor_rank]+
                    ppack->pmesh->nmb_eachrank[neighbor_rank]))) {
              ImplicitRadiationError(
                  "Invalid MeshBlock rank/local-index metadata in the implicit-radiation "
                  "coarse topology");
            }
            if (neighbor_rank == global_variable::my_rank &&
                (neighbor < ppack->gids ||
                 neighbor >= ppack->gids+ppack->nmb_thispack)) {
              ImplicitRadiationError(
                  "Same-rank implicit-radiation neighbor is outside the active "
                  "MeshBlockPack");
            }
            implicit_coarse_neighbor_gid_device_.h_view(gid, face) = neighbor;
            implicit_coarse_neighbor_rank_device_.h_view(gid, face) = neighbor_rank;
            if (gid >= ppack->gids && gid < ppack->gids+ppack->nmb_thispack &&
                neighbor_rank >= 0 && neighbor_rank != global_variable::my_rank) {
              implicit_multilevel_has_remote_neighbors_ = true;
            }
          }
          const auto &location = ppack->pmesh->lloc_eachmb[gid];
          const int level_nx1[3] = {
              indcs.nx1, implicit_multilevel_nx1_[0], implicit_multilevel_nx1_[1]};
          const int level_nx2[3] = {
              indcs.nx2, implicit_multilevel_nx2_[0], implicit_multilevel_nx2_[1]};
          const int level_nx3[3] = {
              indcs.nx3, implicit_multilevel_nx3_[0], implicit_multilevel_nx3_[1]};
          for (int level = 0; level < 3; ++level) {
            implicit_multilevel_block_parity_.h_view(gid, level) =
                (location.lx1*level_nx1[level]+location.lx2*level_nx2[level]+
                 location.lx3*level_nx3[level]) & 1;
          }
        }
        implicit_coarse_neighbor_gid_device_.modify_host();
        implicit_coarse_neighbor_rank_device_.modify_host();
        implicit_multilevel_block_parity_.modify_host();
        implicit_coarse_neighbor_gid_device_.sync_device();
        implicit_coarse_neighbor_rank_device_.sync_device();
        implicit_multilevel_block_parity_.sync_device();
      }
    }
  }
}

//----------------------------------------------------------------------------------------

ThermalRadiation::~ThermalRadiation() {
  DestroyImplicitGroupWorkers();
  delete implicit_native_solver_;
  delete implicit_halo_;
  DestroyImplicitAMRAggregateState();
#if MPI_PARALLEL_ENABLED
  if (implicit_multilevel_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&implicit_multilevel_comm_);
  }
#endif
  if (!borrowed_opacity_) {
    if (opacity_table_ != nullptr) delete opacity_table_;
    if (mixed_opacity_table_ != nullptr) delete mixed_opacity_table_;
  }
}

//----------------------------------------------------------------------------------------
//! Initialize every group from a Planck spectrum at the requested radiation temperature.

void ThermalRadiation::Initialize(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                  int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  Real trad_left = initial_radiation_temperature_;
  Real trad_right = initial_radiation_temperature_right_;
  Real xsplit = initial_radiation_x1_;
  int profile = initial_profile_mode_;
  Real arad = arad_;
  auto bounds = group_bounds_.d_view;
  auto diag = diagnostics;
  auto size = pmy_pack_->pmb->mb_size;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is;
  int nx1 = indcs.nx1;

  par_for("thermal_rad_init", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real density = prim(m, IDN, k, j, i);
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real trad = (profile == 1 && x1v >= xsplit) ? trad_right : trad_left;
    Real total = 0.0;
    Real blackbody = arad*trad*trad*trad*trad;
    // Reuse endpoints while preserving the complementary tails of weak groups.
    PlanckSpectrum planck(bounds(0), trad);
    for (int g = 0; g < ng; ++g) {
      const Real fraction = planck.Fraction(bounds(g+1));
      Real eg = blackbody*fraction;
      cons(m, i0+g, k, j, i) = eg;
      prim(m, i0+g, k, j, i) = eg/density;
      total += eg;
    }
    diag(m, 0, k, j, i) = total/density;
    diag(m, 1, k, j, i) = pow(total/arad, 0.25);
  });
}

//----------------------------------------------------------------------------------------
//! Recompute total radiation energy and radiation temperature diagnostics.

void ThermalRadiation::UpdateDiagnostics(const DvceArray5D<Real> &cons,
    const DvceArray5D<Real> &prim, int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  Real arad = arad_;
  auto diag = diagnostics;
  par_for("thermal_rad_diagnostics", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real total = 0.0;
    for (int g = 0; g < ng; ++g) total += fmax(cons(m, i0+g, k, j, i), 0.0);
    diag(m, 0, k, j, i) = total/prim(m, IDN, k, j, i);
    diag(m, 1, k, j, i) = pow(total/arad, 0.25);
  });
}

//----------------------------------------------------------------------------------------
//! Add q_g=-c_hat*D_g*grad(E_g) to each radiation-group finite-volume flux.

void ThermalRadiation::AddFluxes(const DvceArray5D<Real> &w0,
                                 const DvceArray5D<Real> &temperature,
                                 DvceFaceFld5D<Real> &flx, const FaceFldOrigin origin) {
  // In FLASH-like implicit mode the ordinary fluid flux still advects every radiation
  // scalar, while diffusion is applied once as an operator-split backward-Euler solve.
  if (implicit_transport_) return;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size;
  auto kt = kappa_transport_.d_view;
  bool use_table = use_opacity_table_;
  OpacityTableDevice opacity;
  if (use_table) opacity = opacity_table_->DeviceData();
  bool use_mixed_table = use_mixed_opacity_table_;
  MixedOpacityTableDevice mixed_opacity;
  if (use_mixed_table) mixed_opacity = mixed_opacity_table_->DeviceData();
  bool use_materials = use_material_mixture_;
  auto mixture = material_mixture_;
  int iele = iele_;
  Real gm1 = gamma_minus_one_;
  Real fe = cv_e_fraction_;
  Real chat = chat_;
  Real alpha = flux_limit_coefficient_;
  Real floor = energy_floor_;
  int mode = limiter_mode_;
  Real streaming_threshold = ap_streaming_threshold_;
  Real optical_depth_threshold = ap_optical_depth_threshold_;
  bool use_ap_transport = use_ap_transport_;

  auto flx1 = MakeBandView(flx.x1f, origin);
  par_for("thermal_rad_flux1", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real dx1 = size.d_view(m).dx1;
    const Real dx2 = size.d_view(m).dx2;
    const Real dx3 = size.d_view(m).dx3;
    const FLDFaceMaterialState material = X1FaceMaterialState(
        w0, temperature, m, iele, k, j, i, gm1, fe, use_materials, mixture);
    OpacityTableLocation opacity_location;
    MixedOpacityTableLocation mixed_opacity_location;
    if (use_mixed_table) {
      mixed_opacity_location = mixed_opacity.Locate(
          material.density, material.electron_temperature,
          material.composition);
    } else if (use_table) {
      opacity_location = opacity.Locate(
          material.density, material.electron_temperature);
    }
    for (int g = 0; g < ng; ++g) {
      const int n = i0 + g;
      const FLDRadiationFaceState state = X1RadiationFaceState(
          w0, m, n, k, j, i, multi_d, three_d, dx1, dx2, dx3);
      const Real kappa = use_mixed_table ? mixed_opacity.Get(
          opacity_transport, g, mixed_opacity_location) : (use_table ? opacity.Get(
          opacity_transport, g, opacity_location) : kt(g));
      const Real sigma = material.density*kappa;
      const FLDLinearization properties = FLDProperties(
          sigma, state.energy, state.gradient, state.normal_gradient,
          alpha, floor, mode);
      const bool use_ap_face = use_ap_transport && mode != 0 &&
          (properties.streaming_fraction >= streaming_threshold ||
           sigma*dx1 <= optical_depth_threshold);
      flx1(m, n, k, j, i) += FLDNumericalFlux(
          properties, state.normal_gradient, state.energy_left,
          state.energy_right, chat, use_ap_face);
    }
  });
  if (pmy_pack_->pmesh->one_d) return;

  auto flx2 = MakeBandView(flx.x2f, origin);
  par_for("thermal_rad_flux2", DevExeSpace(), 0, nmb1,
          ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real dx1 = size.d_view(m).dx1;
    const Real dx2 = size.d_view(m).dx2;
    const Real dx3 = size.d_view(m).dx3;
    const FLDFaceMaterialState material = X2FaceMaterialState(
        w0, temperature, m, iele, k, j, i, gm1, fe, use_materials, mixture);
    OpacityTableLocation opacity_location;
    MixedOpacityTableLocation mixed_opacity_location;
    if (use_mixed_table) {
      mixed_opacity_location = mixed_opacity.Locate(
          material.density, material.electron_temperature,
          material.composition);
    } else if (use_table) {
      opacity_location = opacity.Locate(
          material.density, material.electron_temperature);
    }
    for (int g = 0; g < ng; ++g) {
      const int n = i0 + g;
      const FLDRadiationFaceState state = X2RadiationFaceState(
          w0, m, n, k, j, i, three_d, dx1, dx2, dx3);
      const Real kappa = use_mixed_table ? mixed_opacity.Get(
          opacity_transport, g, mixed_opacity_location) : (use_table ? opacity.Get(
          opacity_transport, g, opacity_location) : kt(g));
      const Real sigma = material.density*kappa;
      const FLDLinearization properties = FLDProperties(
          sigma, state.energy, state.gradient, state.normal_gradient,
          alpha, floor, mode);
      const bool use_ap_face = use_ap_transport && mode != 0 &&
          (properties.streaming_fraction >= streaming_threshold ||
           sigma*dx2 <= optical_depth_threshold);
      flx2(m, n, k, j, i) += FLDNumericalFlux(
          properties, state.normal_gradient, state.energy_left,
          state.energy_right, chat, use_ap_face);
    }
  });
  if (pmy_pack_->pmesh->two_d) return;

  auto flx3 = MakeBandView(flx.x3f, origin);
  par_for("thermal_rad_flux3", DevExeSpace(), 0, nmb1,
          ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real dx1 = size.d_view(m).dx1;
    const Real dx2 = size.d_view(m).dx2;
    const Real dx3 = size.d_view(m).dx3;
    const FLDFaceMaterialState material = X3FaceMaterialState(
        w0, temperature, m, iele, k, j, i, gm1, fe, use_materials, mixture);
    OpacityTableLocation opacity_location;
    MixedOpacityTableLocation mixed_opacity_location;
    if (use_mixed_table) {
      mixed_opacity_location = mixed_opacity.Locate(
          material.density, material.electron_temperature,
          material.composition);
    } else if (use_table) {
      opacity_location = opacity.Locate(
          material.density, material.electron_temperature);
    }
    for (int g = 0; g < ng; ++g) {
      const int n = i0 + g;
      const FLDRadiationFaceState state = X3RadiationFaceState(
          w0, m, n, k, j, i, dx1, dx2, dx3);
      const Real kappa = use_mixed_table ? mixed_opacity.Get(
          opacity_transport, g, mixed_opacity_location) : (use_table ? opacity.Get(
          opacity_transport, g, opacity_location) : kt(g));
      const Real sigma = material.density*kappa;
      const FLDLinearization properties = FLDProperties(
          sigma, state.energy, state.gradient, state.normal_gradient,
          alpha, floor, mode);
      const bool use_ap_face = use_ap_transport && mode != 0 &&
          (properties.streaming_fraction >= streaming_threshold ||
           sigma*dx3 <= optical_depth_threshold);
      flx3(m, n, k, j, i) += FLDNumericalFlux(
          properties, state.normal_gradient, state.energy_left,
          state.energy_right, chat, use_ap_face);
    }
  });
}

//----------------------------------------------------------------------------------------
//! Fill the scalar solver's physical ghost cells.  Internal and periodic faces are filled
//! by the ordinary cell-centered communicator.  `homogeneous_boundary` supplies the
//! linearized boundary operator required when conjugate gradient applies A to a search
//! direction.

void ThermalRadiation::ApplyImplicitPhysicalBoundaries(
    DvceArray5D<Real> &field, const bool homogeneous_boundary,
    const bool coefficient_field, const bool coarse_grid) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  // The transverse reconstruction needs one coarse donor layer. Filling more
  // can reflect past a tiny coarse interior into another concurrently filled halo.
  const int ng = coarse_grid ? 1 : indcs.ng;
  const int is = coarse_grid ? indcs.cis : indcs.is;
  const int ie = coarse_grid ? indcs.cie : indcs.ie;
  const int js = coarse_grid ? indcs.cjs : indcs.js;
  const int je = coarse_grid ? indcs.cje : indcs.je;
  const int ks = coarse_grid ? indcs.cks : indcs.ks;
  const int ke = coarse_grid ? indcs.cke : indcs.ke;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const int n1 = ie+ng+1;
  const int n2 = (indcs.nx2 > 1) ? je+ng+1 : 1;
  const int n3 = (indcs.nx3 > 1) ? ke+ng+1 : 1;
  auto mb_bcs = pmy_pack_->pmb->mb_bcs;
  auto f = field;

  const int b0 = implicit_boundary_type_[0];
  const int b1 = implicit_boundary_type_[1];
  const Real v0 = homogeneous_boundary ? 0.0 : implicit_boundary_value_[0];
  const Real v1 = homogeneous_boundary ? 0.0 : implicit_boundary_value_[1];
  par_for("thermal_rad_impl_bc_x1", DevExeSpace(), 0, nmb1,
          0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int j, int n) {
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::inner_x1))) {
      if (coefficient_field || b0 == 0) {
        f(m, 0, k, j, is-n-1) = f(m, 0, k, j, is);
      } else if (b0 == 1) {
        f(m, 0, k, j, is-n-1) = 2.0*v0-f(m, 0, k, j, is+n);
      } else {
        f(m, 0, k, j, is-n-1) = 0.0;
      }
    }
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::outer_x1))) {
      if (coefficient_field || b1 == 0) {
        f(m, 0, k, j, ie+n+1) = f(m, 0, k, j, ie);
      } else if (b1 == 1) {
        f(m, 0, k, j, ie+n+1) = 2.0*v1-f(m, 0, k, j, ie-n);
      } else {
        f(m, 0, k, j, ie+n+1) = 0.0;
      }
    }
  });
  if (pmy_pack_->pmesh->one_d) return;

  const int b2 = implicit_boundary_type_[2];
  const int b3 = implicit_boundary_type_[3];
  const Real v2 = homogeneous_boundary ? 0.0 : implicit_boundary_value_[2];
  const Real v3 = homogeneous_boundary ? 0.0 : implicit_boundary_value_[3];
  par_for("thermal_rad_impl_bc_x2", DevExeSpace(), 0, nmb1,
          0, n3-1, 0, n1-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int i, int n) {
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::inner_x2))) {
      if (coefficient_field || b2 == 0) {
        f(m, 0, k, js-n-1, i) = f(m, 0, k, js, i);
      } else if (b2 == 1) {
        f(m, 0, k, js-n-1, i) = 2.0*v2-f(m, 0, k, js+n, i);
      } else {
        f(m, 0, k, js-n-1, i) = 0.0;
      }
    }
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::outer_x2))) {
      if (coefficient_field || b3 == 0) {
        f(m, 0, k, je+n+1, i) = f(m, 0, k, je, i);
      } else if (b3 == 1) {
        f(m, 0, k, je+n+1, i) = 2.0*v3-f(m, 0, k, je-n, i);
      } else {
        f(m, 0, k, je+n+1, i) = 0.0;
      }
    }
  });
  if (pmy_pack_->pmesh->two_d) return;

  const int b4 = implicit_boundary_type_[4];
  const int b5 = implicit_boundary_type_[5];
  const Real v4 = homogeneous_boundary ? 0.0 : implicit_boundary_value_[4];
  const Real v5 = homogeneous_boundary ? 0.0 : implicit_boundary_value_[5];
  par_for("thermal_rad_impl_bc_x3", DevExeSpace(), 0, nmb1,
          0, n2-1, 0, n1-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int j, int i, int n) {
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::inner_x3))) {
      if (coefficient_field || b4 == 0) {
        f(m, 0, ks-n-1, j, i) = f(m, 0, ks, j, i);
      } else if (b4 == 1) {
        f(m, 0, ks-n-1, j, i) = 2.0*v4-f(m, 0, ks+n, j, i);
      } else {
        f(m, 0, ks-n-1, j, i) = 0.0;
      }
    }
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::outer_x3))) {
      if (coefficient_field || b5 == 0) {
        f(m, 0, ke+n+1, j, i) = f(m, 0, ke, j, i);
      } else if (b5 == 1) {
        f(m, 0, ke+n+1, j, i) = 2.0*v5-f(m, 0, ke-n, j, i);
      } else {
        f(m, 0, ke+n+1, j, i) = 0.0;
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! Exchange one scalar with the donor stencil needed by the current operator.

void ThermalRadiation::ExchangeImplicitField(
    DvceArray5D<Real> &field, MeshBoundaryValuesCC *pbval,
    const bool homogeneous_boundary, const bool coefficient_field, const bool inner_halo) {
  RadiationTimerScope phase_timer(solver_timers_, RadiationTimer::Exchange);
  const bool multilevel = pmy_pack_->pmesh->multilevel;
  const bool compact = implicit_halo_ != nullptr && implicit_halo_compact_;
  const bool faces_only = implicit_halo_ != nullptr && implicit_halo_faces_ &&
      (inner_halo || (compact && !implicit_amr_high_order_));
  const int width = compact ? (implicit_amr_high_order_ ? 2 : 1) : 0;
  if (multilevel) {
    // Scalar fields reserve the maximum AMR block capacity. RestrictCC uses its
    // input extent, so pass a contiguous view of only the currently active blocks.
    DvceArray5D<Real> active_field = Kokkos::subview(field,
        std::make_pair(0, pmy_pack_->nmb_thispack), Kokkos::ALL, Kokkos::ALL,
        Kokkos::ALL, Kokkos::ALL);
    pmy_pack_->pmesh->pmr->RestrictCC(active_field, implicit_coarse_scratch_);
  }
  if (implicit_halo_ != nullptr) {
    implicit_halo_->Exchange(field, implicit_coarse_scratch_, faces_only, width);
  } else {
    if (pbval->InitRecv(1) != TaskStatus::complete) {
      ImplicitRadiationError("Could not initialize implicit-radiation halo receives");
    }
    if (pbval->PackAndSendCC(field, implicit_coarse_scratch_, 1) != TaskStatus::complete) {
      ImplicitRadiationError("Could not send implicit-radiation halo data");
    }
    TaskStatus status = TaskStatus::incomplete;
    while (status == TaskStatus::incomplete) {
      status = pbval->RecvAndUnpackCC(field, implicit_coarse_scratch_, 1);
    }
    if (status != TaskStatus::complete) {
      ImplicitRadiationError("Could not receive implicit-radiation halo data");
    }
    if (pbval->ClearSend() != TaskStatus::complete ||
        pbval->ClearRecv() != TaskStatus::complete) {
      ImplicitRadiationError("Could not clear implicit-radiation halo communication");
    }
  }
  if (multilevel) {
    if (implicit_amr_high_order_) {
      // Complete the transverse coarse donors at block edges/corners exactly as
      // for fluid prolongation, but retain a fixed linear radiation stencil.
      pbval->FillCoarseInBndryCC(field, implicit_coarse_scratch_, false, compact ? 1 : -1);
      ApplyImplicitPhysicalBoundaries(implicit_coarse_scratch_,
                                       homogeneous_boundary, coefficient_field, true);
    }
    // Constant injection is linear and couples each fine face to its actual
    // coarse cell.  Slope-limited fluid prolongation would make the CG operator
    // nonlinear and would destroy symmetry across refinement boundaries.
    const auto indcs = pmy_pack_->pmesh->mb_indcs;
    const bool multi_d = pmy_pack_->pmesh->multi_d;
    const bool three_d = pmy_pack_->pmesh->three_d;
    const int nnghbr = pmy_pack_->pmb->nnghbr;
    auto nghbr = pmy_pack_->pmb->nghbr;
    auto level = pmy_pack_->pmb->mb_lev;
    auto &rbuf = pbval->recvbuf;
    auto coarse = implicit_coarse_scratch_;
    auto fine = field;
    Kokkos::TeamPolicy<> policy(
        DevExeSpace(), pmy_pack_->nmb_thispack*nnghbr, Kokkos::AUTO);
    Kokkos::parallel_for("thermal_rad_impl_inject", policy,
    KOKKOS_LAMBDA(TeamMember_t team) {
      const int m = team.league_rank()/nnghbr;
      const int n = team.league_rank()%nnghbr;
      if (nghbr.d_view(m, n).gid >= 0 &&
          (!faces_only || ImplicitRadiationHalo::IsFace(n)) &&
          nghbr.d_view(m, n).lev < level.d_view(m)) {
        auto bounds = rbuf[n].iprol[0];
        if (compact) {
          bounds = ImplicitRadiationHalo::CompactBounds(
              bounds, indcs, true, faces_only, 1);
        }
        const int ni = bounds.bie-bounds.bis+1;
        const int nj = bounds.bje-bounds.bjs+1;
        const int nk = bounds.bke-bounds.bks+1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, ni*nj*nk),
        [&](const int idx) {
          const int ci = idx%ni+bounds.bis;
          const int cj = (idx/ni)%nj+bounds.bjs;
          const int ck = idx/(ni*nj)+bounds.bks;
          const int fi = 2*(ci-indcs.cis)+indcs.is;
          const int fj = multi_d ? 2*(cj-indcs.cjs)+indcs.js : 0;
          const int fk = three_d ? 2*(ck-indcs.cks)+indcs.ks : 0;
          for (int dk = 0; dk < (three_d ? 2 : 1); ++dk) {
            for (int dj = 0; dj < (multi_d ? 2 : 1); ++dj) {
              for (int di = 0; di < 2; ++di) {
                fine(m, 0, fk+dk, fj+dj, fi+di) = coarse(m, 0, ck, cj, ci);
              }
            }
          }
        });
      }
    });
  }
  ApplyImplicitPhysicalBoundaries(
      field, homogeneous_boundary, coefficient_field);
}

//----------------------------------------------------------------------------------------
//! Apply A=I-dt*div(K*grad), with the time-lagged FLD coefficient K=c*lambda/sigma.

void ThermalRadiation::ApplyImplicitOperator(
    const DvceArray5D<Real> &field, DvceArray5D<Real> &result, const Real dt) {
  RadiationTimerScope phase_timer(solver_timers_, RadiationTimer::Matvec);
  if (pmy_pack_->pmesh->multilevel) {
    ApplyImplicitAMROperator(field, result, dt);
    return;
  }
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size;
  auto mb_bcs = pmy_pack_->pmb->mb_bcs;
  const int b0 = implicit_boundary_type_[0];
  const int b1 = implicit_boundary_type_[1];
  const int b2 = implicit_boundary_type_[2];
  const int b3 = implicit_boundary_type_[3];
  const int b4 = implicit_boundary_type_[4];
  const int b5 = implicit_boundary_type_[5];
  const Real vacuum_cap_coefficient = 0.5*chat_*flux_limit_coefficient_;
  auto coefficient = implicit_coefficient_;
  auto f = field;
  auto out = result;

  par_for("thermal_rad_impl_operator", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real kc = coefficient(m, 0, k, j, i);
    const Real dx1 = size.d_view(m).dx1;
    const bool vacuum_p1 = i == ie && b1 == 2 && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::outer_x1));
    const bool vacuum_m1 = i == is && b0 == 2 && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::inner_x1));
    const Real kp1 = ImplicitFrozenFaceCoefficient(
        kc, coefficient(m, 0, k, j, i+1), vacuum_p1,
        vacuum_cap_coefficient*dx1);
    const Real km1 = ImplicitFrozenFaceCoefficient(
        kc, coefficient(m, 0, k, j, i-1), vacuum_m1,
        vacuum_cap_coefficient*dx1);
    Real lap = (kp1*(f(m, 0, k, j, i+1)-f(m, 0, k, j, i))-
                km1*(f(m, 0, k, j, i)-f(m, 0, k, j, i-1)))/(dx1*dx1);
    if (multi_d) {
      const Real dx2 = size.d_view(m).dx2;
      const bool vacuum_p2 = j == je && b3 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x2));
      const bool vacuum_m2 = j == js && b2 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x2));
      const Real kp2 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k, j+1, i), vacuum_p2,
          vacuum_cap_coefficient*dx2);
      const Real km2 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k, j-1, i), vacuum_m2,
          vacuum_cap_coefficient*dx2);
      lap += (kp2*(f(m, 0, k, j+1, i)-f(m, 0, k, j, i))-
              km2*(f(m, 0, k, j, i)-f(m, 0, k, j-1, i)))/(dx2*dx2);
    }
    if (three_d) {
      const Real dx3 = size.d_view(m).dx3;
      const bool vacuum_p3 = k == ke && b5 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x3));
      const bool vacuum_m3 = k == ks && b4 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x3));
      const Real kp3 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k+1, j, i), vacuum_p3,
          vacuum_cap_coefficient*dx3);
      const Real km3 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k-1, j, i), vacuum_m3,
          vacuum_cap_coefficient*dx3);
      lap += (kp3*(f(m, 0, k+1, j, i)-f(m, 0, k, j, i))-
              km3*(f(m, 0, k, j, i)-f(m, 0, k-1, j, i)))/(dx3*dx3);
    }
    out(m, 0, k, j, i) = f(m, 0, k, j, i)-dt*lap;
  });
}

//----------------------------------------------------------------------------------------

Real ThermalRadiation::ImplicitGlobalDot(
    const DvceArray5D<Real> &lhs, const DvceArray5D<Real> &rhs) const {
  RadiationTimerScope phase_timer(solver_timers_, RadiationTimer::Dot);
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pmy_pack_->nmb_thispack*nkji;
  const bool multilevel = pmy_pack_->pmesh->multilevel;
  auto size = pmy_pack_->pmb->mb_size;
  auto a = lhs;
  auto b = rhs;
  Real sum = 0.0;
  Kokkos::parallel_reduce("thermal_rad_impl_dot",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int idx, Real &local_sum) {
    const int m = idx/nkji;
    const int local = idx-m*nkji;
    const int k = local/nji+ks;
    const int j = (local-(k-ks)*nji)/nx1+js;
    const int i = local-(k-ks)*nji-(j-js)*nx1+is;
    // The composite finite-volume operator is symmetric in the cell-volume
    // inner product.  Retain the existing normalization on uniform meshes.
    const Real volume = multilevel ?
        size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3 : 1.0;
    local_sum += volume*a(m, 0, k, j, i)*b(m, 0, k, j, i);
  }, sum);
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    RadiationBatchAllreduce(MPI_IN_PLACE, &sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  return sum;
}

//----------------------------------------------------------------------------------------
//! Return max_i |b-Ax|_i/(|b_i|+(|A||x|)_i).  For the diffusion part, each
//! face contributes |K_face*x_i|+|K_face*x_neighbor|.  Keeping these matrix
//! terms separate is essential: using |K_face*(x_neighbor-x_i)| would cancel
//! the very large opposing terms whose roundoff this backward error measures.
//! The caller must first exchange `field`, so this uses exactly the same internal,
//! periodic, and physical ghosts as ApplyImplicitOperator().

Real ThermalRadiation::ImplicitComponentwiseBackwardError(
    const DvceArray5D<Real> &field, const DvceArray5D<Real> &rhs,
    const DvceArray5D<Real> &residual, const Real dt) {
  RadiationTimerScope phase_timer(solver_timers_, RadiationTimer::ResidualCheck);
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pmy_pack_->nmb_thispack*nkji;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size;
  auto mb_bcs = pmy_pack_->pmb->mb_bcs;
  const int b0 = implicit_boundary_type_[0];
  const int b1 = implicit_boundary_type_[1];
  const int b2 = implicit_boundary_type_[2];
  const int b3 = implicit_boundary_type_[3];
  const int b4 = implicit_boundary_type_[4];
  const int b5 = implicit_boundary_type_[5];
  const Real vacuum_cap_coefficient = 0.5*chat_*flux_limit_coefficient_;
  const Real invalid_error = std::numeric_limits<Real>::max();
  auto coefficient = implicit_coefficient_;
  auto f = field;
  auto b = rhs;
  auto r = residual;

  if (pmy_pack_->pmesh->multilevel) {
    if (implicit_amr_high_order_) {
      // Bound |A_ii| from below without summing signed reconstruction terms.
      // A fine interface contributes half its positive face conductance to
      // the diagonal. On the coarse side the central trace has weight one;
      // all transverse off-center weights sum in magnitude to at most
      // (1+1/8+1/8)^2-1 = 9/16. Even if restriction or physical reflection
      // folds every one back onto this cell with the adverse sign, at least
      // 7/16 remains. Same-level and physical faces have their exact diagonal
      // weights in implicit_amr_diagonal_. Thus 1+7/16*(D-1) <= A_ii.
      // Keeping only this diagonal term underestimates |A||x|+|b_linear|,
      // giving an UPPER bound on the componentwise backward error. Omitting
      // b also handles affine (inhomogeneous Dirichlet) boundary terms.
      // Unlike |Ax|, this denominator retains the stiff terms that cancel in
      // a nearly constant field. GMRES must still meet its projected target
      // before this roundoff-aware criterion can accept an iterate.
      auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
      Real maximum_error = 0.0;
      Kokkos::parallel_reduce("thermal_rad_impl_amr_residual_bound",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
      KOKKOS_LAMBDA(const int idx, Real &local_maximum) {
        const int m = idx/nkji;
        const int local = idx-m*nkji;
        const int k = local/nji+ks;
        const int j = (local-(k-ks)*nji)/nx1+js;
        const int i = local-(k-ks)*nji-(j-js)*nx1+is;
        const Real numerator = fabs(r(m, 0, k, j, i));
        const Real diagonal_lower = 1.0+0.4375*(diagonal(m, 0, k, j, i)-1.0);
        const Real denominator = diagonal_lower*fabs(f(m, 0, k, j, i));
        Real error = invalid_error;
        if (Kokkos::isfinite(numerator) && Kokkos::isfinite(denominator)) {
          if (denominator > 0.0) error = numerator/denominator;
          else if (numerator == 0.0) error = 0.0;
        }
        local_maximum = fmax(local_maximum, error);
      }, Kokkos::Max<Real>(maximum_error));
#if MPI_PARALLEL_ENABLED
      RadiationBatchAllreduce(MPI_IN_PLACE, &maximum_error, 1, MPI_ATHENA_REAL,
                    MPI_MAX, MPI_COMM_WORLD);
#endif
      return maximum_error;
    }
    // Sum positive face contributions before restricting them, just as in the
    // actual operator.  Coarse ghost averages alone lose coefficient weighting.
    ApplyImplicitAMROperator(field, implicit_coarse_scratch_, dt, true);
    auto absolute_action = implicit_coarse_scratch_;
    Real maximum_error = 0.0;
    Kokkos::parallel_reduce("thermal_rad_impl_amr_backward_error",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(const int idx, Real &local_maximum) {
      const int m = idx/nkji;
      const int local = idx-m*nkji;
      const int k = local/nji+ks;
      const int j = (local-(k-ks)*nji)/nx1+js;
      const int i = local-(k-ks)*nji-(j-js)*nx1+is;
      const Real numerator = fabs(r(m, 0, k, j, i));
      const Real denominator = fabs(b(m, 0, k, j, i))+
          absolute_action(m, 0, k, j, i);
      Real error = invalid_error;
      if (Kokkos::isfinite(numerator) && Kokkos::isfinite(denominator) &&
          denominator >= 0.0) {
        if (denominator > 0.0) error = numerator/denominator;
        else if (numerator == 0.0) error = 0.0;
      }
      local_maximum = fmax(local_maximum, error);
    }, Kokkos::Max<Real>(maximum_error));
#if MPI_PARALLEL_ENABLED
    RadiationBatchAllreduce(MPI_IN_PLACE, &maximum_error, 1, MPI_ATHENA_REAL,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    return maximum_error;
  }

  Real maximum_error = 0.0;
  Kokkos::parallel_reduce("thermal_rad_impl_backward_error",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int idx, Real &local_maximum) {
    const int m = idx/nkji;
    const int local = idx-m*nkji;
    const int k = local/nji+ks;
    const int j = (local-(k-ks)*nji)/nx1+js;
    const int i = local-(k-ks)*nji-(j-js)*nx1+is;
    const Real center = f(m, 0, k, j, i);
    const Real kc = coefficient(m, 0, k, j, i);
    const Real dx1 = size.d_view(m).dx1;
    const bool vacuum_p1 = i == ie && b1 == 2 && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::outer_x1));
    const bool vacuum_m1 = i == is && b0 == 2 && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::inner_x1));
    const Real kp1 = ImplicitFrozenFaceCoefficient(
        kc, coefficient(m, 0, k, j, i+1), vacuum_p1,
        vacuum_cap_coefficient*dx1);
    const Real km1 = ImplicitFrozenFaceCoefficient(
        kc, coefficient(m, 0, k, j, i-1), vacuum_m1,
        vacuum_cap_coefficient*dx1);
    Real face_sum = (fabs(kp1*center)+fabs(kp1*f(m, 0, k, j, i+1))+
                     fabs(km1*center)+fabs(km1*f(m, 0, k, j, i-1)))/(dx1*dx1);
    if (multi_d) {
      const Real dx2 = size.d_view(m).dx2;
      const bool vacuum_p2 = j == je && b3 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x2));
      const bool vacuum_m2 = j == js && b2 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x2));
      const Real kp2 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k, j+1, i), vacuum_p2,
          vacuum_cap_coefficient*dx2);
      const Real km2 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k, j-1, i), vacuum_m2,
          vacuum_cap_coefficient*dx2);
      face_sum += (fabs(kp2*center)+fabs(kp2*f(m, 0, k, j+1, i))+
                   fabs(km2*center)+fabs(km2*f(m, 0, k, j-1, i)))/(dx2*dx2);
    }
    if (three_d) {
      const Real dx3 = size.d_view(m).dx3;
      const bool vacuum_p3 = k == ke && b5 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x3));
      const bool vacuum_m3 = k == ks && b4 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x3));
      const Real kp3 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k+1, j, i), vacuum_p3,
          vacuum_cap_coefficient*dx3);
      const Real km3 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k-1, j, i), vacuum_m3,
          vacuum_cap_coefficient*dx3);
      face_sum += (fabs(kp3*center)+fabs(kp3*f(m, 0, k+1, j, i))+
                   fabs(km3*center)+fabs(km3*f(m, 0, k-1, j, i)))/(dx3*dx3);
    }
    const Real numerator = fabs(r(m, 0, k, j, i));
    const Real denominator = fabs(b(m, 0, k, j, i))+fabs(center)+dt*face_sum;
    Real error = 0.0;
    if (!Kokkos::isfinite(numerator) || !Kokkos::isfinite(denominator) ||
        denominator < 0.0) {
      error = invalid_error;
    } else if (denominator > 0.0) {
      error = numerator/denominator;
    } else if (numerator > 0.0) {
      error = invalid_error;
    }
    local_maximum = fmax(local_maximum, error);
  }, Kokkos::Max<Real>(maximum_error));
#if MPI_PARALLEL_ENABLED
  RadiationBatchAllreduce(MPI_IN_PLACE, &maximum_error, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
#endif
  return maximum_error;
}

//----------------------------------------------------------------------------------------
//! Assemble and factor E=Z^T A Z for piecewise-constant MeshBlock basis functions.
//! Internal faces cancel exactly.  Consequently the coarse matrix needs only the mass
//! term, inter-MeshBlock face conductances, and homogeneous physical-boundary terms.

void ThermalRadiation::BuildImplicitBlockCoarsePreconditioner(const Real dt) {
  RadiationTimerScope phase_timer(solver_timers_, RadiationTimer::PreconditionerSetup);
  if (implicit_preconditioner_mode_ == 5) {
    if (implicit_native_solver_ == nullptr) {
      implicit_native_solver_ = new diffusion::NativeDiffusionSolver;
    }
    implicit_native_solver_->Build(pmy_pack_, implicit_coefficient_,
        implicit_amr_conductance_, implicit_amr_face_origin_, DvceArray5D<Real>{}, dt,
        implicit_boundary_type_, 0.5*chat_*flux_limit_coefficient_,
        implicit_native_device_mpi_, implicit_native_max_bytes_);
    if (solver_timers_.enabled) {
      std::cout << "# radiation native multigrid: rank=" << global_variable::my_rank
                << " levels=" << implicit_native_solver_->levels()
                << " device_bytes=" << implicit_native_solver_->bytes() << std::endl;
    }
    return;
  }
  if (pmy_pack_->pmesh->multilevel) {
    if (implicit_preconditioner_mode_ == 1) BuildImplicitAMRBlockCoarsePreconditioner(dt);
    if (implicit_preconditioner_mode_ >= 2) BuildImplicitAMRAggregatePreconditioner(dt);
    return;
  }
  if (implicit_preconditioner_mode_ != 1 || !implicit_multilevel_enabled_) return;

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack_->nmb_thispack;
  const int gid0 = pmy_pack_->gids;
  const int ncoarse = pmy_pack_->pmesh->nmb_total;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  auto coefficient = implicit_coefficient_;
  auto face_sums = implicit_coarse_faces_.d_view;
  auto size = pmy_pack_->pmb->mb_size;
  auto mb_bcs = pmy_pack_->pmb->mb_bcs;
  const int b0 = implicit_boundary_type_[0];
  const int b1 = implicit_boundary_type_[1];
  const int b2 = implicit_boundary_type_[2];
  const int b3 = implicit_boundary_type_[3];
  const int b4 = implicit_boundary_type_[4];
  const int b5 = implicit_boundary_type_[5];
  const Real vacuum_cap_coefficient = 0.5*chat_*flux_limit_coefficient_;

  // Build the diagonals and cache face conductances of the two Galerkin matrices
  // once per frozen group operator, including its current dt and boundary caps.
  // A coarse basis function is one on
  // its aggregate and zero elsewhere, so its mass is the aggregate volume and only fine
  // faces on the aggregate boundary survive P^T A P.  The global V-cycle applies the
  // matching inter-MeshBlock off-diagonals after each level's face exchange.
  if (implicit_multilevel_enabled_) {
    auto multilevel_diagonal = implicit_multilevel_vector_;
    auto conductance = implicit_multilevel_conductance_;
    for (int level = 0; level < 2; ++level) {
      const int cnx1 = implicit_multilevel_nx1_[level];
      const int cnx2 = implicit_multilevel_nx2_[level];
      const int cnx3 = implicit_multilevel_nx3_[level];
      const int aggregate1 = indcs.nx1/cnx1;
      const int aggregate2 = indcs.nx2/cnx2;
      const int aggregate3 = indcs.nx3/cnx3;
      const int level_offset = implicit_multilevel_offset_[level];
      par_for("thermal_rad_impl_multilevel_diagonal", DevExeSpace(), 0, nmb-1,
              0, cnx3-1, 0, cnx2-1, 0, cnx1-1,
      KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
        const int packed = level_offset+(ck*cnx2+cj)*cnx1+ci;
        const bool physical_m1 = ci == 0 && IsPhysicalBoundary(
            mb_bcs.d_view(m, BoundaryFace::inner_x1));
        const bool physical_p1 = ci == cnx1-1 && IsPhysicalBoundary(
            mb_bcs.d_view(m, BoundaryFace::outer_x1));
        const Real dx1 = size.d_view(m).dx1;
        const Real gm1 = ImplicitAggregateFaceConductance(
            coefficient, m, 0, -1, ci, cj, ck,
            aggregate1, aggregate2, aggregate3, is, js, ks, dt, dx1,
            physical_m1 && b0 == 2, vacuum_cap_coefficient*dx1);
        const Real gp1 = ImplicitAggregateFaceConductance(
            coefficient, m, 0, 1, ci, cj, ck,
            aggregate1, aggregate2, aggregate3, is, js, ks, dt, dx1,
            physical_p1 && b1 == 2, vacuum_cap_coefficient*dx1);
        conductance(m, 0, packed) = gm1;
        conductance(m, 1, packed) = gp1;
        Real diagonal = static_cast<Real>(aggregate1*aggregate2*aggregate3)+
            ImplicitFaceDiagonalWeight(physical_m1, b0)*gm1+
            ImplicitFaceDiagonalWeight(physical_p1, b1)*gp1;
        if (multi_d) {
          const bool physical_m2 = cj == 0 && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::inner_x2));
          const bool physical_p2 = cj == cnx2-1 && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::outer_x2));
          const Real dx2 = size.d_view(m).dx2;
          const Real gm2 = ImplicitAggregateFaceConductance(
              coefficient, m, 1, -1, ci, cj, ck,
              aggregate1, aggregate2, aggregate3, is, js, ks, dt, dx2,
              physical_m2 && b2 == 2, vacuum_cap_coefficient*dx2);
          const Real gp2 = ImplicitAggregateFaceConductance(
              coefficient, m, 1, 1, ci, cj, ck,
              aggregate1, aggregate2, aggregate3, is, js, ks, dt, dx2,
              physical_p2 && b3 == 2, vacuum_cap_coefficient*dx2);
          conductance(m, 2, packed) = gm2;
          conductance(m, 3, packed) = gp2;
          diagonal += ImplicitFaceDiagonalWeight(physical_m2, b2)*gm2+
              ImplicitFaceDiagonalWeight(physical_p2, b3)*gp2;
        }
        if (three_d) {
          const bool physical_m3 = ck == 0 && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::inner_x3));
          const bool physical_p3 = ck == cnx3-1 && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::outer_x3));
          const Real dx3 = size.d_view(m).dx3;
          const Real gm3 = ImplicitAggregateFaceConductance(
              coefficient, m, 2, -1, ci, cj, ck,
              aggregate1, aggregate2, aggregate3, is, js, ks, dt, dx3,
              physical_m3 && b4 == 2, vacuum_cap_coefficient*dx3);
          const Real gp3 = ImplicitAggregateFaceConductance(
              coefficient, m, 2, 1, ci, cj, ck,
              aggregate1, aggregate2, aggregate3, is, js, ks, dt, dx3,
              physical_p3 && b5 == 2, vacuum_cap_coefficient*dx3);
          conductance(m, 4, packed) = gm3;
          conductance(m, 5, packed) = gp3;
          diagonal += ImplicitFaceDiagonalWeight(physical_m3, b4)*gm3+
              ImplicitFaceDiagonalWeight(physical_p3, b5)*gp3;
        }
        multilevel_diagonal(m, packed) = diagonal;
      });
    }
  }

  Kokkos::deep_copy(DevExeSpace(), face_sums, 0.0);
  Kokkos::parallel_for("thermal_rad_impl_coarse_faces_x1",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
          {0, ks, js}, {nmb, ke+1, je+1}),
  KOKKOS_LAMBDA(int m, int k, int j) {
    const int gid = gid0+m;
    const Real dx = size.d_view(m).dx1;
    const Real scale = dt/(dx*dx);
    const bool vacuum_m = b0 == 2 && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::inner_x1));
    const bool vacuum_p = b1 == 2 && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::outer_x1));
    const Real km = ImplicitFrozenFaceCoefficient(
        coefficient(m, 0, k, j, is), coefficient(m, 0, k, j, is-1),
        vacuum_m, vacuum_cap_coefficient*dx);
    const Real kp = ImplicitFrozenFaceCoefficient(
        coefficient(m, 0, k, j, ie), coefficient(m, 0, k, j, ie+1),
        vacuum_p, vacuum_cap_coefficient*dx);
    Kokkos::atomic_add(&face_sums(gid, 0), scale*km);
    Kokkos::atomic_add(&face_sums(gid, 1), scale*kp);
  });
  if (multi_d) {
    Kokkos::parallel_for("thermal_rad_impl_coarse_faces_x2",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, ks, is}, {nmb, ke+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int i) {
      const int gid = gid0+m;
      const Real dx = size.d_view(m).dx2;
      const Real scale = dt/(dx*dx);
      const bool vacuum_m = b2 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x2));
      const bool vacuum_p = b3 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x2));
      const Real km = ImplicitFrozenFaceCoefficient(
          coefficient(m, 0, k, js, i), coefficient(m, 0, k, js-1, i),
          vacuum_m, vacuum_cap_coefficient*dx);
      const Real kp = ImplicitFrozenFaceCoefficient(
          coefficient(m, 0, k, je, i), coefficient(m, 0, k, je+1, i),
          vacuum_p, vacuum_cap_coefficient*dx);
      Kokkos::atomic_add(&face_sums(gid, 2), scale*km);
      Kokkos::atomic_add(&face_sums(gid, 3), scale*kp);
    });
  }
  if (three_d) {
    Kokkos::parallel_for("thermal_rad_impl_coarse_faces_x3",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, js, is}, {nmb, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int j, int i) {
      const int gid = gid0+m;
      const Real dx = size.d_view(m).dx3;
      const Real scale = dt/(dx*dx);
      const bool vacuum_m = b4 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x3));
      const bool vacuum_p = b5 == 2 && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x3));
      const Real km = ImplicitFrozenFaceCoefficient(
          coefficient(m, 0, ks, j, i), coefficient(m, 0, ks-1, j, i),
          vacuum_m, vacuum_cap_coefficient*dx);
      const Real kp = ImplicitFrozenFaceCoefficient(
          coefficient(m, 0, ke, j, i), coefficient(m, 0, ke+1, j, i),
          vacuum_p, vacuum_cap_coefficient*dx);
      Kokkos::atomic_add(&face_sums(gid, 4), scale*km);
      Kokkos::atomic_add(&face_sums(gid, 5), scale*kp);
    });
  }
  implicit_coarse_faces_.template modify<DevExeSpace>();
  implicit_coarse_faces_.template sync<HostMemSpace>();
  auto host_faces = implicit_coarse_faces_.h_view;
#if MPI_PARALLEL_ENABLED
  RadiationBatchAllreduce(MPI_IN_PLACE, host_faces.data(), 6*ncoarse, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif

  auto &factor = implicit_coarse_cholesky_;
  std::fill(factor.begin(), factor.end(), 0.0);
  const Real block_cells = static_cast<Real>(
      indcs.nx1*indcs.nx2*indcs.nx3);
  for (int gid = 0; gid < ncoarse; ++gid) {
    factor[static_cast<std::size_t>(gid)*ncoarse+gid] = block_cells;
  }

  const int active_faces = three_d ? 6 : (multi_d ? 4 : 2);
  for (int gid = 0; gid < ncoarse; ++gid) {
    for (int face = 0; face < active_faces; ++face) {
      const Real local_conductance = host_faces(gid, face);
      if (!std::isfinite(local_conductance) || local_conductance < 0.0) {
        ImplicitRadiationError(
            "Implicit-radiation coarse face conductance is negative or non-finite");
      }
      const int neighbor = implicit_coarse_neighbor_gid_[6*gid+face];
      if (neighbor >= 0) {
        // A periodic face may connect a one-block direction to itself.  Such a face
        // annihilates a block-constant vector and contributes nothing to E.
        if (neighbor == gid || gid > neighbor) continue;
        const Real neighbor_conductance = host_faces(neighbor, face^1);
        if (!std::isfinite(neighbor_conductance) ||
            neighbor_conductance < 0.0) {
          ImplicitRadiationError(
              "Implicit-radiation peer face conductance is negative or non-finite");
        }
        // Both oriented sums represent the same shared faces.  Permit reduction-order
        // roundoff, but reject a discrepancy large enough to indicate inconsistent
        // coefficient halos or topology before symmetrizing the coarse matrix.
        const Real face_scale = std::max(
            std::max(local_conductance, neighbor_conductance),
            std::numeric_limits<Real>::min());
        const Real mismatch = std::abs(local_conductance-neighbor_conductance);
        if (mismatch > 8192.0*kRealEpsilon*face_scale) {
          ImplicitRadiationError(
              "Implicit-radiation peer face sums disagree for MeshBlock "+
              std::to_string(gid)+", face "+std::to_string(face));
        }
        const Real conductance = 0.5*(local_conductance+neighbor_conductance);
        const std::size_t gg = static_cast<std::size_t>(gid)*ncoarse+gid;
        const std::size_t nn = static_cast<std::size_t>(neighbor)*ncoarse+neighbor;
        const std::size_t gn = static_cast<std::size_t>(gid)*ncoarse+neighbor;
        const std::size_t ng = static_cast<std::size_t>(neighbor)*ncoarse+gid;
        factor[gg] += conductance;
        factor[nn] += conductance;
        factor[gn] -= conductance;
        factor[ng] -= conductance;
      } else {
        const Real boundary_weight = implicit_boundary_type_[face] == 0 ? 0.0 :
            (implicit_boundary_type_[face] == 1 ? 2.0 : 1.0);
        factor[static_cast<std::size_t>(gid)*ncoarse+gid] +=
            boundary_weight*local_conductance;
      }
    }
  }

  // Symmetric diagonal equilibration protects the mass term when the diffusion
  // conductance is many orders of magnitude larger.  If B=S E S and B=L L^T, then
  // E^-1 = S L^-T L^-1 S.
  auto &scaling = implicit_coarse_scaling_;
  for (int row = 0; row < ncoarse; ++row) {
    const Real diagonal = factor[static_cast<std::size_t>(row)*ncoarse+row];
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
      ImplicitRadiationError(
          "Implicit-radiation coarse matrix has a non-positive or non-finite diagonal");
    }
    scaling[row] = 1.0/std::sqrt(diagonal);
  }
  for (int row = 0; row < ncoarse; ++row) {
    for (int col = 0; col < ncoarse; ++col) {
      factor[static_cast<std::size_t>(row)*ncoarse+col] *=
          scaling[row]*scaling[col];
    }
  }

  // Dense Cholesky is inexpensive for the intended 343-block DCI mesh, avoids another
  // iterative tolerance inside PCG, and is redundantly reproducible on every rank.
  for (int row = 0; row < ncoarse; ++row) {
    for (int col = 0; col <= row; ++col) {
      Real entry = factor[static_cast<std::size_t>(row)*ncoarse+col];
      for (int k = 0; k < col; ++k) {
        entry -= factor[static_cast<std::size_t>(row)*ncoarse+k]*
                 factor[static_cast<std::size_t>(col)*ncoarse+k];
      }
      if (row == col) {
        if (!(entry > 0.0) || !std::isfinite(entry)) {
          ImplicitRadiationError(
              "Implicit-radiation coarse Cholesky factorization failed at pivot "+
              std::to_string(row)+" (value="+std::to_string(entry)+")");
        }
        factor[static_cast<std::size_t>(row)*ncoarse+col] = std::sqrt(entry);
      } else {
        const Real pivot = factor[static_cast<std::size_t>(col)*ncoarse+col];
        const Real value = entry/pivot;
        if (!std::isfinite(value)) {
          ImplicitRadiationError(
              "Implicit-radiation coarse Cholesky factor contains a non-finite value");
        }
        factor[static_cast<std::size_t>(row)*ncoarse+col] = value;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! Exchange one face layer of a fine or aggregate correction.  Only the six faces are
//! needed by the seven-point Galerkin operator.  Same-rank neighbors are read directly;
//! off-rank faces use a dedicated CUDA-aware MPI communicator so tags cannot collide
//! with fluid or laser traffic.

void ThermalRadiation::ExchangeImplicitMultilevelFaces(
    const DvceArray5D<Real> &field, const int level) {
  if (!implicit_multilevel_enabled_ || level < 0 || level > 2) {
    ImplicitRadiationError("Invalid implicit-radiation multilevel face exchange");
  }
  // Same-rank neighbors are accessed directly. All producer/consumer kernels use
  // the same ordered execution-space stream, so no host fence or packing is needed
  // when this rank has no remote faces. Retain the CUDA-aware MPI path otherwise.
  if (!implicit_multilevel_has_remote_neighbors_) return;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int fine_nx1 = indcs.nx1, fine_nx2 = indcs.nx2;
  const int nx1 = level == 0 ? indcs.nx1 : implicit_multilevel_nx1_[level-1];
  const int nx2 = level == 0 ? indcs.nx2 : implicit_multilevel_nx2_[level-1];
  const int nx3 = level == 0 ? indcs.nx3 : implicit_multilevel_nx3_[level-1];
  const int offset = level == 0 ? 0 : implicit_multilevel_offset_[level-1];
  const int active_faces = pmy_pack_->pmesh->three_d ? 6 :
      (pmy_pack_->pmesh->multi_d ? 4 : 2);
  const int max_face_cells = std::max(
      std::max(nx2*nx3, nx1*nx3), nx1*nx2);
  const int nmb = pmy_pack_->nmb_thispack;
  auto send_faces = implicit_multilevel_send_faces_;
  auto source = field;

  Kokkos::parallel_for("thermal_rad_impl_pack_multilevel_faces",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
          {0, 0, 0}, {nmb, active_faces, max_face_cells}),
  KOKKOS_LAMBDA(int m, int face, int q) {
    int ci = 0, cj = 0, ck = 0, count = 0;
    if (face < 2) {
      count = nx2*nx3;
      ck = q/nx2;
      cj = q-ck*nx2;
      ci = face == 0 ? 0 : nx1-1;
    } else if (face < 4) {
      count = nx1*nx3;
      ck = q/nx1;
      ci = q-ck*nx1;
      cj = face == 2 ? 0 : nx2-1;
    } else {
      count = nx1*nx2;
      cj = q/nx1;
      ci = q-cj*nx1;
      ck = face == 4 ? 0 : nx3-1;
    }
    if (q >= count) return;
    if (level == 0) {
      send_faces(m, face, q) = source(m, 0, ks+ck, js+cj, is+ci);
    } else {
      const int packed = offset+(ck*nx2+cj)*nx1+ci;
      send_faces(m, face, q) = ImplicitPackedScratchValue(
          source, m, packed, fine_nx1, fine_nx2, is, js, ks);
    }
  });
  Kokkos::fence();

#if MPI_PARALLEL_ENABLED
  std::vector<MPI_Request> requests;
  requests.reserve(2*nmb*active_faces);
  auto recv_faces = implicit_multilevel_recv_faces_;
  const int my_rank = global_variable::my_rank;
  for (int m = 0; m < nmb; ++m) {
    const int gid = pmy_pack_->gids+m;
    for (int face = 0; face < active_faces; ++face) {
      const int neighbor = implicit_coarse_neighbor_gid_[6*gid+face];
      if (neighbor < 0) continue;
      const int rank = pmy_pack_->pmesh->rank_eachmb[neighbor];
      if (rank == my_rank) continue;
      const int count = face < 2 ? nx2*nx3 :
          (face < 4 ? nx1*nx3 : nx1*nx2);
      auto recv = Kokkos::subview(recv_faces, m, face, Kokkos::ALL);
      requests.push_back(MPI_REQUEST_NULL);
      const int tag = m*6+face;
      if (MPI_Irecv(recv.data(), count, MPI_ATHENA_REAL, rank, tag,
                    implicit_multilevel_comm_, &requests.back()) != MPI_SUCCESS) {
        ImplicitRadiationError(
            "Could not post an implicit-radiation multilevel face receive");
      }
    }
  }
  for (int m = 0; m < nmb; ++m) {
    const int gid = pmy_pack_->gids+m;
    for (int face = 0; face < active_faces; ++face) {
      const int neighbor = implicit_coarse_neighbor_gid_[6*gid+face];
      if (neighbor < 0) continue;
      const int rank = pmy_pack_->pmesh->rank_eachmb[neighbor];
      if (rank == my_rank) continue;
      const int count = face < 2 ? nx2*nx3 :
          (face < 4 ? nx1*nx3 : nx1*nx2);
      auto send = Kokkos::subview(send_faces, m, face, Kokkos::ALL);
      requests.push_back(MPI_REQUEST_NULL);
      const int destination_lid =
          neighbor-pmy_pack_->pmesh->gids_eachrank[rank];
      const int tag = destination_lid*6+(face^1);
      if (MPI_Isend(send.data(), count, MPI_ATHENA_REAL, rank, tag,
                    implicit_multilevel_comm_, &requests.back()) != MPI_SUCCESS) {
        ImplicitRadiationError(
            "Could not post an implicit-radiation multilevel face send");
      }
    }
  }
  if (!requests.empty() && MPI_Waitall(
          static_cast<int>(requests.size()), requests.data(),
          MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    ImplicitRadiationError(
        "Could not complete implicit-radiation multilevel face communication");
  }
  Kokkos::fence();
#endif
}

//----------------------------------------------------------------------------------------
//! Reduce and solve the replicated exact MeshBlock-root Galerkin system.

void ThermalRadiation::SolveImplicitBlockRootSystem() {
  const int ncoarse = pmy_pack_->pmesh->nmb_total;
  implicit_coarse_vector_.template modify<DevExeSpace>();
  implicit_coarse_vector_.template sync<HostMemSpace>();
  auto coarse = implicit_coarse_vector_.h_view;
#if MPI_PARALLEL_ENABLED
  RadiationBatchAllreduce(MPI_IN_PLACE, coarse.data(), ncoarse, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  const auto &factor = implicit_coarse_cholesky_;
  const auto &scaling = implicit_coarse_scaling_;
  for (int row = 0; row < ncoarse; ++row) coarse(row) *= scaling[row];
  for (int row = 0; row < ncoarse; ++row) {
    Real value = coarse(row);
    for (int col = 0; col < row; ++col) {
      value -= factor[static_cast<std::size_t>(row)*ncoarse+col]*coarse(col);
    }
    coarse(row) = value/factor[static_cast<std::size_t>(row)*ncoarse+row];
  }
  for (int row = ncoarse-1; row >= 0; --row) {
    Real value = coarse(row);
    for (int col = row+1; col < ncoarse; ++col) {
      value -= factor[static_cast<std::size_t>(col)*ncoarse+row]*coarse(col);
    }
    coarse(row) = value/factor[static_cast<std::size_t>(row)*ncoarse+row];
  }
  for (int row = 0; row < ncoarse; ++row) {
    coarse(row) *= scaling[row];
    if (!std::isfinite(coarse(row))) {
      ImplicitRadiationError(
          "Implicit-radiation coarse triangular solve produced a non-finite value");
    }
  }
  implicit_coarse_vector_.template modify<HostMemSpace>();
  implicit_coarse_vector_.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! Apply either point Jacobi or a fixed linear SPD Galerkin preconditioner.  Compatible
//! factor-three MeshBlocks use a true global V-cycle with face exchanges at every level
//! and an exact MeshBlock-root solve at the bottom.  Other block sizes use point Jacobi.

void ThermalRadiation::ApplyImplicitPreconditioner(
    const DvceArray5D<Real> &input_residual,
    DvceArray5D<Real> &output_preconditioned, const Real dt) {
  RadiationTimerScope phase_timer(solver_timers_, RadiationTimer::Preconditioner);
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int fine_nx1 = indcs.nx1;
  const int fine_nx2 = indcs.nx2;
  const int fine_nx3 = indcs.nx3;
  const int nmb = pmy_pack_->nmb_thispack;
  const int nmb1 = nmb-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  auto coefficient = implicit_coefficient_;
  auto diagonal = implicit_operator_;
  auto residual = input_residual;
  auto preconditioned = output_preconditioned;
  if (implicit_preconditioner_mode_ == 5) {
    implicit_native_solver_->Apply(pmy_pack_, input_residual, output_preconditioned);
    return;
  }
  if (pmy_pack_->pmesh->multilevel) {
    if (implicit_preconditioner_mode_ == 4) {
      ApplyImplicitAMRAggregatePreconditioner(input_residual, output_preconditioned);
      return;
    }
    auto amr_diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
    auto aggregate_diagonal = implicit_multilevel_vector_;
    auto aggregate_correction = implicit_coarse_scratch_;
    const bool local_levels = implicit_multilevel_enabled_ &&
        implicit_preconditioner_mode_ >= 1;
    const bool global_aggregate = implicit_preconditioner_mode_ >= 2;
    const int global_level = implicit_preconditioner_mode_ == 3 ? 0 : 1;
    const int global_sum_offset = implicit_multilevel_offset_[1]+
        implicit_multilevel_nx1_[1]*implicit_multilevel_nx2_[1]*
        implicit_multilevel_nx3_[1];
    if (local_levels) {
      // Each local term is P diag(P^T V A_positive P)^-1 P^T V. Cell volume
      // cancels within a block. Save both corrections before one final cell pass.
      for (int level = 0; level < 2; ++level) {
        const int cnx1 = implicit_multilevel_nx1_[level];
        const int cnx2 = implicit_multilevel_nx2_[level];
        const int cnx3 = implicit_multilevel_nx3_[level];
        const int aggregate1 = fine_nx1/cnx1;
        const int aggregate2 = fine_nx2/cnx2;
        const int aggregate3 = fine_nx3/cnx3;
        const int aggregate_cells = aggregate1*aggregate2*aggregate3;
        const int coarse_cells = cnx1*cnx2*cnx3;
        const int level_offset = implicit_multilevel_offset_[level];
        const bool save_global_sum = global_aggregate && level == global_level;
        par_for_outer("thermal_rad_amr_aggregate_restrict", DevExeSpace(), 0, 0,
                      0, nmb1, 0, coarse_cells-1,
        KOKKOS_LAMBDA(TeamMember_t tmember, const int m, const int cell) {
          const int ci = cell%cnx1;
          const int cj = (cell/cnx1)%cnx2;
          const int ck = cell/(cnx1*cnx2);
          Real sum = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, aggregate_cells),
          [=](const int child, Real &value) {
            const int i = is+ci*aggregate1+child%aggregate1;
            const int j = js+cj*aggregate2+(child/aggregate1)%aggregate2;
            const int k = ks+ck*aggregate3+child/(aggregate1*aggregate2);
            value += residual(m, 0, k, j, i);
          }, sum);
          Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
            const int packed = level_offset+cell;
            SetImplicitPackedScratchValue(aggregate_correction, m, packed,
                fine_nx1, fine_nx2, is, js, ks,
                sum/aggregate_diagonal(m, packed));
            if (save_global_sum) {
              // Reuse the global graph's factor-three or factor-nine restriction.
              // Store raw sums after both local levels, keeping them disjoint.
              SetImplicitPackedScratchValue(aggregate_correction, m,
                  global_sum_offset+cell, fine_nx1, fine_nx2, is, js, ks, sum);
            }
          });
        });
      }
    }
    if (global_aggregate) {
      // Finish Jacobi, both local terms and global prolongation in one cell pass.
      ApplyImplicitAMRAggregatePreconditioner(input_residual, output_preconditioned);
      return;
    }
    const int level0_nx1 = implicit_multilevel_nx1_[0];
    const int level0_nx2 = implicit_multilevel_nx2_[0];
    const int level1_nx1 = implicit_multilevel_nx1_[1];
    const int level1_nx2 = implicit_multilevel_nx2_[1];
    const int level1_offset = implicit_multilevel_offset_[1];
    const int ay0 = fine_nx2 == 1 ? 1 : 3, az0 = fine_nx3 == 1 ? 1 : 3;
    const int ay1 = fine_nx2 == 1 ? 1 : 9, az1 = fine_nx3 == 1 ? 1 : 9;
    par_for("thermal_rad_impl_amr_local_correct", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real value = residual(m, 0, k, j, i)/amr_diagonal(m, 0, k, j, i);
      if (local_levels) {
        const int a0 = (((k-ks)/az0)*level0_nx2+(j-js)/ay0)*level0_nx1+(i-is)/3;
        const int a1 = level1_offset+
            (((k-ks)/az1)*level1_nx2+(j-js)/ay1)*level1_nx1+(i-is)/9;
        value += ImplicitPackedScratchValue(
            aggregate_correction, m, a0, fine_nx1, fine_nx2, is, js, ks);
        value += ImplicitPackedScratchValue(
            aggregate_correction, m, a1, fine_nx1, fine_nx2, is, js, ks);
      }
      preconditioned(m, 0, k, j, i) = value;
    });
    if (implicit_preconditioner_mode_ == 1) {
      ApplyImplicitAMRBlockCoarseCorrection(input_residual, output_preconditioned);
    }
    // Packed scratch remains valid until the next halo exchange on this stream.
    return;
  }
  auto size = pmy_pack_->pmb->mb_size;
  auto mb_bcs = pmy_pack_->pmb->mb_bcs;
  const int gid0 = pmy_pack_->gids;
  const int b0 = implicit_boundary_type_[0];
  const int b1 = implicit_boundary_type_[1];
  const int b2 = implicit_boundary_type_[2];
  const int b3 = implicit_boundary_type_[3];
  const int b4 = implicit_boundary_type_[4];
  const int b5 = implicit_boundary_type_[5];
  const Real vacuum_cap_coefficient = 0.5*chat_*flux_limit_coefficient_;
  const bool exact_boundary_diagonal = implicit_multilevel_enabled_;

  // The block solve uses the exact diagonal of the homogeneous operator.
  // Preserve the legacy point-Jacobi estimate in the default mode so selecting no new
  // preconditioner does not perturb existing runs.
  auto build_fine_diagonal = [&]() {
    par_for("thermal_rad_impl_diagonal", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real kc = coefficient(m, 0, k, j, i);
    const Real dx1 = size.d_view(m).dx1;
    const bool physical_p1 = i == ie && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::outer_x1));
    const bool physical_m1 = i == is && IsPhysicalBoundary(
        mb_bcs.d_view(m, BoundaryFace::inner_x1));
    const Real kp1 = ImplicitFrozenFaceCoefficient(
        kc, coefficient(m, 0, k, j, i+1), physical_p1 && b1 == 2,
        vacuum_cap_coefficient*dx1);
    const Real km1 = ImplicitFrozenFaceCoefficient(
        kc, coefficient(m, 0, k, j, i-1), physical_m1 && b0 == 2,
        vacuum_cap_coefficient*dx1);
    const Real wp1 = exact_boundary_diagonal ?
        ImplicitFaceDiagonalWeight(physical_p1, b1) : 1.0;
    const Real wm1 = exact_boundary_diagonal ?
        ImplicitFaceDiagonalWeight(physical_m1, b0) : 1.0;
    Real rate = (wp1*kp1+wm1*km1)/(dx1*dx1);
    if (multi_d) {
      const Real dx2 = size.d_view(m).dx2;
      const bool physical_p2 = j == je && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x2));
      const bool physical_m2 = j == js && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x2));
      const Real kp2 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k, j+1, i), physical_p2 && b3 == 2,
          vacuum_cap_coefficient*dx2);
      const Real km2 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k, j-1, i), physical_m2 && b2 == 2,
          vacuum_cap_coefficient*dx2);
      const Real wp2 = exact_boundary_diagonal ?
          ImplicitFaceDiagonalWeight(physical_p2, b3) : 1.0;
      const Real wm2 = exact_boundary_diagonal ?
          ImplicitFaceDiagonalWeight(physical_m2, b2) : 1.0;
      rate += (wp2*kp2+wm2*km2)/(dx2*dx2);
    }
    if (three_d) {
      const Real dx3 = size.d_view(m).dx3;
      const bool physical_p3 = k == ke && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::outer_x3));
      const bool physical_m3 = k == ks && IsPhysicalBoundary(
          mb_bcs.d_view(m, BoundaryFace::inner_x3));
      const Real kp3 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k+1, j, i), physical_p3 && b5 == 2,
          vacuum_cap_coefficient*dx3);
      const Real km3 = ImplicitFrozenFaceCoefficient(
          kc, coefficient(m, 0, k-1, j, i), physical_m3 && b4 == 2,
          vacuum_cap_coefficient*dx3);
      const Real wp3 = exact_boundary_diagonal ?
          ImplicitFaceDiagonalWeight(physical_p3, b5) : 1.0;
      const Real wm3 = exact_boundary_diagonal ?
          ImplicitFaceDiagonalWeight(physical_m3, b4) : 1.0;
      rate += (wp3*kp3+wm3*km3)/(dx3*dx3);
    }
      diagonal(m, 0, k, j, i) = 1.0+dt*rate;
    });
  };
  build_fine_diagonal();

  if (implicit_preconditioner_mode_ == 0 || !implicit_multilevel_enabled_) {
    par_for("thermal_rad_impl_jacobi", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      preconditioned(m, 0, k, j, i) =
          residual(m, 0, k, j, i)/diagonal(m, 0, k, j, i);
    });
    return;
  }

  Kokkos::deep_copy(DevExeSpace(), output_preconditioned, 0.0);

  // True global Galerkin V-cycle.  P is piecewise-constant injection and P^T is exact
  // summation.  Each red/black pre-sweep is paired with the exact black/red transpose.
  auto hierarchy_rhs = implicit_coarse_scratch_;
  auto hierarchy_solution = implicit_operator_;
  auto hierarchy_diagonal = implicit_multilevel_vector_;
  auto conductance = implicit_multilevel_conductance_;
  auto neighbor_gid = implicit_coarse_neighbor_gid_device_.d_view;
  auto neighbor_rank = implicit_coarse_neighbor_rank_device_.d_view;
  auto block_parity = implicit_multilevel_block_parity_.d_view;
  auto remote_faces = implicit_multilevel_recv_faces_;
  const int my_rank = global_variable::my_rank;
  const int level1_nx1 = implicit_multilevel_nx1_[0];
  const int level1_nx2 = implicit_multilevel_nx2_[0];
  const int level1_nx3 = implicit_multilevel_nx3_[0];
  const int level2_nx1 = implicit_multilevel_nx1_[1];
  const int level2_nx2 = implicit_multilevel_nx2_[1];
  const int level2_nx3 = implicit_multilevel_nx3_[1];
  const int level1_offset = implicit_multilevel_offset_[0];
  const int level2_offset = implicit_multilevel_offset_[1];
  const int aggregate1_x1 = fine_nx1/level1_nx1;
  const int aggregate1_x2 = fine_nx2/level1_nx2;
  const int aggregate1_x3 = fine_nx3/level1_nx3;
  const int child1 = level1_nx1/level2_nx1;
  const int child2 = level1_nx2/level2_nx2;
  const int child3 = level1_nx3/level2_nx3;

  par_for("thermal_rad_impl_vcycle_fine_red_forward", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int parity = (i-is)+(j-js)+(k-ks)+block_parity(gid0+m, 0);
    if (parity%2 == 0) {
      preconditioned(m, 0, k, j, i) =
          residual(m, 0, k, j, i)/diagonal(m, 0, k, j, i);
    }
  });
  ExchangeImplicitMultilevelFaces(output_preconditioned, 0);
  par_for("thermal_rad_impl_vcycle_fine_black_forward", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int parity = (i-is)+(j-js)+(k-ks)+block_parity(gid0+m, 0);
    if (parity%2 != 0) {
      const Real neighbor_sum = ImplicitFineGlobalNeighborSum(
          coefficient, preconditioned, remote_faces, neighbor_gid, neighbor_rank,
          m, gid0+m, gid0, my_rank, k, j, i, is, ie, js, je, ks, ke,
          fine_nx1, fine_nx2, dt, size.d_view(m).dx1,
          multi_d ? size.d_view(m).dx2 : 1.0,
          three_d ? size.d_view(m).dx3 : 1.0, multi_d, three_d);
      preconditioned(m, 0, k, j, i) =
          (residual(m, 0, k, j, i)+neighbor_sum)/diagonal(m, 0, k, j, i);
    }
  });
  ExchangeImplicitMultilevelFaces(output_preconditioned, 0);

  // A second fine-grid sweep damps the strongly heterogeneous cell-scale modes that
  // remain after one red/black pass in very stiff multigroup systems.  Its matching
  // transpose sweep below keeps the V-cycle linear and symmetric for ordinary PCG.
  for (int color = 0; color <= 1; ++color) {
    par_for("thermal_rad_impl_vcycle_fine_forward_polish", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const int parity = (i-is)+(j-js)+(k-ks)+block_parity(gid0+m, 0);
      if (parity%2 == color) {
        const Real neighbor_sum = ImplicitFineGlobalNeighborSum(
            coefficient, preconditioned, remote_faces, neighbor_gid, neighbor_rank,
            m, gid0+m, gid0, my_rank, k, j, i, is, ie, js, je, ks, ke,
            fine_nx1, fine_nx2, dt, size.d_view(m).dx1,
            multi_d ? size.d_view(m).dx2 : 1.0,
            three_d ? size.d_view(m).dx3 : 1.0, multi_d, three_d);
        const Real x = preconditioned(m, 0, k, j, i);
        const Real fine_residual = residual(m, 0, k, j, i)-
            (diagonal(m, 0, k, j, i)*x-neighbor_sum);
        preconditioned(m, 0, k, j, i) =
            x+fine_residual/diagonal(m, 0, k, j, i);
      }
    });
    ExchangeImplicitMultilevelFaces(output_preconditioned, 0);
  }

  par_for("thermal_rad_impl_vcycle_restrict_fine", DevExeSpace(), 0, nmb1,
          0, level1_nx3-1, 0, level1_nx2-1, 0, level1_nx1-1,
  KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
    Real restricted = 0.0;
    for (int ok = 0; ok < aggregate1_x3; ++ok) {
      const int k = ks+ck*aggregate1_x3+ok;
      for (int oj = 0; oj < aggregate1_x2; ++oj) {
        const int j = js+cj*aggregate1_x2+oj;
        for (int oi = 0; oi < aggregate1_x1; ++oi) {
          const int i = is+ci*aggregate1_x1+oi;
          const Real neighbor_sum = ImplicitFineGlobalNeighborSum(
              coefficient, preconditioned, remote_faces, neighbor_gid, neighbor_rank,
              m, gid0+m, gid0, my_rank, k, j, i, is, ie, js, je, ks, ke,
              fine_nx1, fine_nx2, dt, size.d_view(m).dx1,
              multi_d ? size.d_view(m).dx2 : 1.0,
              three_d ? size.d_view(m).dx3 : 1.0, multi_d, three_d);
          const Real applied = diagonal(m, 0, k, j, i)*
              preconditioned(m, 0, k, j, i)-neighbor_sum;
          restricted += residual(m, 0, k, j, i)-applied;
        }
      }
    }
    const int packed = level1_offset+(ck*level1_nx2+cj)*level1_nx1+ci;
    SetImplicitPackedScratchValue(
        hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks, restricted);
  });

  par_for("thermal_rad_impl_vcycle_l1_red_forward", DevExeSpace(), 0, nmb1,
          0, level1_nx3-1, 0, level1_nx2-1, 0, level1_nx1-1,
  KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
    const int parity = ci+cj+ck+block_parity(gid0+m, 1);
    if (parity%2 == 0) {
      const int packed = level1_offset+(ck*level1_nx2+cj)*level1_nx1+ci;
      const Real value = ImplicitPackedScratchValue(
          hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)/
          hierarchy_diagonal(m, packed);
      SetImplicitPackedScratchValue(
          hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks, value);
    }
  });
  ExchangeImplicitMultilevelFaces(hierarchy_solution, 1);
  par_for("thermal_rad_impl_vcycle_l1_black_forward", DevExeSpace(), 0, nmb1,
          0, level1_nx3-1, 0, level1_nx2-1, 0, level1_nx1-1,
  KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
    const int parity = ci+cj+ck+block_parity(gid0+m, 1);
    if (parity%2 != 0) {
      const int packed = level1_offset+(ck*level1_nx2+cj)*level1_nx1+ci;
      const Real neighbor_sum = ImplicitAggregateGlobalNeighborSum(
          conductance, hierarchy_solution, remote_faces, neighbor_gid, neighbor_rank,
          m, gid0+m, gid0, my_rank, packed, level1_offset, ci, cj, ck,
          level1_nx1, level1_nx2, level1_nx3,
          fine_nx1, fine_nx2, is, js, ks, multi_d, three_d);
      const Real value = (ImplicitPackedScratchValue(
          hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)+
          neighbor_sum)/hierarchy_diagonal(m, packed);
      SetImplicitPackedScratchValue(
          hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks, value);
    }
  });
  ExchangeImplicitMultilevelFaces(hierarchy_solution, 1);

  // Parallelize the child residuals with one team per destination aggregate.
  // For 27 blocks of 36^3 cells this exposes 1728 teams instead of only 14 CUDA
  // blocks, with each thread reading the cached coarse conductances.
  const int level2_cells = level2_nx1*level2_nx2*level2_nx3;
  const int level2_plane = level2_nx1*level2_nx2;
  const int child_plane = child1*child2;
  const int children = child_plane*child3;
  par_for_outer("thermal_rad_impl_vcycle_restrict_l1", DevExeSpace(), 0, 0,
                0, nmb1, 0, level2_cells-1,
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m, const int coarse) {
    const int ck2 = coarse/level2_plane;
    const int cj2 = (coarse-ck2*level2_plane)/level2_nx1;
    const int ci2 = coarse%level2_nx1;
    Real restricted = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, children),
    [=](const int child, Real &sum) {
      const int ck = ck2*child3+child/child_plane;
      const int cj = cj2*child2+(child%child_plane)/child1;
      const int ci = ci2*child1+child%child1;
      const int packed = level1_offset+(ck*level1_nx2+cj)*level1_nx1+ci;
      const Real neighbor_sum = ImplicitAggregateGlobalNeighborSum(
          conductance, hierarchy_solution, remote_faces, neighbor_gid, neighbor_rank,
          m, gid0+m, gid0, my_rank, packed, level1_offset, ci, cj, ck,
          level1_nx1, level1_nx2, level1_nx3,
          fine_nx1, fine_nx2, is, js, ks, multi_d, three_d);
      const Real x = ImplicitPackedScratchValue(
          hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks);
      const Real applied = hierarchy_diagonal(m, packed)*x-neighbor_sum;
      sum += ImplicitPackedScratchValue(
          hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)-applied;
    }, restricted);
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      SetImplicitPackedScratchValue(
          hierarchy_rhs, m, level2_offset+coarse, fine_nx1, fine_nx2,
          is, js, ks, restricted);
    });
  });

  par_for("thermal_rad_impl_vcycle_l2_red_forward", DevExeSpace(), 0, nmb1,
          0, level2_nx3-1, 0, level2_nx2-1, 0, level2_nx1-1,
  KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
    const int parity = ci+cj+ck+block_parity(gid0+m, 2);
    if (parity%2 == 0) {
      const int packed = level2_offset+(ck*level2_nx2+cj)*level2_nx1+ci;
      const Real value = ImplicitPackedScratchValue(
          hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)/
          hierarchy_diagonal(m, packed);
      SetImplicitPackedScratchValue(
          hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks, value);
    }
  });
  ExchangeImplicitMultilevelFaces(hierarchy_solution, 2);
  par_for("thermal_rad_impl_vcycle_l2_black_forward", DevExeSpace(), 0, nmb1,
          0, level2_nx3-1, 0, level2_nx2-1, 0, level2_nx1-1,
  KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
    const int parity = ci+cj+ck+block_parity(gid0+m, 2);
    if (parity%2 != 0) {
      const int packed = level2_offset+(ck*level2_nx2+cj)*level2_nx1+ci;
      const Real neighbor_sum = ImplicitAggregateGlobalNeighborSum(
          conductance, hierarchy_solution, remote_faces, neighbor_gid, neighbor_rank,
          m, gid0+m, gid0, my_rank, packed, level2_offset, ci, cj, ck,
          level2_nx1, level2_nx2, level2_nx3,
          fine_nx1, fine_nx2, is, js, ks, multi_d, three_d);
      const Real value = (ImplicitPackedScratchValue(
          hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)+
          neighbor_sum)/hierarchy_diagonal(m, packed);
      SetImplicitPackedScratchValue(
          hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks, value);
    }
  });
  ExchangeImplicitMultilevelFaces(hierarchy_solution, 2);

  auto coarse_vector = implicit_coarse_vector_.d_view;
  Kokkos::deep_copy(DevExeSpace(), coarse_vector, 0.0);
  par_for_outer("thermal_rad_impl_vcycle_restrict_root", DevExeSpace(), 0, 0,
                0, nmb1,
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real block_sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, level2_cells),
    [=](const int idx, Real &sum) {
      const int ck = idx/level2_plane;
      const int remainder = idx-ck*level2_plane;
      const int cj = remainder/level2_nx1;
      const int ci = remainder-cj*level2_nx1;
      const int packed = level2_offset+idx;
      const Real neighbor_sum = ImplicitAggregateGlobalNeighborSum(
          conductance, hierarchy_solution, remote_faces, neighbor_gid, neighbor_rank,
          m, gid0+m, gid0, my_rank, packed, level2_offset, ci, cj, ck,
          level2_nx1, level2_nx2, level2_nx3,
          fine_nx1, fine_nx2, is, js, ks, multi_d, three_d);
      const Real x = ImplicitPackedScratchValue(
          hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks);
      const Real applied = hierarchy_diagonal(m, packed)*x-neighbor_sum;
      sum += ImplicitPackedScratchValue(
          hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)-applied;
    }, block_sum);
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      coarse_vector(gid0+m) = block_sum;
    });
  });
  SolveImplicitBlockRootSystem();
  par_for("thermal_rad_impl_vcycle_prolong_root", DevExeSpace(), 0, nmb1,
          0, level2_nx3-1, 0, level2_nx2-1, 0, level2_nx1-1,
  KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
    const int packed = level2_offset+(ck*level2_nx2+cj)*level2_nx1+ci;
    const Real value = ImplicitPackedScratchValue(
        hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks)+
        coarse_vector(gid0+m);
    SetImplicitPackedScratchValue(
        hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks, value);
  });

  // Transpose post-sweep at level 2: exchange the prolonged root, update black,
  // exchange black, then update red.
  ExchangeImplicitMultilevelFaces(hierarchy_solution, 2);
  for (int color = 1; color >= 0; --color) {
    par_for("thermal_rad_impl_vcycle_l2_backward", DevExeSpace(), 0, nmb1,
            0, level2_nx3-1, 0, level2_nx2-1, 0, level2_nx1-1,
    KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
      const int parity = ci+cj+ck+block_parity(gid0+m, 2);
      if (parity%2 == color) {
        const int packed = level2_offset+(ck*level2_nx2+cj)*level2_nx1+ci;
        const Real neighbor_sum = ImplicitAggregateGlobalNeighborSum(
            conductance, hierarchy_solution, remote_faces, neighbor_gid, neighbor_rank,
            m, gid0+m, gid0, my_rank, packed, level2_offset, ci, cj, ck,
            level2_nx1, level2_nx2, level2_nx3,
            fine_nx1, fine_nx2, is, js, ks, multi_d, three_d);
        const Real x = ImplicitPackedScratchValue(
            hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks);
        const Real coarse_residual = ImplicitPackedScratchValue(
            hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)-
            (hierarchy_diagonal(m, packed)*x-neighbor_sum);
        SetImplicitPackedScratchValue(
            hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks,
            x+coarse_residual/hierarchy_diagonal(m, packed));
      }
    });
    if (color == 1) ExchangeImplicitMultilevelFaces(hierarchy_solution, 2);
  }

  par_for("thermal_rad_impl_vcycle_prolong_l2", DevExeSpace(), 0, nmb1,
          0, level1_nx3-1, 0, level1_nx2-1, 0, level1_nx1-1,
  KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
    const int packed1 = level1_offset+(ck*level1_nx2+cj)*level1_nx1+ci;
    const int ci2 = ci/child1;
    const int cj2 = cj/child2;
    const int ck2 = ck/child3;
    const int packed2 = level2_offset+(ck2*level2_nx2+cj2)*level2_nx1+ci2;
    const Real value = ImplicitPackedScratchValue(
        hierarchy_solution, m, packed1, fine_nx1, fine_nx2, is, js, ks)+
        ImplicitPackedScratchValue(
            hierarchy_solution, m, packed2, fine_nx1, fine_nx2, is, js, ks);
    SetImplicitPackedScratchValue(
        hierarchy_solution, m, packed1, fine_nx1, fine_nx2, is, js, ks, value);
  });
  ExchangeImplicitMultilevelFaces(hierarchy_solution, 1);
  for (int color = 1; color >= 0; --color) {
    par_for("thermal_rad_impl_vcycle_l1_backward", DevExeSpace(), 0, nmb1,
            0, level1_nx3-1, 0, level1_nx2-1, 0, level1_nx1-1,
    KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
      const int parity = ci+cj+ck+block_parity(gid0+m, 1);
      if (parity%2 == color) {
        const int packed = level1_offset+(ck*level1_nx2+cj)*level1_nx1+ci;
        const Real neighbor_sum = ImplicitAggregateGlobalNeighborSum(
            conductance, hierarchy_solution, remote_faces, neighbor_gid, neighbor_rank,
            m, gid0+m, gid0, my_rank, packed, level1_offset, ci, cj, ck,
            level1_nx1, level1_nx2, level1_nx3,
            fine_nx1, fine_nx2, is, js, ks, multi_d, three_d);
        const Real x = ImplicitPackedScratchValue(
            hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks);
        const Real coarse_residual = ImplicitPackedScratchValue(
            hierarchy_rhs, m, packed, fine_nx1, fine_nx2, is, js, ks)-
            (hierarchy_diagonal(m, packed)*x-neighbor_sum);
        SetImplicitPackedScratchValue(
            hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks,
            x+coarse_residual/hierarchy_diagonal(m, packed));
      }
    });
    if (color == 1) ExchangeImplicitMultilevelFaces(hierarchy_solution, 1);
  }

  par_for("thermal_rad_impl_vcycle_prolong_l1", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int ci = (i-is)/aggregate1_x1;
    const int cj = (j-js)/aggregate1_x2;
    const int ck = (k-ks)/aggregate1_x3;
    const int packed = level1_offset+(ck*level1_nx2+cj)*level1_nx1+ci;
    preconditioned(m, 0, k, j, i) += ImplicitPackedScratchValue(
        hierarchy_solution, m, packed, fine_nx1, fine_nx2, is, js, ks);
  });

  // Coarse solutions alias packed entries of the fine diagonal.  Restore it before the
  // fine transpose sweep, then exchange the newly prolonged correction.
  build_fine_diagonal();
  ExchangeImplicitMultilevelFaces(output_preconditioned, 0);
  for (int sweep = 0; sweep < 2; ++sweep) {
    for (int color = 1; color >= 0; --color) {
      par_for("thermal_rad_impl_vcycle_fine_backward", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const int parity = (i-is)+(j-js)+(k-ks)+block_parity(gid0+m, 0);
        if (parity%2 == color) {
          const Real neighbor_sum = ImplicitFineGlobalNeighborSum(
              coefficient, preconditioned, remote_faces, neighbor_gid, neighbor_rank,
              m, gid0+m, gid0, my_rank, k, j, i, is, ie, js, je, ks, ke,
              fine_nx1, fine_nx2, dt, size.d_view(m).dx1,
              multi_d ? size.d_view(m).dx2 : 1.0,
              three_d ? size.d_view(m).dx3 : 1.0, multi_d, three_d);
          const Real x = preconditioned(m, 0, k, j, i);
          const Real fine_residual = residual(m, 0, k, j, i)-
              (diagonal(m, 0, k, j, i)*x-neighbor_sum);
          preconditioned(m, 0, k, j, i) =
              x+fine_residual/diagonal(m, 0, k, j, i);
        }
      });
      if (sweep != 1 || color != 0) {
        ExchangeImplicitMultilevelFaces(output_preconditioned, 0);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! FLASH-like multigroup diffusion: freeze opacity and limiter coefficients at the old
//! state, then solve one centered backward-Euler scalar diffusion equation for each
//! group.  The explicit AP/upwind correction is not part of this matrix.  The group loop
//! uses bounded worker storage; group batches combine communication while retaining
//! independent scalar equations and convergence checks.

void ThermalRadiation::SolveImplicitTransport(
    const Real dt, DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &temperature, MeshBoundaryValuesCC *pbval) {
  if (!implicit_transport_ || !(dt > 0.0)) return;
  if (implicit_group_batch_size_ > 1 && ngroups > 1 && !implicit_batch_worker_) {
    SolveImplicitTransportBatched(dt, cons, prim, temperature, pbval);
    return;
  }
  solver_timers_.Reset();
  RadiationTimerScope transport_timer(solver_timers_, RadiationTimer::Transport);
  RadiationSummaryScope summary(solver_summary_ && !implicit_batch_worker_,
      pmy_pack_->pmesh->ncycle, dt, global_variable::my_rank, "transport");
  if (pbval == nullptr) {
    ImplicitRadiationError("Implicit thermal radiation requires a fluid communicator");
  }
  implicit_bvals_ = pbval;
  if (implicit_halo_exchange_ != 0) {
    if (implicit_halo_ == nullptr) implicit_halo_ = new ImplicitRadiationHalo;
    implicit_halo_->Prepare(pmy_pack_, pbval, implicit_halo_exchange_ == 2);
  }

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ngh = indcs.ng;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const int n1 = indcs.nx1+2*ngh;
  const int n2 = multi_d ? indcs.nx2+2*ngh : 1;
  const int n3 = three_d ? indcs.nx3+2*ngh : 1;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const int ncell = pmy_pack_->nmb_thispack*indcs.nx1*indcs.nx2*indcs.nx3;

  // Equal-level leaves have no coarse/fine reconstruction: both spatial orders
  // give the same volume-symmetric matrix, so retain CG on root-only and fully
  // refined meshes. Recheck the global host topology after every AMR change.
  bool mixed_levels = false;
  if (pmy_pack_->pmesh->multilevel) {
    const auto *locations = pmy_pack_->pmesh->lloc_eachmb;
    const int first_level = locations[0].level;
    for (int gid = 1; gid < pmy_pack_->pmesh->nmb_total; ++gid) {
      if (locations[gid].level != first_level) {
        mixed_levels = true;
        break;
      }
    }
  }

  auto kt = kappa_transport_.d_view;
  const bool use_table = use_opacity_table_;
  OpacityTableDevice opacity;
  if (use_table) opacity = opacity_table_->DeviceData();
  const bool use_mixed_table = use_mixed_opacity_table_;
  MixedOpacityTableDevice mixed_opacity;
  if (use_mixed_table) mixed_opacity = mixed_opacity_table_->DeviceData();
  const bool use_materials = use_material_mixture_;
  auto mixture = material_mixture_;
  auto size = pmy_pack_->pmb->mb_size;
  auto mb_bcs = pmy_pack_->pmb->mb_bcs;
  const int ielectron = iele_;
  const Real gm1 = gamma_minus_one_;
  const Real fe = cv_e_fraction_;
  const Real chat = chat_;
  const Real alpha = flux_limit_coefficient_;
  const Real floor = energy_floor_;
  const int mode = limiter_mode_;
  const int b0 = implicit_boundary_type_[0];
  const int b1 = implicit_boundary_type_[1];
  const int b2 = implicit_boundary_type_[2];
  const int b3 = implicit_boundary_type_[3];
  const int b4 = implicit_boundary_type_[4];
  const int b5 = implicit_boundary_type_[5];
  const Real vacuum_cap_coefficient = 0.5*chat*alpha;

  auto old = implicit_old_;
  auto solution = implicit_solution_;
  auto coefficient = implicit_coefficient_;
  auto residual = implicit_residual_;
  auto direction = implicit_direction_;
  auto preconditioned = implicit_preconditioned_;
  auto applied = implicit_operator_;

  const bool deferred_low_order = implicit_batch_retry_low_order_;
  implicit_batch_retry_low_order_ = false;
  if (!deferred_low_order) {
    implicit_iterations_last_solve = 0;
    implicit_residual_replacements_last_solve = 0;
  }
  implicit_residual_last_solve = 0.0;
  implicit_backward_error_last_solve = 0.0;
  implicit_boundary_power = 0.0;
  implicit_amr_fallbacks_last_solve = deferred_low_order ? 1 : 0;
  bool retry_low_order = deferred_low_order;
  if (deferred_low_order && implicit_report_ && global_variable::my_rank == 0) {
    std::cout << "# implicit radiation AMR resume: group=" << implicit_group_begin_
              << " order=1 reason=batched_negative_energy" << std::endl;
  }
  int positivity_refinements = 0;
  int positivity_iterations = 0;
  int positivity_replacements = 0;
  Real equation_amplitude = 1.0;
  Real cg_tolerance = implicit_tolerance_;
  constexpr int max_positivity_refinements = 3;
  int timed_group = -1, timed_attempt = 0;
  for (int g = implicit_group_begin_;
       g < (implicit_group_end_ < 0 ? ngroups : implicit_group_end_); ++g) {
    PrepareImplicitAMRReuseGroup(implicit_amr_aggregate_state_, g,
        pmy_pack_->pmesh->ncycle, implicit_amg_reuse_ &&
        RadiationGroupBatch::Active() != nullptr);
    for (int face = 0; face < 6; ++face) {
      implicit_boundary_value_[face] = implicit_boundary_group_values_[face*ngroups+g];
    }
    const bool warm_low_order = retry_low_order && positivity_refinements == 0;
    const bool amr_gradient = pmy_pack_->pmesh->multilevel && implicit_amr_order_ == 2;
    implicit_amr_high_order_ = amr_gradient && mixed_levels && !retry_low_order;
    if (g != timed_group) { timed_group = g; timed_attempt = 0; }
    RadiationTimerScope group_timer(solver_timers_, RadiationTimer::Group, true,
                                    g, ++timed_attempt, implicit_amr_high_order_ ? 2 : 1);
    retry_low_order = false;
    const int group_index = ifirst+g;
    const Real tolerance_squared = cg_tolerance*cg_tolerance;
    const int cg_max_iterations = implicit_max_iterations_-positivity_iterations;
    if (positivity_refinements == 0) {
      Kokkos::deep_copy(DevExeSpace(), implicit_old_, 0.0);
      if (!warm_low_order) Kokkos::deep_copy(DevExeSpace(), implicit_solution_, 0.0);
      Kokkos::deep_copy(DevExeSpace(), implicit_coefficient_, 0.0);
      Kokkos::deep_copy(DevExeSpace(), implicit_residual_, 0.0);
      Kokkos::deep_copy(DevExeSpace(), implicit_direction_, 0.0);
      Kokkos::deep_copy(DevExeSpace(), implicit_preconditioned_, 0.0);
      Kokkos::deep_copy(DevExeSpace(), implicit_operator_, 0.0);

      par_for("thermal_rad_impl_copy", DevExeSpace(), 0, nmb1,
              0, n3-1, 0, n2-1, 0, n1-1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        old(m, 0, k, j, i) = cons(m, group_index, k, j, i);
      });
      int negative_rhs_cells = 0;
      int nonfinite_rhs_cells = 0;
      Kokkos::parallel_reduce("thermal_rad_impl_validate_rhs_negative",
          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
              {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(int m, int k, int j, int i, int &negative) {
        const Real value = old(m, 0, k, j, i);
        if (Kokkos::isfinite(value) && value < 0.0) ++negative;
      }, negative_rhs_cells);
      Kokkos::parallel_reduce("thermal_rad_impl_validate_rhs_finite",
          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
              {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(int m, int k, int j, int i, int &nonfinite) {
        if (!Kokkos::isfinite(old(m, 0, k, j, i))) ++nonfinite;
      }, nonfinite_rhs_cells);
  #if MPI_PARALLEL_ENABLED
      RadiationBatchAllreduce(MPI_IN_PLACE, &negative_rhs_cells, 1, MPI_INT,
                    MPI_SUM, MPI_COMM_WORLD);
      RadiationBatchAllreduce(MPI_IN_PLACE, &nonfinite_rhs_cells, 1, MPI_INT,
                    MPI_SUM, MPI_COMM_WORLD);
  #endif
      if (negative_rhs_cells != 0 || nonfinite_rhs_cells != 0) {
        ImplicitRadiationError(
            "Implicit radiation right-hand side contains "+
            std::to_string(negative_rhs_cells)+" negative and "+
            std::to_string(nonfinite_rhs_cells)+" non-finite cells in group "+
            std::to_string(g));
      }
      const bool high_order_transport = implicit_amr_high_order_;
      implicit_amr_high_order_ = amr_gradient;
      ExchangeImplicitField(implicit_old_, pbval, false);
      if (amr_gradient) PrepareImplicitAMRGradient(implicit_old_);
      implicit_amr_high_order_ = high_order_transport;
      auto amr_gradients = ImplicitAMRCellBand(implicit_amr_gradient_);
      if (!warm_low_order) {
        Kokkos::deep_copy(DevExeSpace(), implicit_solution_, implicit_old_);
      }

      // Freeze the nonlinear diffusion coefficient from the state entering this transport
      // substep (after any preceding operator-split conduction).  Centered cell gradients
      // set the cell coefficients; faces use arithmetic averages on the same level and
      // summed half-cell resistances across refinement interfaces.
      RadiationTimerScope coefficient_timer(solver_timers_, RadiationTimer::Coefficient);
      par_for("thermal_rad_impl_coefficient", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real density = prim(m, IDN, k, j, i);
        materials::MaterialComposition composition;
        Real tele;
        if (use_materials) {
          composition = mixture.CompositionFromPrimitive(prim, m, k, j, i);
          tele = temperature(m, 1, k, j, i);
        } else {
          tele = gm1*prim(m, ielectron, k, j, i)/fe;
        }
        OpacityTableLocation opacity_location;
        if (use_table) opacity_location = opacity.Locate(density, tele);
        MixedOpacityTableLocation mixed_location;
        if (use_mixed_table) {
          mixed_location = mixed_opacity.Locate(density, tele, composition);
        }
        const Real kappa = use_mixed_table ? mixed_opacity.Get(
            opacity_transport, g, mixed_location) : (use_table ? opacity.Get(
            opacity_transport, g, opacity_location) : kt(g));
        const Real sigma = density*kappa;
        const Real dx1 = size.d_view(m).dx1;
        const Real grad1 = amr_gradient ? amr_gradients(m, 0, k, j, i) :
            (old(m, 0, k, j, i+1)-old(m, 0, k, j, i-1))/(2.0*dx1);
        Real grad2 = 0.0;
        Real grad3 = 0.0;
        if (multi_d) {
          const Real dx2 = size.d_view(m).dx2;
          grad2 = amr_gradient ? amr_gradients(m, 1, k, j, i) :
              (old(m, 0, k, j+1, i)-old(m, 0, k, j-1, i))/(2.0*dx2);
        }
        if (three_d) {
          const Real dx3 = size.d_view(m).dx3;
          grad3 = amr_gradient ? amr_gradients(m, 2, k, j, i) :
              (old(m, 0, k+1, j, i)-old(m, 0, k-1, j, i))/(2.0*dx3);
        }
        const Real gradient = RadiationGradientMagnitude(grad1, grad2, grad3);
        const FLDLinearization properties = FLDProperties(
            sigma, old(m, 0, k, j, i), gradient, gradient,
            alpha, floor, mode);
        // The backward-Euler solve has no explicit diffusion stability limit.
        // Keep the physical frozen FLD coefficient continuous at zero gradient.
        coefficient(m, 0, k, j, i) = chat*properties.diffusion_coefficient;
      });
      coefficient_timer.Stop();
      // A symmetric CG matrix requires both cells sharing an internal face to use
      // identical frozen coefficients.  In particular, an earlier implicit-conduction
      // solve refreshes only interior material temperatures, so independently evaluating
      // coefficient ghosts would use stale thermodynamic data.  Exchange the interior
      // coefficients explicitly; physical coefficient ghosts use zero-gradient values.
      ExchangeImplicitField(implicit_coefficient_, pbval, false, true);
      if (pmy_pack_->pmesh->multilevel) {
        RadiationTimerScope operator_timer(solver_timers_, RadiationTimer::OperatorSetup);
        BuildImplicitAMROperator(dt);
      }
      if (implicit_preconditioner_mode_ < 4) BuildImplicitBlockCoarsePreconditioner(dt);
    } else {
      // A failed positivity projection has not committed any state. Reuse the
      // original RHS, frozen coefficient/operator and preconditioner exactly;
      // start a tighter solve from the original nonnegative state.
      Kokkos::deep_copy(DevExeSpace(), implicit_solution_, implicit_old_);
    }

    if (positivity_refinements == 0) {
      // Squaring a faint spectral band's physical energies can underflow even
      // when every cell value is representable. Solve the same frozen equation
      // in power-of-two scaled units; do not change its coefficient, tolerance,
      // positivity test, or conserved state. Include prescribed boundary data so
      // a dark domain illuminated from a boundary has a valid normalization.
      Real amplitude = 0.0;
      Kokkos::parallel_reduce("thermal_rad_impl_equation_amplitude",
          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
              {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
        maximum = fmax(maximum, fabs(old(m, 0, k, j, i)));
      }, Kokkos::Max<Real>(amplitude));
#if MPI_PARALLEL_ENABLED
      RadiationBatchAllreduce(MPI_IN_PLACE, &amplitude, 1, MPI_ATHENA_REAL,
                              MPI_MAX, MPI_COMM_WORLD);
#endif
      for (int face = 0; face < 6; ++face) {
        if (implicit_boundary_type_[face] == 1 &&
            IsPhysicalBoundary(pmy_pack_->pmesh->mesh_bcs[face])) {
          amplitude = std::max(amplitude, std::abs(implicit_boundary_value_[face]));
        }
      }
      equation_amplitude = 1.0;
      const Real threshold = std::sqrt(std::sqrt(std::numeric_limits<Real>::min()));
      if (amplitude > 0.0 && (amplitude < threshold || amplitude > 1.0/threshold)) {
        int exponent = 0;
        std::frexp(amplitude, &exponent);
        exponent = std::max(std::numeric_limits<Real>::min_exponent-1,
            std::min(std::numeric_limits<Real>::max_exponent-1, exponent));
        equation_amplitude = std::ldexp(static_cast<Real>(1.0), exponent);
        const Real divisor = equation_amplitude;
        par_for("thermal_rad_impl_scale_equation", DevExeSpace(), 0, nmb1,
                0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          old(m, 0, k, j, i) /= divisor;
          solution(m, 0, k, j, i) /= divisor;
        });
      }
    }
    for (int face = 0; face < 6; ++face) {
      implicit_boundary_value_[face] /= equation_amplitude;
    }
    auto restore_solution_units = [&]() {
      if (equation_amplitude == 1.0) return;
      const Real multiplier = equation_amplitude;
      par_for("thermal_rad_impl_restore_units", DevExeSpace(), 0, nmb1,
              0, n3-1, 0, n2-1, 0, n1-1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        solution(m, 0, k, j, i) *= multiplier;
      });
      for (int face = 0; face < 6; ++face) {
        implicit_boundary_value_[face] = implicit_boundary_group_values_[face*ngroups+g];
      }
    };

    Real cold_residual_norm = 0.0;
    if (warm_low_order) {
      // Compare cold and warm residuals before selecting the low-order guess.
      // The convergence scale below is independent of either initial guess.
      ExchangeImplicitField(implicit_old_, pbval, false);
      ApplyImplicitOperator(implicit_old_, implicit_operator_, dt);
      par_for("thermal_rad_impl_cold_residual", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        direction(m, 0, k, j, i) = old(m, 0, k, j, i)-applied(m, 0, k, j, i);
      });
      cold_residual_norm = ImplicitGlobalDot(implicit_direction_, implicit_direction_);
    }
    ExchangeImplicitField(implicit_solution_, pbval, false);
    ApplyImplicitOperator(implicit_solution_, implicit_operator_, dt);

    par_for("thermal_rad_impl_initial_residual", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      residual(m, 0, k, j, i) =
          old(m, 0, k, j, i)-applied(m, 0, k, j, i);
    });
    const Real rhs_norm = ImplicitGlobalDot(implicit_old_, implicit_old_);
    Real residual_norm = ImplicitGlobalDot(implicit_residual_, implicit_residual_);
    if (warm_low_order) {
      const bool use_warm = std::isfinite(cold_residual_norm) &&
          std::isfinite(residual_norm) &&
          residual_norm <= cold_residual_norm;
      const Real ratio = cold_residual_norm > 0.0 ?
          sqrt(residual_norm/cold_residual_norm) :
          (residual_norm == 0.0 ? 0.0 : std::numeric_limits<Real>::infinity());
      if (!use_warm) {
        Kokkos::deep_copy(DevExeSpace(), implicit_solution_, implicit_old_);
        par_for("thermal_rad_impl_restore_cold_residual", DevExeSpace(), 0, nmb1,
                ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          residual(m, 0, k, j, i) = direction(m, 0, k, j, i);
        });
        residual_norm = cold_residual_norm;
      }
      if (implicit_report_ && global_variable::my_rank == 0) {
        std::cout << "# implicit radiation AMR initial guess: group=" << g
                  << " selected=" << (use_warm ? "warm" : "cold")
                  << " residual_ratio=" << ratio << std::endl;
      }
    }
    // Normalize by the equation's RHS, not by b-A*x_initial. In stiff diffusion
    // the latter can exceed b by many orders of magnitude and silently permit
    // a correspondingly inaccurate solution. Inhomogeneous Dirichlet data are
    // affine: form b_eff=b-A(0) only when needed, using the same frozen operator.
    Real effective_rhs_norm = rhs_norm;
    bool affine_boundary = false;
    for (int face = 0; face < 6; ++face) {
      affine_boundary |= implicit_boundary_type_[face] == 1 &&
                         implicit_boundary_value_[face] != 0.0;
    }
    if (affine_boundary) {
      Kokkos::deep_copy(DevExeSpace(), implicit_direction_, 0.0);
      ExchangeImplicitField(implicit_direction_, pbval, false);
      ApplyImplicitOperator(implicit_direction_, implicit_operator_, dt);
      par_for("thermal_rad_effective_rhs", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        preconditioned(m, 0, k, j, i) =
            old(m, 0, k, j, i)-applied(m, 0, k, j, i);
      });
      effective_rhs_norm = ImplicitGlobalDot(implicit_preconditioned_,
                                             implicit_preconditioned_);
    }
    const Real scale = fmax(effective_rhs_norm,
        std::numeric_limits<Real>::min()*static_cast<Real>(ncell));
    // A cell hierarchy is expensive enough to build only when a correction is
    // needed. Still run the usual true-equation and positivity checks below.
    bool preconditioner_ready = implicit_preconditioner_mode_ < 4 ||
        positivity_refinements > 0;
    auto ensure_preconditioner = [&]() {
      if (!preconditioner_ready) {
        BuildImplicitBlockCoarsePreconditioner(dt);
        preconditioner_ready = true;
      }
    };
    Real rz = 0.0;
    if (implicit_preconditioner_mode_ < 4 || residual_norm > tolerance_squared*scale) {
      ensure_preconditioner();
      ApplyImplicitPreconditioner(implicit_residual_, implicit_preconditioned_, dt);
      par_for("thermal_rad_impl_initial_direction", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        direction(m, 0, k, j, i) = preconditioned(m, 0, k, j, i);
      });
      rz = ImplicitGlobalDot(implicit_residual_, implicit_preconditioned_);
    }
    int iterations = 0;
    int residual_replacements = 0;
    int refinement_sweeps = 0;
    Real recursive_relative_residual = sqrt(residual_norm/scale);
    Real relative_residual = std::numeric_limits<Real>::infinity();
    Real componentwise_backward_error = std::numeric_limits<Real>::infinity();
    constexpr Real true_residual_factor = 16.0;
    constexpr Real replacement_gap_fraction = 0.25;
    bool converged = false;
    bool recursive_claims_convergence = residual_norm <= tolerance_squared*scale;
    bool check_residual = recursive_claims_convergence;
    if (implicit_amr_high_order_) {
      converged = SolveImplicitAMRGMRES(dt, pbval, scale, iterations,
          residual_replacements, relative_residual, componentwise_backward_error);
      recursive_relative_residual = relative_residual;
    }
    while (!implicit_amr_high_order_ && !converged) {
      if (!check_residual) {
        ExchangeImplicitField(implicit_direction_, pbval, true);
        ApplyImplicitOperator(implicit_direction_, implicit_operator_, dt);
        const Real pap = ImplicitGlobalDot(implicit_direction_, implicit_operator_);
        if (!(pap > 0.0) || !std::isfinite(pap) || !std::isfinite(rz)) {
          ImplicitRadiationError("Implicit radiation operator is not positive definite");
        }
        const Real step = rz/pap;
        par_for("thermal_rad_impl_cg_update", DevExeSpace(), 0, nmb1,
                ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          solution(m, 0, k, j, i) += step*direction(m, 0, k, j, i);
          residual(m, 0, k, j, i) -= step*applied(m, 0, k, j, i);
        });
        residual_norm = ImplicitGlobalDot(implicit_residual_, implicit_residual_);
        ++iterations;
        solver_timers_.Add(RadiationCounter::CGIterations);
        recursive_relative_residual = sqrt(residual_norm/scale);
        recursive_claims_convergence =
            residual_norm <= tolerance_squared*scale;
        check_residual = recursive_claims_convergence ||
            iterations%implicit_residual_check_interval_ == 0 ||
            iterations == cg_max_iterations;

        if (!check_residual) {
          ApplyImplicitPreconditioner(
              implicit_residual_, implicit_preconditioned_, dt);
          const Real rz_new = ImplicitGlobalDot(
              implicit_residual_, implicit_preconditioned_);
          const Real beta = rz_new/rz;
          par_for("thermal_rad_impl_direction", DevExeSpace(), 0, nmb1,
                  ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
            direction(m, 0, k, j, i) = preconditioned(m, 0, k, j, i)+
                beta*direction(m, 0, k, j, i);
          });
          rz = rz_new;
          continue;
        }
      }

      // Reliable update: validate the recursively updated residual against b-Ax using
      // the actual solution ghosts.  If neither convergence measure passes, continue
      // PCG from this true residual without resetting the global iteration count.
      ExchangeImplicitField(implicit_solution_, pbval, false);
      ApplyImplicitOperator(implicit_solution_, implicit_operator_, dt);
      par_for("thermal_rad_impl_true_residual", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        applied(m, 0, k, j, i) =
            old(m, 0, k, j, i)-applied(m, 0, k, j, i);
      });
      Real true_residual_norm = ImplicitGlobalDot(
          implicit_operator_, implicit_operator_);
      relative_residual = sqrt(true_residual_norm/scale);
      componentwise_backward_error = ImplicitComponentwiseBackwardError(
          implicit_solution_, implicit_old_, implicit_operator_, dt);
      converged = std::isfinite(relative_residual) &&
          (relative_residual <= true_residual_factor*cg_tolerance ||
           componentwise_backward_error <= cg_tolerance);
      if (converged || iterations >= cg_max_iterations) break;

      // Krylov reductions eventually resolve rounding error in the stiff
      // cells instead of relative errors in much fainter cells. At that point,
      // use bounded component-relative GMRES to resolve the remaining error.
      // The true equation and total iteration budget still govern acceptance.
      bool refined_componentwise = false;
      if (implicit_preconditioner_mode_ == 5 && pmy_pack_->pmesh->multilevel &&
          iterations >= std::max(2, 2*implicit_residual_check_interval_)) {
        auto refinement_diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
        auto refinement_size = size.d_view;
        const Real roundoff_epsilon = std::numeric_limits<Real>::epsilon();
        Real noise_squared = 0.0;
        Kokkos::parallel_reduce("thermal_rad_impl_roundoff_estimate",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
                {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(int m, int k, int j, int i, Real &noise) {
          const Real amplitude = roundoff_epsilon*refinement_diagonal(m, 0, k, j, i)*
              fabs(solution(m, 0, k, j, i));
          const Real volume = refinement_size(m).dx1*refinement_size(m).dx2*
              refinement_size(m).dx3;
          noise += volume*amplitude*amplitude;
        }, noise_squared);
#if MPI_PARALLEL_ENABLED
        RadiationBatchAllreduce(MPI_IN_PLACE, &noise_squared, 1, MPI_ATHENA_REAL,
                                MPI_SUM, MPI_COMM_WORLD);
#endif
        // This estimate only selects an iteration method. Full true residual
        // and componentwise backward error still determine acceptance below.
        if (std::isfinite(noise_squared) && true_residual_norm <= 1024.0*noise_squared) {
          // Global dot products are dominated by stencil roundoff in the
          // brightest stiff cells. Resolve relative errors with a fixed
          // component metric within each bounded GMRES restart. The low-order
          // equation and its original true/BW acceptance tests stay unchanged.
          const int saved_limit = implicit_max_iterations_;
          const Real saved_tolerance = implicit_tolerance_;
          implicit_max_iterations_ = std::min(256, cg_max_iterations-iterations);
          implicit_tolerance_ = cg_tolerance;
          int correction_iterations = 0, correction_restarts = 0;
          converged = SolveImplicitAMRGMRES(dt, pbval, scale,
              correction_iterations, correction_restarts,
              relative_residual, componentwise_backward_error);
          implicit_max_iterations_ = saved_limit;
          implicit_tolerance_ = saved_tolerance;
          iterations += correction_iterations;
          refinement_sweeps += correction_iterations;
          residual_replacements += correction_restarts;
          refined_componentwise = true;
          // GMRES leaves the fresh residual in residual_, while PCG's reliable
          // update below expects it in operator_. Restart PCG after this phase.
          Kokkos::deep_copy(DevExeSpace(), implicit_operator_, implicit_residual_);
          true_residual_norm = ImplicitGlobalDot(implicit_operator_, implicit_operator_);
          relative_residual = sqrt(true_residual_norm/scale);
          recursive_relative_residual = relative_residual;
          if (implicit_report_ && global_variable::my_rank == 0) {
            std::cout << "# implicit radiation component metric refinement: group=" << g
                      << " iterations=" << iterations
                      << " corrections=" << refinement_sweeps
                      << " true_relative_residual=" << relative_residual
                      << " backward_error=" << componentwise_backward_error
                      << " roundoff_relative_estimate=" << sqrt(noise_squared/scale)
                      << std::endl;
          }
          if (converged || iterations >= cg_max_iterations) break;
        }
      }

      // Replace the recursive residual after every failed check.  A small replacement
      // gap is just the finite-precision correction to the ordinary PCG recurrence, so
      // retain its search direction in that case.  Restart from the preconditioned true
      // residual only after material drift, or when the recurrence falsely claimed
      // convergence.  Unconditionally discarding a healthy Krylov basis at every fixed
      // interval can prevent a stiff system from making enough conjugate-gradient steps.
      par_for("thermal_rad_impl_residual_gap", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        preconditioned(m, 0, k, j, i) =
            residual(m, 0, k, j, i)-applied(m, 0, k, j, i);
      });
      const Real residual_gap_norm = ImplicitGlobalDot(
          implicit_preconditioned_, implicit_preconditioned_);
      const Real residual_reference_norm = fmax(
          residual_norm, true_residual_norm);
      const bool restart_direction = refined_componentwise || recursive_claims_convergence ||
          residual_gap_norm > replacement_gap_fraction*replacement_gap_fraction*
              residual_reference_norm;
      ++residual_replacements;
      par_for("thermal_rad_impl_replace_residual", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        residual(m, 0, k, j, i) = applied(m, 0, k, j, i);
      });
      residual_norm = true_residual_norm;
      ensure_preconditioner();
      ApplyImplicitPreconditioner(
          implicit_residual_, implicit_preconditioned_, dt);
      const Real rz_new = ImplicitGlobalDot(
          implicit_residual_, implicit_preconditioned_);
      if (restart_direction) {
        par_for("thermal_rad_impl_replacement_direction", DevExeSpace(), 0, nmb1,
                ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          direction(m, 0, k, j, i) = preconditioned(m, 0, k, j, i);
        });
      } else {
        const Real beta = rz_new/rz;
        par_for("thermal_rad_impl_checked_direction", DevExeSpace(), 0, nmb1,
                ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          direction(m, 0, k, j, i) = preconditioned(m, 0, k, j, i)+
              beta*direction(m, 0, k, j, i);
        });
      }
      rz = rz_new;
      recursive_relative_residual = relative_residual;
      recursive_claims_convergence = false;
      check_residual = false;
    }
    // The tighter target only reserves accuracy for projection. At the total
    // iteration limit, still validate a candidate meeting the user's original
    // criterion; positivity and the projected residual remain mandatory below.
    if (!converged && positivity_refinements > 0 &&
        std::isfinite(relative_residual) &&
        (relative_residual <= true_residual_factor*implicit_tolerance_ ||
         componentwise_backward_error <= implicit_tolerance_)) {
      converged = true;
    }
    // A global norm can accept a solution with order-one relative errors in
    // faint, mass-dominated cells. Resolve their principal block with bounded
    // point relaxation; stiff rows remain fixed. This does not relax acceptance:
    // keep the correction only after the original full-equation check passes.
    if (converged && pmy_pack_->pmesh->multilevel &&
        iterations < cg_max_iterations &&
        componentwise_backward_error > cg_tolerance) {
      auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
      const bool high_order = implicit_amr_high_order_;
      auto weak_error = [&]() {
        Real error = 0.0;
        Kokkos::parallel_reduce("thermal_rad_impl_weak_backward_bound",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
                {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
          const Real d = diagonal(m, 0, k, j, i);
          if (d > 4.0) return;
          const Real numerator = fabs(applied(m, 0, k, j, i));
          const Real lower = high_order ? 1.0+0.4375*(d-1.0) : d;
          const Real denominator = lower*fabs(solution(m, 0, k, j, i));
          const Real value = numerator == 0.0 ? 0.0 :
              (denominator > 0.0 ? numerator/denominator : 1.0/kRealEpsilon);
          maximum = fmax(maximum, value);
        }, Kokkos::Max<Real>(error));
#if MPI_PARALLEL_ENABLED
        RadiationBatchAllreduce(MPI_IN_PLACE, &error, 1, MPI_ATHENA_REAL,
                                MPI_MAX, MPI_COMM_WORLD);
#endif
        return error;
      };
      auto true_residual = [&]() {
        ExchangeImplicitField(implicit_solution_, pbval, false);
        ApplyImplicitOperator(implicit_solution_, implicit_operator_, dt);
        par_for("thermal_rad_impl_weak_true_residual", DevExeSpace(), 0, nmb1,
                ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          applied(m, 0, k, j, i) = old(m, 0, k, j, i)-applied(m, 0, k, j, i);
        });
      };
      true_residual();
      const Real before_weak = weak_error();
      if (before_weak > cg_tolerance) {
        Kokkos::deep_copy(DevExeSpace(), implicit_preconditioned_, implicit_solution_);
        const Real before_relative = relative_residual;
        const Real before_backward = componentwise_backward_error;
        const int first_sweep = iterations;
        const int last_sweep = std::min(cg_max_iterations, iterations+256);
        Real after_weak = before_weak;
        bool accepted = false;
        do {
          par_for("thermal_rad_impl_weak_relax", DevExeSpace(), 0, nmb1,
                  ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
            const Real d = diagonal(m, 0, k, j, i);
            if (d <= 4.0) solution(m, 0, k, j, i) +=
                (2.0/3.0)*applied(m, 0, k, j, i)/d;
          });
          ++iterations;
          solver_timers_.Add(RadiationCounter::CGIterations);
          true_residual();
          if ((iterations-first_sweep)%8 == 0 || iterations == last_sweep) {
            after_weak = weak_error();
            if (std::isfinite(after_weak) && after_weak <= cg_tolerance) {
              const Real norm = ImplicitGlobalDot(implicit_operator_, implicit_operator_);
              relative_residual = sqrt(norm/scale);
              componentwise_backward_error = ImplicitComponentwiseBackwardError(
                  implicit_solution_, implicit_old_, implicit_operator_, dt);
              accepted = std::isfinite(relative_residual) &&
                  (relative_residual <= true_residual_factor*cg_tolerance ||
                   componentwise_backward_error <= cg_tolerance);
              break;
            }
          }
        } while (iterations < last_sweep);
        if (!accepted) {
          Kokkos::deep_copy(DevExeSpace(), implicit_solution_, implicit_preconditioned_);
          relative_residual = before_relative;
          componentwise_backward_error = before_backward;
        }
        if (implicit_report_ && global_variable::my_rank == 0) {
          std::cout << "# implicit radiation weak refinement: group=" << g
                    << " accepted=" << accepted << " iterations=" << iterations
                    << " sweeps=" << iterations-first_sweep
                    << " before_backward_bound=" << before_weak
                    << " after_backward_bound=" << after_weak
                    << " true_relative_residual=" << relative_residual << std::endl;
        }
      }
    }
    // A small global residual can hide relative errors in a faint diffusive
    // region even after the weak-row cleanup. Resolve those components before
    // accepting the monotone native solve, within the original total budget.
    if (converged && implicit_preconditioner_mode_ == 5 &&
        !implicit_amr_high_order_ && pmy_pack_->pmesh->multilevel &&
        componentwise_backward_error > cg_tolerance && iterations < cg_max_iterations) {
      ensure_preconditioner();
      const int saved_limit = implicit_max_iterations_;
      const Real saved_tolerance = implicit_tolerance_;
      implicit_max_iterations_ = cg_max_iterations-iterations;
      implicit_tolerance_ = cg_tolerance;
      int correction_iterations = 0, correction_restarts = 0;
      converged = SolveImplicitAMRGMRES(dt, pbval, scale,
          correction_iterations, correction_restarts,
          relative_residual, componentwise_backward_error);
      implicit_max_iterations_ = saved_limit;
      implicit_tolerance_ = saved_tolerance;
      iterations += correction_iterations;
      residual_replacements += correction_restarts;
      recursive_relative_residual = relative_residual;
      if (implicit_report_ && global_variable::my_rank == 0) {
        std::cout << "# implicit radiation accepted component refinement: group=" << g
                  << " accepted=" << converged << " iterations=" << iterations
                  << " corrections=" << correction_iterations
                  << " true_relative_residual=" << relative_residual
                  << " backward_error=" << componentwise_backward_error << std::endl;
      }
    }
    ObserveImplicitAMRIterations(implicit_amr_aggregate_state_, iterations, converged);
    if (!converged && ImplicitAMRHierarchyReused(implicit_amr_aggregate_state_) &&
        RadiationGroupBatch::Active() != nullptr) {
      // No conserved state has been committed. Let all lanes finish, release
      // the batch, and retry this group with a freshly built scalar hierarchy.
      restore_solution_units();
      implicit_batch_retry_ = true;
      return;
    }
    if (!converged) {
      std::ostringstream message;
      message << std::scientific << std::setprecision(17)
              << "Implicit radiation solve failed to converge: group=" << g
              << " iterations=" << iterations+positivity_iterations
              << " replacements=" << residual_replacements+positivity_replacements
              << " recursive_residual=" << recursive_relative_residual
              << " true_residual=" << relative_residual
              << " backward_error=" << componentwise_backward_error
              << " tolerance=" << implicit_tolerance_
              << " internal_tolerance=" << cg_tolerance;
      ImplicitRadiationError(message.str());
    }
    implicit_iterations_last_solve =
        std::max(implicit_iterations_last_solve, iterations+positivity_iterations);
    implicit_residual_replacements_last_solve = std::max(
        implicit_residual_replacements_last_solve,
        residual_replacements+positivity_replacements);

    int negative_solution_cells = 0;
    int nonfinite_solution_cells = 0;
    Kokkos::parallel_reduce("thermal_rad_impl_validate_solution",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, int &negative) {
      const Real value = solution(m, 0, k, j, i);
      if (Kokkos::isfinite(value) && value < 0.0) ++negative;
    }, negative_solution_cells);
    Kokkos::parallel_reduce("thermal_rad_impl_validate_finite",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, int &nonfinite) {
      if (!Kokkos::isfinite(solution(m, 0, k, j, i))) ++nonfinite;
    }, nonfinite_solution_cells);
    Real minimum_solution = 0.0;
    Kokkos::parallel_reduce("thermal_rad_impl_minimum_solution",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &minimum) {
      const Real value = solution(m, 0, k, j, i);
      if (Kokkos::isfinite(value) && value < minimum) minimum = value;
    }, Kokkos::Min<Real>(minimum_solution));
    Real maximum_rhs = 0.0;
    Kokkos::parallel_reduce("thermal_rad_impl_maximum_rhs",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &maximum) {
      maximum = fmax(maximum, fabs(old(m, 0, k, j, i)));
    }, Kokkos::Max<Real>(maximum_rhs));
    Real negative_solution_energy = 0.0;
    Kokkos::parallel_reduce("thermal_rad_impl_negative_solution_energy",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &negative_energy) {
      const Real value = solution(m, 0, k, j, i);
      if (Kokkos::isfinite(value) && value < 0.0) {
        negative_energy -= value*size.d_view(m).dx1*size.d_view(m).dx2*
                           size.d_view(m).dx3;
      }
    }, negative_solution_energy);
    Real positive_solution_energy = 0.0;
    Kokkos::parallel_reduce("thermal_rad_impl_positive_solution_energy",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            {0, ks, js, is}, {pmy_pack_->nmb_thispack, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &positive_energy) {
      const Real value = solution(m, 0, k, j, i);
      if (Kokkos::isfinite(value) && value > 0.0) {
        positive_energy += value*size.d_view(m).dx1*size.d_view(m).dx2*
                           size.d_view(m).dx3;
      }
    }, positive_solution_energy);
#if MPI_PARALLEL_ENABLED
    RadiationBatchAllreduce(MPI_IN_PLACE, &negative_solution_cells, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    RadiationBatchAllreduce(MPI_IN_PLACE, &nonfinite_solution_cells, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    RadiationBatchAllreduce(MPI_IN_PLACE, &minimum_solution, 1, MPI_ATHENA_REAL,
                  MPI_MIN, MPI_COMM_WORLD);
    RadiationBatchAllreduce(MPI_IN_PLACE, &maximum_rhs, 1, MPI_ATHENA_REAL,
                  MPI_MAX, MPI_COMM_WORLD);
    RadiationBatchAllreduce(MPI_IN_PLACE, &negative_solution_energy, 1, MPI_ATHENA_REAL,
                  MPI_SUM, MPI_COMM_WORLD);
    RadiationBatchAllreduce(MPI_IN_PLACE, &positive_solution_energy, 1, MPI_ATHENA_REAL,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    if (nonfinite_solution_cells != 0) {
      if (global_variable::my_rank == 0) {
        std::cerr << std::scientific << std::setprecision(17)
                  << "# implicit radiation positivity diagnostic: group=" << g
                  << " minimum=" << minimum_solution
                  << " negative_energy=" << negative_solution_energy
                  << " positive_energy=" << positive_solution_energy << std::endl;
      }
      ImplicitRadiationError(
          "Implicit radiation produced "+std::to_string(negative_solution_cells)+
          " negative and "+std::to_string(nonfinite_solution_cells)+
          " non-finite cells in group "+std::to_string(g)+
          " (minimum="+std::to_string(minimum_solution)+
          ", negative energy="+std::to_string(negative_solution_energy)+")");
    }
    if (implicit_amr_high_order_ && negative_solution_cells != 0 &&
        positive_solution_energy > negative_solution_energy) {
      // A converged Krylov iterate can contain tiny negative tails even when
      // the high-order solution is nonnegative. Test a conservative projection
      // against the SAME frozen high-order operator before changing schemes.
      // This is the same positivity/true-residual safeguard used by the monotone
      // solve below. A material undershoot fails it and still takes the low-order
      // retry. Keep the original iterate as that retry's candidate initial guess.
      Kokkos::deep_copy(DevExeSpace(), implicit_preconditioned_, implicit_solution_);
      const Real positive_scale =
          (positive_solution_energy-negative_solution_energy)/positive_solution_energy;
      par_for("thermal_rad_impl_try_high_order_positive", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real value = solution(m, 0, k, j, i);
        solution(m, 0, k, j, i) = value > 0.0 ? positive_scale*value : 0.0;
      });
      ExchangeImplicitField(implicit_solution_, pbval, false);
      ApplyImplicitOperator(implicit_solution_, implicit_operator_, dt);
      par_for("thermal_rad_impl_high_order_projected_residual", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        residual(m, 0, k, j, i) = old(m, 0, k, j, i)-applied(m, 0, k, j, i);
      });
      const Real projected_norm = ImplicitGlobalDot(implicit_residual_, implicit_residual_);
      const Real projected_relative = sqrt(projected_norm/scale);
      const Real projected_backward = ImplicitComponentwiseBackwardError(
          implicit_solution_, implicit_old_, implicit_residual_, dt);
      if (std::isfinite(projected_relative) && std::isfinite(projected_backward) &&
          (projected_relative <= 16.0*implicit_tolerance_ ||
           projected_backward <= implicit_tolerance_)) {
        negative_solution_cells = 0;
        residual_norm = projected_norm;
        relative_residual = projected_relative;
        componentwise_backward_error = projected_backward;
        if (implicit_report_ && global_variable::my_rank == 0) {
          std::cout << "# implicit radiation AMR positive projection: group=" << g
                    << " true_relative_residual=" << relative_residual
                    << " backward_error=" << componentwise_backward_error << std::endl;
        }
      } else {
        Kokkos::deep_copy(DevExeSpace(), implicit_solution_, implicit_preconditioned_);
      }
    }
    if (implicit_amr_high_order_ && negative_solution_cells != 0) {
      // Signed reconstruction weights need not form an M-matrix. Re-solve the
      // whole group with the conservative monotone operator; clipping a genuinely
      // negative high-order solution would no longer solve its implicit equation.
      // Preserve this finite iterate as a candidate guess only. CG allows signed
      // guesses, while its original RHS and final positivity checks stay unchanged.
      ++implicit_amr_fallbacks_last_solve;
      if (implicit_report_ && global_variable::my_rank == 0) {
        std::cout << "# implicit radiation AMR fallback: group=" << g
                  << " reason=negative_energy minimum=" << minimum_solution*equation_amplitude
                  << std::endl;
      }
      restore_solution_units();  // The low-order retry copies a physical RHS again.
      // Changing one group's hierarchy must wait until every batch lane retires.
      // Carry the already converged, signed candidate into that scalar retry;
      // repeating its high-order solve can otherwise cost hundreds of iterations.
      if (RadiationGroupBatch::Active() != nullptr) {
        implicit_batch_retry_ = true;
        implicit_batch_retry_low_order_ = true;
        return;
      }
      retry_low_order = true;
      --g;
      continue;
    }
    if (negative_solution_cells != 0) {
      if (RadiationGroupBatch::Active() != nullptr) {
        restore_solution_units();
        implicit_batch_retry_ = true;
        return;
      }
      // The low-order backward-Euler matrix is an M-matrix. A finite negative value
      // can only be iterative error. Preserve the volume-integrated group energy
      // while projecting onto the nonnegative cone.  A separate amplitude cutoff is
      // unreliable for nearly extinguished groups: roundoff may be concentrated in one
      // cell or spread across many.  The authoritative guard is the full-stencil true
      // residual and componentwise backward error recomputed after projection below.
      if (!(positive_solution_energy > negative_solution_energy)) {
        if (global_variable::my_rank == 0) {
          std::cerr << std::scientific << std::setprecision(17)
                    << "# implicit radiation positivity diagnostic: group=" << g
                    << " minimum=" << minimum_solution
                    << " maximum_rhs=" << maximum_rhs
                    << " negative_energy=" << negative_solution_energy
                    << " positive_energy=" << positive_solution_energy << std::endl;
        }
        ImplicitRadiationError(
            "Implicit radiation negativity exceeds the solver tolerance in group "+
            std::to_string(g));
      }
      const Real positive_scale =
          (positive_solution_energy-negative_solution_energy)/
          positive_solution_energy;
      par_for("thermal_rad_impl_project_positive", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real value = solution(m, 0, k, j, i);
        solution(m, 0, k, j, i) = value > 0.0 ? positive_scale*value : 0.0;
      });

      // Projection is accepted only when it remains a converged solution of the same
      // frozen operator.  This prevents positivity cleanup from hiding a material
      // transport error or silently changing the implicit equation.
      ExchangeImplicitField(implicit_solution_, pbval, false);
      ApplyImplicitOperator(implicit_solution_, implicit_operator_, dt);
      par_for("thermal_rad_impl_projected_residual", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        residual(m, 0, k, j, i) =
            old(m, 0, k, j, i)-applied(m, 0, k, j, i);
      });
      residual_norm = ImplicitGlobalDot(implicit_residual_, implicit_residual_);
      relative_residual = sqrt(residual_norm/scale);
      componentwise_backward_error = ImplicitComponentwiseBackwardError(
          implicit_solution_, implicit_old_, implicit_residual_, dt);
      constexpr Real projection_residual_factor = 16.0;
      const bool projection_converged = std::isfinite(relative_residual) &&
          (relative_residual <= projection_residual_factor*implicit_tolerance_ ||
           componentwise_backward_error <= implicit_tolerance_);
      if (!projection_converged) {
        const Real tighter_tolerance = fmax(0.1*cg_tolerance, 64.0*kRealEpsilon);
        if (std::isfinite(relative_residual) &&
            positivity_refinements < max_positivity_refinements &&
            iterations < cg_max_iterations && tighter_tolerance < cg_tolerance) {
          // The M-matrix solution is nonnegative, so this is still iterative
          // error. Reserve more accuracy for the conservative projection while
          // retaining the original final acceptance criterion and total budget.
          ++positivity_refinements;
          positivity_iterations += iterations;
          positivity_replacements += residual_replacements;
          cg_tolerance = tighter_tolerance;
          if (implicit_report_ && global_variable::my_rank == 0) {
            std::cout << "# implicit radiation positivity refinement: group=" << g
                      << " attempt=" << positivity_refinements
                      << " tolerance=" << cg_tolerance
                      << " projected_residual=" << relative_residual << std::endl;
          }
          retry_low_order = true;
          --g;
          continue;
        }
        std::ostringstream message;
        message << std::scientific << std::setprecision(17)
                << "Implicit radiation positivity projection failed convergence: group="
                << g << " iterations=" << iterations+positivity_iterations
                << " replacements=" << residual_replacements+positivity_replacements
                << " recursive_residual=" << recursive_relative_residual
                << " true_residual=" << relative_residual
                << " backward_error=" << componentwise_backward_error
                << " tolerance=" << implicit_tolerance_
                << " internal_tolerance=" << cg_tolerance
                << " refinements=" << positivity_refinements;
        ImplicitRadiationError(message.str());
      }
    }
    implicit_residual_last_solve =
        std::max(implicit_residual_last_solve, relative_residual);
    implicit_backward_error_last_solve = std::max(
        implicit_backward_error_last_solve, componentwise_backward_error);

    restore_solution_units();
    // Refresh the converged ghosts once, both for the next conserved-state boundary
    // fill and for a conservative boundary-loss diagnostic.  This is a rank-local
    // surface integral; AthenaK's history writer performs the MPI sum.
    ExchangeImplicitField(implicit_solution_, pbval, false);
    const int impl_nx1 = indcs.nx1;
    const int impl_nx2 = indcs.nx2;
    const int impl_nx3 = indcs.nx3;
    const int impl_nkji = impl_nx3*impl_nx2*impl_nx1;
    const int impl_nji = impl_nx2*impl_nx1;
    Real group_boundary_power = 0.0;
    Kokkos::parallel_reduce("thermal_rad_impl_boundary_power",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(const int idx, Real &power) {
      const int m = idx/impl_nkji;
      const int local = idx-m*impl_nkji;
      const int k = local/impl_nji+ks;
      const int j = (local-(k-ks)*impl_nji)/impl_nx1+js;
      const int i = local-(k-ks)*impl_nji-(j-js)*impl_nx1+is;
      const Real kc = coefficient(m, 0, k, j, i);
      const Real dx1 = size.d_view(m).dx1;
      const Real dx2 = size.d_view(m).dx2;
      const Real dx3 = size.d_view(m).dx3;
      if (i == is && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::inner_x1))) {
        const Real kface = ImplicitFrozenFaceCoefficient(
            kc, coefficient(m, 0, k, j, i-1), b0 == 2,
            vacuum_cap_coefficient*dx1);
        power += dx2*dx3*kface*
            (solution(m, 0, k, j, i)-solution(m, 0, k, j, i-1))/dx1;
      }
      if (i == ie && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::outer_x1))) {
        const Real kface = ImplicitFrozenFaceCoefficient(
            kc, coefficient(m, 0, k, j, i+1), b1 == 2,
            vacuum_cap_coefficient*dx1);
        power += dx2*dx3*kface*
            (solution(m, 0, k, j, i)-solution(m, 0, k, j, i+1))/dx1;
      }
      if (multi_d && j == js && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::inner_x2))) {
        const Real kface = ImplicitFrozenFaceCoefficient(
            kc, coefficient(m, 0, k, j-1, i), b2 == 2,
            vacuum_cap_coefficient*dx2);
        power += dx1*dx3*kface*
            (solution(m, 0, k, j, i)-solution(m, 0, k, j-1, i))/dx2;
      }
      if (multi_d && j == je && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::outer_x2))) {
        const Real kface = ImplicitFrozenFaceCoefficient(
            kc, coefficient(m, 0, k, j+1, i), b3 == 2,
            vacuum_cap_coefficient*dx2);
        power += dx1*dx3*kface*
            (solution(m, 0, k, j, i)-solution(m, 0, k, j+1, i))/dx2;
      }
      if (three_d && k == ks && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::inner_x3))) {
        const Real kface = ImplicitFrozenFaceCoefficient(
            kc, coefficient(m, 0, k-1, j, i), b4 == 2,
            vacuum_cap_coefficient*dx3);
        power += dx1*dx2*kface*
            (solution(m, 0, k, j, i)-solution(m, 0, k-1, j, i))/dx3;
      }
      if (three_d && k == ke && IsPhysicalBoundary(
              mb_bcs.d_view(m, BoundaryFace::outer_x3))) {
        const Real kface = ImplicitFrozenFaceCoefficient(
            kc, coefficient(m, 0, k+1, j, i), b5 == 2,
            vacuum_cap_coefficient*dx3);
        power += dx1*dx2*kface*
            (solution(m, 0, k, j, i)-solution(m, 0, k+1, j, i))/dx3;
      }
    }, group_boundary_power);
    implicit_boundary_power += group_boundary_power;

    par_for("thermal_rad_impl_commit", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real value = solution(m, 0, k, j, i);
      cons(m, group_index, k, j, i) = value;
      prim(m, group_index, k, j, i) = value/cons(m, IDN, k, j, i);
    });
    positivity_refinements = 0;
    positivity_iterations = 0;
    positivity_replacements = 0;
    cg_tolerance = implicit_tolerance_;
  }

  if (implicit_report_ && global_variable::my_rank == 0 && !implicit_batch_worker_) {
    std::cout << "# implicit thermal radiation: groups=" << ngroups
              << " max_iterations=" << implicit_iterations_last_solve
              << " max_residual_replacements="
              << implicit_residual_replacements_last_solve
              << " max_relative_residual=" << implicit_residual_last_solve
              << " max_componentwise_backward_error="
              << implicit_backward_error_last_solve
              << std::endl;
  }
  transport_timer.Stop();
  if (!implicit_batch_worker_) {
    solver_timers_.Report(pmy_pack_->pmesh->ncycle, dt, global_variable::my_rank, "transport");
  }
}

//----------------------------------------------------------------------------------------
//! Apply the local matter-radiation source update.
//!
//! In nonlinear mode every group is analytically eliminated at a trial end-of-step
//! electron temperature.  A safeguarded bracketed scalar solve then enforces the
//! local electron-plus-radiation energy invariant.  The compatibility mode retains
//! the original time-lagged Planck/emission coefficients.  A failed nonlinear cell
//! uses bounded lagged substeps instead of committing a partially converged state.

void ThermalRadiation::Couple(Real dt, DvceArray5D<Real> &cons,
    DvceArray5D<Real> &prim, DvceArray5D<Real> &temperature,
    Real material_pressure_floor, Real material_temperature_floor,
    int il, int iu, int jl, int ju, int kl, int ku) {
  if (!couple_matter_ || dt <= 0.0) {
    UpdateDiagnostics(cons, prim, il, iu, jl, ju, kl, ku);
    return;
  }
  solver_timers_.Reset();
  RadiationTimerScope source_timer(solver_timers_, RadiationTimer::Source);
  RadiationSummaryScope summary(solver_summary_, pmy_pack_->pmesh->ncycle, dt,
      global_variable::my_rank, "source");

  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  int ie = iele_;
  Real gm1 = gamma_minus_one_;
  Real fe = cv_e_fraction_;
  Real arad = arad_;
  Real chat = chat_;
  Real energy_floor = energy_floor_;
  auto bounds = group_bounds_.d_view;
  auto ka = kappa_absorption_.d_view;
  auto ke = kappa_emission_.d_view;
  bool use_table = use_opacity_table_;
  OpacityTableDevice opacity;
  if (use_table) opacity = opacity_table_->DeviceData();
  bool use_mixed_table = use_mixed_opacity_table_;
  MixedOpacityTableDevice mixed_opacity;
  if (use_mixed_table) mixed_opacity = mixed_opacity_table_->DeviceData();
  bool use_materials = use_material_mixture_;
  auto mixture = material_mixture_;
  auto diag = diagnostics;
  const Real pressure_floor = material_pressure_floor;
  const Real temperature_floor = material_temperature_floor;
  const bool nonlinear_source = nonlinear_source_;
  const Real nonlinear_tolerance = source_nonlinear_tolerance_;
  const Real nonlinear_absolute_tolerance =
      source_nonlinear_absolute_tolerance_;
  const int max_iterations = source_max_iterations_;
  const int fallback_substeps = source_fallback_substeps_;
  const bool source_report = source_report_ && nonlinear_source_;
  auto integer_stats = source_integer_stats_.d_view;
  auto real_stats = source_real_stats_.d_view;

  source_iterations_last_solve = 0;
  source_fallbacks_last_solve = 0;
  source_residual_last_solve = 0.0;
  if (source_report) {
    Kokkos::deep_copy(source_integer_stats_.d_view, 0);
    Kokkos::deep_copy(source_real_stats_.d_view, 0.0);
  }

  par_for("thermal_rad_couple", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real density = cons(m, IDN, k, j, i);
    const Real eele_old = fmax(cons(m, ie, k, j, i), 0.0);
    materials::MaterialComposition composition;
    Real tele_old;
    if (use_materials) {
      composition = mixture.CompositionFromConserved(cons, m, k, j, i);
      tele_old = temperature(m, 1, k, j, i);
    } else {
      tele_old = gm1*eele_old/(density*fe);
    }

    Real eele_floor = 0.0;
    if (use_materials &&
        (nonlinear_source || mixture.UsesTabularEOS())) {
      const materials::MaterialPressureEnergyState floor_state =
          mixture.MinimumPressureEnergyState(
              density, composition, pressure_floor, temperature_floor);
      eele_floor = density*floor_state.electron_specific_internal_energy;
    }

    LaggedSourceResult source_result;
    bool used_fallback = false;
    int iterations = 0;
    Real relative_residual = 0.0;

    if (!nonlinear_source) {
      source_result = ApplyLaggedSourceSubsteps(
          1, dt, density, eele_old, eele_floor, tele_old, chat, arad,
          gm1, fe, ng, i0, m, k, j, i, cons, prim, bounds, ka, ke,
          use_table, opacity, use_mixed_table, mixed_opacity,
          use_materials, mixture, composition);
    } else {
      Real old_radiation = 0.0;
      for (int g = 0; g < ng; ++g) {
        old_radiation += fmax(cons(m, i0+g, k, j, i), 0.0);
      }
      const Real local_energy = eele_old+old_radiation;
      const Real coupling_depth = dt*chat*density;
      const CoupledSourceEvaluation warm = EvaluateCoupledSource(
          tele_old, eele_old, density, coupling_depth, arad,
          gm1, fe, ng, i0, m, k, j, i, cons, prim, bounds, ka, ke,
          use_table, opacity, use_mixed_table, mixed_opacity,
          use_materials, mixture, composition, false);
      // Scale by material energy and available absorbed heat. An unconstrained
      // emission estimate can greatly exceed the entire electron reservoir and must
      // not loosen convergence; uncoupled background groups contribute nothing.
      const Real residual_scale = fmax(
          fmax(eele_old, -warm.radiation_delta), energy_floor);
      const Real energy_tolerance = nonlinear_absolute_tolerance+
          nonlinear_tolerance*residual_scale;

      Real temperature_low = 0.0;
      Real temperature_high = 0.0;
      if (use_materials && mixture.UsesTabularEOS()) {
        temperature_low = fmax(
            mixture.MinimumTransportTemperature(composition), temperature_floor);
        if (eele_floor > 0.0) {
          temperature_low = fmax(temperature_low, mixture.ElectronTemperature(
              density, eele_floor/density, composition));
        }
        temperature_high = mixture.MaximumTransportTemperature(composition);
      } else {
        const Real local_fraction = use_materials
            ? mixture.ElectronHeatCapacityFraction(composition) : fe;
        const Real heat_capacity = density*local_fraction/gm1;
        temperature_low = (heat_capacity > 0.0)
            ? eele_floor/heat_capacity : 0.0;
        temperature_high = (heat_capacity > 0.0)
            ? local_energy/heat_capacity : 0.0;
      }

      bool converged = false;
      bool bracketed = false;
      Real root_temperature = fmin(fmax(
          tele_old, temperature_low), temperature_high);
      Real previous_temperature = root_temperature;
      Real previous_residual = 1.0/kRealEpsilon;
      bool have_previous_temperature = false;
      CoupledSourceEvaluation low;
      CoupledSourceEvaluation high;
      if (Kokkos::isfinite(temperature_low) &&
          Kokkos::isfinite(temperature_high) &&
          temperature_high >= temperature_low && temperature_high >= 0.0) {
        low = EvaluateCoupledSource(
            temperature_low, eele_old, density, coupling_depth, arad,
            gm1, fe, ng, i0, m, k, j, i, cons, prim, bounds, ka, ke,
            use_table, opacity, use_mixed_table, mixed_opacity,
            use_materials, mixture, composition, false);
        high = EvaluateCoupledSource(
            temperature_high, eele_old, density, coupling_depth, arad,
            gm1, fe, ng, i0, m, k, j, i, cons, prim, bounds, ka, ke,
            use_table, opacity, use_mixed_table, mixed_opacity,
            use_materials, mixture, composition, false);
        const bool finite_low = Kokkos::isfinite(low.residual);
        const bool finite_high = Kokkos::isfinite(high.residual);
        if (finite_low && fabs(low.residual) <= energy_tolerance) {
          converged = true;
          root_temperature = temperature_low;
        } else if (finite_high && fabs(high.residual) <= energy_tolerance) {
          converged = true;
          root_temperature = temperature_high;
        } else if (finite_low && finite_high &&
                   ((low.residual < 0.0 && high.residual > 0.0) ||
                    (low.residual > 0.0 && high.residual < 0.0))) {
          bracketed = true;
        }
      }

      // The synchronized old temperature is normally close to the root for a resolved
      // source step.  Probe it once to tighten the bracket before false-position steps.
      if (!converged && bracketed && tele_old > temperature_low &&
          tele_old < temperature_high && Kokkos::isfinite(tele_old)) {
        ++iterations;
        if (Kokkos::isfinite(warm.residual)) {
          if (fabs(warm.residual) <= energy_tolerance) {
            converged = true;
            root_temperature = tele_old;
          } else if ((warm.residual > 0.0) == (low.residual > 0.0)) {
            temperature_low = tele_old;
            low = warm;
          } else {
            temperature_high = tele_old;
            high = warm;
          }
          previous_temperature = tele_old;
          previous_residual = fabs(warm.residual);
          have_previous_temperature = true;
        }
      }

      // Illinois-weighted false position retains a sign-changing bracket but avoids
      // repeatedly pinning the same endpoint on strongly curved EOS/T^4 residuals.
      Real weighted_low_residual = low.residual;
      Real weighted_high_residual = high.residual;
      int previously_moved_endpoint = -1;
      for (; iterations < max_iterations && !converged && bracketed;
           ++iterations) {
        const Real low_magnitude = fabs(weighted_low_residual);
        const Real high_magnitude = fabs(weighted_high_residual);
        Real secant_fraction = 0.5;
        if (low_magnitude > 0.0 && high_magnitude > 0.0) {
          if (low_magnitude < high_magnitude) {
            const Real ratio = low_magnitude/high_magnitude;
            secant_fraction = ratio/(1.0+ratio);
          } else {
            const Real ratio = high_magnitude/low_magnitude;
            secant_fraction = 1.0/(1.0+ratio);
          }
        }
        Real trial_temperature = (temperature_low > 0.0)
            ? exp(0.5*(log(temperature_low)+log(temperature_high)))
            : 0.5*(temperature_low+temperature_high);
        const Real secant_temperature = temperature_low+
            secant_fraction*(temperature_high-temperature_low);
        if (Kokkos::isfinite(secant_temperature) &&
            secant_temperature > temperature_low &&
            secant_temperature < temperature_high) {
          trial_temperature = secant_temperature;
        }
        if (!(trial_temperature > temperature_low) ||
            !(trial_temperature < temperature_high) ||
            !Kokkos::isfinite(trial_temperature)) {
          if (have_previous_temperature &&
              previous_residual <= energy_tolerance) {
            converged = true;
            root_temperature = previous_temperature;
          } else {
            bracketed = false;
          }
          break;
        }
        const CoupledSourceEvaluation trial = EvaluateCoupledSource(
            trial_temperature, eele_old, density, coupling_depth, arad,
            gm1, fe, ng, i0, m, k, j, i, cons, prim, bounds, ka, ke,
            use_table, opacity, use_mixed_table, mixed_opacity,
            use_materials, mixture, composition, false);
        root_temperature = trial_temperature;
        if (!Kokkos::isfinite(trial.residual)) {
          bracketed = false;
          break;
        }
        if (trial.residual == 0.0) {
          converged = true;
          break;
        }
        const Real temperature_scale = fmax(
            fabs(trial_temperature), fmax(temperature_low, 1.0e-30));
        const Real relative_step = have_previous_temperature
            ? fabs(trial_temperature-previous_temperature)/temperature_scale
            : 1.0/kRealEpsilon;
        if ((trial.residual > 0.0) == (low.residual > 0.0)) {
          temperature_low = trial_temperature;
          low = trial;
          weighted_low_residual = trial.residual;
          weighted_high_residual = (previously_moved_endpoint == 0)
              ? 0.5*weighted_high_residual : high.residual;
          previously_moved_endpoint = 0;
        } else {
          temperature_high = trial_temperature;
          high = trial;
          weighted_high_residual = trial.residual;
          weighted_low_residual = (previously_moved_endpoint == 1)
              ? 0.5*weighted_low_residual : low.residual;
          previously_moved_endpoint = 1;
        }
        if (fabs(trial.residual) <= energy_tolerance &&
            relative_step <= nonlinear_tolerance) {
          converged = true;
        }
        previous_temperature = trial_temperature;
        previous_residual = fabs(trial.residual);
        have_previous_temperature = true;
      }

      CoupledSourceEvaluation final_state;
      if (converged) {
        final_state = EvaluateCoupledSource(
            root_temperature, eele_old, density, coupling_depth, arad,
            gm1, fe, ng, i0, m, k, j, i, cons, prim, bounds, ka, ke,
            use_table, opacity, use_mixed_table, mixed_opacity,
            use_materials, mixture, composition, true);
        relative_residual = fabs(final_state.residual)/residual_scale;
        const Real eele_candidate =
            eele_old-final_state.radiation_delta;
        bool valid = Kokkos::isfinite(final_state.electron_energy) &&
            Kokkos::isfinite(final_state.radiation_energy) &&
            Kokkos::isfinite(eele_candidate) &&
            fabs(final_state.residual) <= energy_tolerance &&
            eele_candidate >= eele_floor-energy_tolerance;
        for (int g = 0; g < ng; ++g) {
          const Real value = prim(m, i0+g, k, j, i);
          valid = valid && Kokkos::isfinite(value) && value >= 0.0;
        }

        if (valid) {
          source_result.electron_energy = fmax(eele_candidate, eele_floor);
          const Real floor_correction = source_result.electron_energy-eele_candidate;
          // A floor correction can be at most the nonlinear tolerance.  Rescale the
          // cached spectrum by that tiny amount so local conservation remains exact.
          if (floor_correction > final_state.radiation_energy) {
            valid = false;
          } else {
            const Real radiation_scale = (final_state.radiation_energy > 0.0)
                ? 1.0-floor_correction/final_state.radiation_energy : 1.0;
            valid = Kokkos::isfinite(radiation_scale) && radiation_scale >= 0.0;
            if (valid) {
              source_result.radiation_energy = 0.0;
              for (int g = 0; g < ng; ++g) {
                const Real value = radiation_scale*prim(m, i0+g, k, j, i);
                cons(m, i0+g, k, j, i) = value;
                prim(m, i0+g, k, j, i) = value/density;
                source_result.radiation_energy += value;
              }
              source_result.electron_temperature = root_temperature;
            }
          }
        }
        converged = valid;
      }

      if (!converged) {
        used_fallback = true;
        Real endpoint_residual = 1.0/kRealEpsilon;
        if (Kokkos::isfinite(low.residual)) {
          endpoint_residual = fabs(low.residual)/residual_scale;
        }
        if (Kokkos::isfinite(high.residual)) {
          endpoint_residual = fmin(
              endpoint_residual, fabs(high.residual)/residual_scale);
        }
        relative_residual = endpoint_residual;
        source_result = ApplyLaggedSourceSubsteps(
            fallback_substeps, dt, density, eele_old, eele_floor,
            tele_old, chat, arad, gm1, fe, ng, i0, m, k, j, i,
            cons, prim, bounds, ka, ke, use_table, opacity,
            use_mixed_table, mixed_opacity, use_materials, mixture,
            composition);
      }
    }

    const Real eele_new = source_result.electron_energy;
    Real matter_delta = eele_new-eele_old;
    cons(m, ie, k, j, i) = eele_new;
    prim(m, ie, k, j, i) = eele_new/density;
    cons(m, IEN, k, j, i) += matter_delta;
    prim(m, IEN, k, j, i) += matter_delta;
    if (nonlinear_source && !used_fallback &&
        use_materials && mixture.UsesTabularEOS()) {
      temperature(m, 1, k, j, i) = source_result.electron_temperature;
    } else if (use_materials) {
      temperature(m, 1, k, j, i) = mixture.ElectronTemperature(
          density, eele_new/density, composition);
    } else {
      temperature(m, 1, k, j, i) = gm1*eele_new/(density*fe);
    }
    if (source_report) {
      Kokkos::atomic_max(&integer_stats(0), iterations);
      if (used_fallback) Kokkos::atomic_add(&integer_stats(1), 1);
      const Real bounded_residual = Kokkos::isfinite(relative_residual)
          ? relative_residual : 1.0/kRealEpsilon;
      Kokkos::atomic_max(&real_stats(0), bounded_residual);
    }
    diag(m, 0, k, j, i) = source_result.radiation_energy/density;
    diag(m, 1, k, j, i) = pow(source_result.radiation_energy/arad, 0.25);
  });

  if (source_report) {
    source_integer_stats_.modify_device();
    source_real_stats_.modify_device();
    source_integer_stats_.sync_host();
    source_real_stats_.sync_host();
    source_iterations_last_solve = source_integer_stats_.h_view(0);
    source_fallbacks_last_solve = source_integer_stats_.h_view(1);
    source_residual_last_solve = source_real_stats_.h_view(0);
#if MPI_PARALLEL_ENABLED
    RadiationBatchAllreduce(MPI_IN_PLACE, &source_iterations_last_solve, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    RadiationBatchAllreduce(MPI_IN_PLACE, &source_fallbacks_last_solve, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    RadiationBatchAllreduce(MPI_IN_PLACE, &source_residual_last_solve, 1,
                  MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "# nonlinear thermal radiation source: max_iterations="
                << source_iterations_last_solve
                << " fallback_cells=" << source_fallbacks_last_solve
                << " max_relative_residual=" << source_residual_last_solve
                << std::endl;
    }
  }
  source_timer.Stop();
  solver_timers_.Report(pmy_pack_->pmesh->ncycle, dt, global_variable::my_rank, "source");
}

//----------------------------------------------------------------------------------------
//! Compute the explicit FLD stability limit and an optional source-accuracy limit.
//!
//! The transport limit is obtained from the differential (Jacobian) response of the
//! actual face-limited flux, not from the optically thick upper bound 1/(3 sigma).
//! For constant D this reduces exactly to the usual Cartesian diffusion condition.  In
//! the streaming limit it instead becomes a causal c_* dt/dx condition.  Maxima are
//! accumulated independently in each direction and then summed, which is conservative
//! for variable coefficients, multiple groups, and multidimensional meshes.

void ThermalRadiation::NewTimeStep(
    const DvceArray5D<Real> &w0,
    const DvceArray5D<Real> &temperature) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int ng = ngroups;
  int i0 = ifirst;
  int ie = iele_;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size;
  auto kt = kappa_transport_.d_view;
  auto ka = kappa_absorption_.d_view;
  auto kem = kappa_emission_.d_view;
  bool use_table = use_opacity_table_;
  OpacityTableDevice opacity;
  if (use_table) opacity = opacity_table_->DeviceData();
  bool use_mixed_table = use_mixed_opacity_table_;
  MixedOpacityTableDevice mixed_opacity;
  if (use_mixed_table) mixed_opacity = mixed_opacity_table_->DeviceData();
  bool use_materials = use_material_mixture_;
  auto mixture = material_mixture_;
  auto bounds = group_bounds_.d_view;
  Real chat = chat_;
  Real alpha = flux_limit_coefficient_;
  Real floor = energy_floor_;
  Real arad = arad_;
  Real gm1 = gamma_minus_one_;
  Real fe = cv_e_fraction_;
  Real source_cfl = source_cfl_;
  bool couple = couple_matter_;
  int mode = limiter_mode_;
  Real streaming_threshold = ap_streaming_threshold_;
  Real optical_depth_threshold = ap_optical_depth_threshold_;
  bool use_ap_transport = use_ap_transport_;

  int nmb = pmy_pack_->nmb_thispack;

  Real transport_dt = FLT_MAX;
  if (!implicit_transport_) {
  // Each directional reduction finds the largest single-face contribution to the
  // diagonal update rate.  Multiplying their sum by two below accounts for the two
  // faces per cell and recovers dt <= [2 c D sum(dx_d^-2)]^-1 for constant diffusion.
  Real max_rate1 = 0.0;
  int nface1 = nx3*nx2*(nx1+1);
  int total_faces1 = nmb*nface1;
  Kokkos::parallel_reduce("thermal_rad_newdt_x1",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, total_faces1),
  KOKKOS_LAMBDA(const int idx, Real &max_rate) {
    int face_idx = idx%nface1;
    int m = idx/nface1;
    int ii = face_idx%(nx1+1);
    int jk = face_idx/(nx1+1);
    int j = jk%nx2 + js;
    int k = jk/nx2 + ks;
    int i = ii + is;
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    Real dx_short = dx1;
    if (multi_d) dx_short = fmin(dx_short, dx2);
    if (three_d) dx_short = fmin(dx_short, dx3);
    const FLDFaceMaterialState material = X1FaceMaterialState(
        w0, temperature, m, ie, k, j, i, gm1, fe, use_materials, mixture);
    OpacityTableLocation opacity_location;
    MixedOpacityTableLocation mixed_opacity_location;
    if (use_mixed_table) {
      mixed_opacity_location = mixed_opacity.Locate(
          material.density, material.electron_temperature,
          material.composition);
    } else if (use_table) {
      opacity_location = opacity.Locate(
          material.density, material.electron_temperature);
    }
    for (int g = 0; g < ng; ++g) {
      const FLDRadiationFaceState state = X1RadiationFaceState(
          w0, m, i0+g, k, j, i, multi_d, three_d, dx1, dx2, dx3);
      const Real kappa = use_mixed_table ? mixed_opacity.Get(
          opacity_transport, g, mixed_opacity_location) : (use_table ? opacity.Get(
          opacity_transport, g, opacity_location) : kt(g));
      const Real sigma = material.density*kappa;
      const FLDLinearization properties = FLDProperties(
          sigma, state.energy, state.gradient, state.normal_gradient,
          alpha, floor, mode);
      const bool use_ap_face = use_ap_transport && mode != 0 &&
          (properties.streaming_fraction >= streaming_threshold ||
           sigma*dx1 <= optical_depth_threshold);
      const Real rate = FLDFaceStabilityRate(
          properties, state.energy, state.normal_gradient,
          dx1, dx_short, alpha, floor, mode, use_ap_face);
      max_rate = fmax(max_rate, rate);
    }
  }, Kokkos::Max<Real>(max_rate1));

  Real max_rate2 = 0.0;
  if (multi_d) {
    int nface2 = nx3*(nx2+1)*nx1;
    int total_faces2 = nmb*nface2;
    Kokkos::parallel_reduce("thermal_rad_newdt_x2",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, total_faces2),
    KOKKOS_LAMBDA(const int idx, Real &max_rate) {
      int face_idx = idx%nface2;
      int m = idx/nface2;
      int i = face_idx%nx1 + is;
      int jk = face_idx/nx1;
      int j = jk%(nx2+1) + js;
      int k = jk/(nx2+1) + ks;
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      Real dx_short = fmin(dx1, dx2);
      if (three_d) dx_short = fmin(dx_short, dx3);
      const FLDFaceMaterialState material = X2FaceMaterialState(
          w0, temperature, m, ie, k, j, i, gm1, fe, use_materials, mixture);
      OpacityTableLocation opacity_location;
      MixedOpacityTableLocation mixed_opacity_location;
      if (use_mixed_table) {
        mixed_opacity_location = mixed_opacity.Locate(
            material.density, material.electron_temperature,
            material.composition);
      } else if (use_table) {
        opacity_location = opacity.Locate(
            material.density, material.electron_temperature);
      }
      for (int g = 0; g < ng; ++g) {
        const FLDRadiationFaceState state = X2RadiationFaceState(
            w0, m, i0+g, k, j, i, three_d, dx1, dx2, dx3);
        const Real kappa = use_mixed_table ? mixed_opacity.Get(
            opacity_transport, g, mixed_opacity_location) : (use_table ? opacity.Get(
            opacity_transport, g, opacity_location) : kt(g));
        const Real sigma = material.density*kappa;
        const FLDLinearization properties = FLDProperties(
            sigma, state.energy, state.gradient, state.normal_gradient,
            alpha, floor, mode);
        const bool use_ap_face = use_ap_transport && mode != 0 &&
            (properties.streaming_fraction >= streaming_threshold ||
             sigma*dx2 <= optical_depth_threshold);
        const Real rate = FLDFaceStabilityRate(
            properties, state.energy, state.normal_gradient,
            dx2, dx_short, alpha, floor, mode, use_ap_face);
        max_rate = fmax(max_rate, rate);
      }
    }, Kokkos::Max<Real>(max_rate2));
  }

  Real max_rate3 = 0.0;
  if (three_d) {
    int nface3 = (nx3+1)*nx2*nx1;
    int total_faces3 = nmb*nface3;
    Kokkos::parallel_reduce("thermal_rad_newdt_x3",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, total_faces3),
    KOKKOS_LAMBDA(const int idx, Real &max_rate) {
      int face_idx = idx%nface3;
      int m = idx/nface3;
      int i = face_idx%nx1 + is;
      int jk = face_idx/nx1;
      int j = jk%nx2 + js;
      int k = jk/nx2 + ks;
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      Real dx_short = fmin(dx1, fmin(dx2, dx3));
      const FLDFaceMaterialState material = X3FaceMaterialState(
          w0, temperature, m, ie, k, j, i, gm1, fe, use_materials, mixture);
      OpacityTableLocation opacity_location;
      MixedOpacityTableLocation mixed_opacity_location;
      if (use_mixed_table) {
        mixed_opacity_location = mixed_opacity.Locate(
            material.density, material.electron_temperature,
            material.composition);
      } else if (use_table) {
        opacity_location = opacity.Locate(
            material.density, material.electron_temperature);
      }
      for (int g = 0; g < ng; ++g) {
        const FLDRadiationFaceState state = X3RadiationFaceState(
            w0, m, i0+g, k, j, i, dx1, dx2, dx3);
        const Real kappa = use_mixed_table ? mixed_opacity.Get(
            opacity_transport, g, mixed_opacity_location) : (use_table ? opacity.Get(
            opacity_transport, g, opacity_location) : kt(g));
        const Real sigma = material.density*kappa;
        const FLDLinearization properties = FLDProperties(
            sigma, state.energy, state.gradient, state.normal_gradient,
            alpha, floor, mode);
        const bool use_ap_face = use_ap_transport && mode != 0 &&
            (properties.streaming_fraction >= streaming_threshold ||
             sigma*dx3 <= optical_depth_threshold);
        const Real rate = FLDFaceStabilityRate(
            properties, state.energy, state.normal_gradient,
            dx3, dx_short, alpha, floor, mode, use_ap_face);
        max_rate = fmax(max_rate, rate);
      }
    }, Kokkos::Max<Real>(max_rate3));
  }

  Real transport_rate = 2.0*chat*(max_rate1 + max_rate2 + max_rate3);
  transport_dt = (transport_rate > 0.0) ? 1.0/transport_rate : FLT_MAX;
  }

  // The source update is implicit and positivity preserving, but retain the configured
  // fractional electron-energy limit for accuracy.  It is reduced separately so source
  // coupling remains active even when transport is in the free-streaming regime.
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;
  int ncell = nmb*nkji;
  Real source_dt = FLT_MAX;
  if (couple && source_cfl > 0.0) {
    Kokkos::parallel_reduce("thermal_rad_newdt_source",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(const int idx, Real &min_dt) {
      int m = idx/nkji;
      int k = (idx-m*nkji)/nji;
      int j = (idx-m*nkji-k*nji)/nx1;
      int i = idx-m*nkji-k*nji-j*nx1;
      i += is;
      j += js;
      k += ks;
      Real density = w0(m, IDN, k, j, i);
      Real cell_dt = FLT_MAX;
      Real source_rate = 0.0;
      materials::MaterialComposition composition;
      Real tele;
      if (use_materials) {
        composition = mixture.CompositionFromPrimitive(w0, m, k, j, i);
        tele = temperature(m, 1, k, j, i);
      } else {
        tele = gm1*w0(m, ie, k, j, i)/fe;
      }
      Real blackbody = arad*tele*tele*tele*tele;
      PlanckSpectrum planck(bounds(0), tele);
      OpacityTableLocation opacity_location;
      if (use_table) opacity_location = opacity.Locate(density, tele);
      MixedOpacityTableLocation mixed_location;
      if (use_mixed_table) {
        mixed_location = mixed_opacity.Locate(density, tele, composition);
      }

      for (int g = 0; g < ng; ++g) {
        int n = i0+g;
        Real energy = density*w0(m, n, k, j, i);
        const Real fraction = planck.Fraction(bounds(g+1));
        Real equilibrium = blackbody*fraction;
        Real kappaa = use_mixed_table ? mixed_opacity.Get(
            opacity_absorption, g, mixed_location) : (use_table ? opacity.Get(
            opacity_absorption, g, opacity_location) : ka(g));
        Real kappae = use_mixed_table ? mixed_opacity.Get(
            opacity_emission, g, mixed_location) : (use_table ? opacity.Get(
            opacity_emission, g, opacity_location) : kem(g));
        source_rate += chat*fabs(density*kappae*equilibrium
                                 - density*kappaa*energy);
      }
      if (source_rate > 0.0) {
        Real eele = density*w0(m, ie, k, j, i);
        cell_dt = fmin(cell_dt, source_cfl*fmax(eele, floor)/source_rate);
      }
      min_dt = fmin(min_dt, cell_dt);
    }, Kokkos::Min<Real>(source_dt));
  }
  dtnew = fmin(transport_dt, source_dt);
}

} // namespace two_temperature
