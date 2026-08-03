//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file conduction_implicit.cpp
//! \brief FLASH-style implicit isotropic electron thermal conduction.
//!
//! The solved equation is
//!
//!   rho c_v,e dT_e/dt = div(K_e grad(T_e)).
//!
//! Conductivity and the optional saturated-flux correction are time lagged, matching the
//! linearization documented for FLASH's general implicit diffusion solver.  The electron
//! EOS energy is retained nonlinearly: Newton corrections use a finite-difference heat
//! capacity and each symmetric positive-definite Jacobian system is solved by diagonally
//! preconditioned conjugate gradients.  Solver vectors use AthenaK's normal MeshBlock/MPI
//! halo exchange, so the matrix spans every block on a uniform-level mesh.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "diffusion/conduction.hpp"
#include "globals.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"

namespace {

[[noreturn]] void ImplicitConductionError(const std::string &message) {
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

} // namespace

//----------------------------------------------------------------------------------------
//! Fill conduction-specific physical boundaries.  The diffusion solve deliberately owns
//! these boundary choices instead of inheriting hydrodynamic outflow/inflow semantics.

void Conduction::ApplyPhysicalBoundaries(DvceArray5D<Real> &field,
                                         const bool homogeneous_boundary) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack-1;
  const int n1 = indcs.nx1+2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2+2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3+2*ng : 1;
  auto mb_bcs = pmy_pack->pmb->mb_bcs;
  auto f = field;

  const int b0 = static_cast<int>(boundary_type_[0]);
  const int b1 = static_cast<int>(boundary_type_[1]);
  const Real v0 = homogeneous_boundary ? 0.0 : boundary_value_[0];
  const Real v1 = homogeneous_boundary ? 0.0 : boundary_value_[1];
  // Fill the complete transverse ghost extent, as the ordinary hydro boundaries do.
  // The diffusion stencil itself uses only face ghosts, but the frozen Spitzer limiter
  // takes centered gradients in those ghosts and therefore also reads edge/corner values.
  par_for("cond_impl_bc_x1", DevExeSpace(), 0, nmb1,
          0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int j, int n) {
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::inner_x1))) {
      f(m, 0, k, j, is-n-1) = (b0 == 0)
          ? f(m, 0, k, j, is)
          : 2.0*v0-f(m, 0, k, j, is+n);
    }
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::outer_x1))) {
      f(m, 0, k, j, ie+n+1) = (b1 == 0)
          ? f(m, 0, k, j, ie)
          : 2.0*v1-f(m, 0, k, j, ie-n);
    }
  });

  if (pmy_pack->pmesh->one_d) return;
  const int b2 = static_cast<int>(boundary_type_[2]);
  const int b3 = static_cast<int>(boundary_type_[3]);
  const Real v2 = homogeneous_boundary ? 0.0 : boundary_value_[2];
  const Real v3 = homogeneous_boundary ? 0.0 : boundary_value_[3];
  par_for("cond_impl_bc_x2", DevExeSpace(), 0, nmb1,
          0, n3-1, 0, n1-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int i, int n) {
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::inner_x2))) {
      f(m, 0, k, js-n-1, i) = (b2 == 0)
          ? f(m, 0, k, js, i)
          : 2.0*v2-f(m, 0, k, js+n, i);
    }
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::outer_x2))) {
      f(m, 0, k, je+n+1, i) = (b3 == 0)
          ? f(m, 0, k, je, i)
          : 2.0*v3-f(m, 0, k, je-n, i);
    }
  });

  if (pmy_pack->pmesh->two_d) return;
  const int b4 = static_cast<int>(boundary_type_[4]);
  const int b5 = static_cast<int>(boundary_type_[5]);
  const Real v4 = homogeneous_boundary ? 0.0 : boundary_value_[4];
  const Real v5 = homogeneous_boundary ? 0.0 : boundary_value_[5];
  par_for("cond_impl_bc_x3", DevExeSpace(), 0, nmb1,
          0, n2-1, 0, n1-1, 0, ng-1,
  KOKKOS_LAMBDA(int m, int j, int i, int n) {
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::inner_x3))) {
      f(m, 0, ks-n-1, j, i) = (b4 == 0)
          ? f(m, 0, ks, j, i)
          : 2.0*v4-f(m, 0, ks+n, j, i);
    }
    if (IsPhysicalBoundary(mb_bcs.d_view(m, BoundaryFace::outer_x3))) {
      f(m, 0, ke+n+1, j, i) = (b5 == 0)
          ? f(m, 0, ke, j, i)
          : 2.0*v5-f(m, 0, ke-n, j, i);
    }
  });
}

//----------------------------------------------------------------------------------------
//! Synchronously exchange one solver component.  This is called from an operator-split
//! task after the ordinary stage communication has been cleared, so reusing pbval_u's
//! communicator and first buffer component is safe.

void Conduction::ExchangeSolverField(DvceArray5D<Real> &field,
                                     MeshBoundaryValuesCC *pbval,
                                     const bool homogeneous_boundary) {
  if (pbval->InitRecv(1) != TaskStatus::complete) {
    ImplicitConductionError("Could not initialize implicit-conduction halo receives");
  }
  if (pbval->PackAndSendCC(field, coarse_scratch_, 1) != TaskStatus::complete) {
    ImplicitConductionError("Could not send implicit-conduction halo data");
  }
  TaskStatus status = TaskStatus::incomplete;
  while (status == TaskStatus::incomplete) {
    status = pbval->RecvAndUnpackCC(field, coarse_scratch_, 1);
  }
  if (status != TaskStatus::complete) {
    ImplicitConductionError("Could not receive implicit-conduction halo data");
  }
  if (pbval->ClearSend() != TaskStatus::complete ||
      pbval->ClearRecv() != TaskStatus::complete) {
    ImplicitConductionError("Could not clear implicit-conduction halo communication");
  }
  ApplyPhysicalBoundaries(field, homogeneous_boundary);
}

//----------------------------------------------------------------------------------------
//! Apply div(K grad(field)) with an arithmetic face conductivity, the centered stencil
//! used by FLASH's isotropic implicit solver.

void Conduction::ApplyDiffusionOperator(const DvceArray5D<Real> &field,
                                        DvceArray5D<Real> &result) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack-1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto size = pmy_pack->pmb->mb_size;
  auto kappa = conductivity_;
  auto f = field;
  auto out = result;

  par_for("cond_impl_laplacian", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real kc = kappa(m, 0, k, j, i);
    const Real dx1 = size.d_view(m).dx1;
    const Real kp1 = 0.5*(kc+kappa(m, 0, k, j, i+1));
    const Real km1 = 0.5*(kc+kappa(m, 0, k, j, i-1));
    Real lap = (kp1*(f(m, 0, k, j, i+1)-f(m, 0, k, j, i))-
                km1*(f(m, 0, k, j, i)-f(m, 0, k, j, i-1)))/(dx1*dx1);
    if (multi_d) {
      const Real dx2 = size.d_view(m).dx2;
      const Real kp2 = 0.5*(kc+kappa(m, 0, k, j+1, i));
      const Real km2 = 0.5*(kc+kappa(m, 0, k, j-1, i));
      lap += (kp2*(f(m, 0, k, j+1, i)-f(m, 0, k, j, i))-
              km2*(f(m, 0, k, j, i)-f(m, 0, k, j-1, i)))/(dx2*dx2);
    }
    if (three_d) {
      const Real dx3 = size.d_view(m).dx3;
      const Real kp3 = 0.5*(kc+kappa(m, 0, k+1, j, i));
      const Real km3 = 0.5*(kc+kappa(m, 0, k-1, j, i));
      lap += (kp3*(f(m, 0, k+1, j, i)-f(m, 0, k, j, i))-
              km3*(f(m, 0, k, j, i)-f(m, 0, k-1, j, i)))/(dx3*dx3);
    }
    out(m, 0, k, j, i) = lap;
  });
}

//----------------------------------------------------------------------------------------
//! Apply the frozen-coefficient Newton Jacobian C_v - theta*dt*div(K grad).

void Conduction::ApplyJacobian(const DvceArray5D<Real> &field,
                               DvceArray5D<Real> &result, const Real dt) {
  ApplyDiffusionOperator(field, result);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nmb1 = pmy_pack->nmb_thispack-1;
  auto cv = capacity_;
  auto f = field;
  auto out = result;
  const Real factor = theta_*dt;
  par_for("cond_impl_jacobian", DevExeSpace(), 0, nmb1,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    out(m, 0, k, j, i) =
        cv(m, 0, k, j, i)*f(m, 0, k, j, i)-factor*out(m, 0, k, j, i);
  });
}

//----------------------------------------------------------------------------------------
//! Global interior-cell dot product.

Real Conduction::GlobalDot(const DvceArray5D<Real> &lhs,
                           const DvceArray5D<Real> &rhs) const {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pmy_pack->nmb_thispack*nkji;
  auto a = lhs;
  auto b = rhs;
  Real sum = 0.0;
  Kokkos::parallel_reduce("cond_impl_dot", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int idx, Real &local_sum) {
    const int m = idx/nkji;
    const int local = idx-m*nkji;
    const int k = local/nji+ks;
    const int j = (local-(k-ks)*nji)/nx1+js;
    const int i = local-(k-ks)*nji-(j-js)*nx1+is;
    local_sum += a(m, 0, k, j, i)*b(m, 0, k, j, i);
  }, sum);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return sum;
}

//----------------------------------------------------------------------------------------
//! Solve one operator-split electron-conduction step.

void Conduction::SolveImplicit(const Real dt, DvceArray5D<Real> &cons,
                               DvceArray5D<Real> &prim,
                               MeshBoundaryValuesCC *pbval,
                               two_temperature::TwoTemperature *ptwo_temp) {
  if (!implicit_ || !(dt > 0.0)) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int nmb1 = pmy_pack->nmb_thispack-1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int il = is-1, iu = ie+1;
  const int jl = multi_d ? js-1 : js;
  const int ju = multi_d ? je+1 : je;
  const int kl = three_d ? ks-1 : ks;
  const int ku = three_d ? ke+1 : ke;
  if (ng < 2) {
    ImplicitConductionError("Implicit thermal conduction requires at least two ghosts");
  }

  auto old_temperature = temperature_old_;
  auto new_temperature = temperature_new_;
  auto electron_temperature = ptwo_temp->temperature;
  auto old_energy = energy_old_;
  auto u = cons;
  const int iele = ptwo_temp->iele;
  const int n1 = indcs.nx1+2*ng;
  const int n2 = multi_d ? indcs.nx2+2*ng : 1;
  const int n3 = three_d ? indcs.nx3+2*ng : 1;

  par_for("cond_impl_copy_state", DevExeSpace(), 0, nmb1,
          0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    old_temperature(m, 0, k, j, i) = electron_temperature(m, 1, k, j, i);
    new_temperature(m, 0, k, j, i) = electron_temperature(m, 1, k, j, i);
  });
  par_for("cond_impl_copy_energy", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    old_energy(m, 0, k, j, i) = u(m, iele, k, j, i);
  });
  ExchangeSolverField(temperature_old_, pbval, false);
  Kokkos::deep_copy(DevExeSpace(), temperature_new_, temperature_old_);

  // Freeze K at the old state, following FLASH's time-lagged-coefficient treatment.
  auto kappa = conductivity_;
  auto size = pmy_pack->pmb->mb_size;
  auto thermo = ptwo_temp->thermodynamics;
  const int limiter = static_cast<int>(flux_limiter_);
  const bool use_spitzer = alpha_spitzer;
  const Real constant_alpha = alpha_iso;
  const Real coulomb_log = coulomb_log_;
  const Real multiplier = spitzer_multiplier_;
  const Real temperature_floor_kelvin = spitzer_temperature_floor_kelvin_;
  const Real flux_coefficient = flux_limit_coefficient_;
  materials::MaterialMixtureDevice mixture;
  const bool has_materials = pmaterials_ != nullptr;
  if (has_materials) mixture = pmaterials_->DeviceData();
  Real temperature_to_kelvin = 1.0;
  Real conductivity_unit = 1.0;
  Real heat_flux_unit = 1.0;
  if (use_spitzer) {
    temperature_to_kelvin = mixture.temperature_to_kelvin;
    const Real pressure_unit = pmy_pack->punit->pressure_cgs();
    const Real velocity_unit = pmy_pack->punit->velocity_cgs();
    conductivity_unit = pressure_unit*velocity_unit*
                        pmy_pack->punit->length_cgs()/temperature_to_kelvin;
    heat_flux_unit = pressure_unit*velocity_unit;
  }
  constexpr Real pi = 3.141592653589793238462643383279502884;
  constexpr Real boltzmann_cgs = 1.3806488e-16;
  constexpr Real electron_charge_cgs = 4.803204712570263e-10;
  constexpr Real electron_mass_cgs = 9.1093837015e-28;
  const Real spitzer_constant = pow(8.0/pi, 1.5)*pow(boltzmann_cgs, 3.5)/
      (pow(electron_charge_cgs, 4)*sqrt(electron_mass_cgs));

  par_for("cond_impl_coefficients", DevExeSpace(), 0, nmb1,
          kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real density = u(m, IDN, k, j, i);
    Real coefficient = constant_alpha*density;
    if (use_spitzer) {
      const Real tele_code = fmax(old_temperature(m, 0, k, j, i), 0.0);
      const Real tele_kelvin = fmax(
          tele_code*temperature_to_kelvin, temperature_floor_kelvin);
      const Real zbar = fmax(thermo(
          m, two_temperature::TwoTemperature::mean_ionization, k, j, i), 1.0e-12);
      const Real ne = fmax(thermo(
          m, two_temperature::TwoTemperature::electron_number_density_cgs,
          k, j, i), 0.0);
      const Real classical_cgs = spitzer_constant/(1.0+3.3/zbar)*
          pow(tele_kelvin, 2.5)/(zbar*coulomb_log);
      Real spitzer_code = multiplier*classical_cgs/conductivity_unit;
      if (limiter != 0 && spitzer_code > 0.0 && ne > 0.0 && tele_kelvin > 0.0) {
        const Real dx1 = size.d_view(m).dx1;
        const Real gx = (old_temperature(m, 0, k, j, i+1)-
                         old_temperature(m, 0, k, j, i-1))/(2.0*dx1);
        Real gy = 0.0, gz = 0.0;
        if (multi_d) {
          const Real dx2 = size.d_view(m).dx2;
          gy = (old_temperature(m, 0, k, j+1, i)-
                old_temperature(m, 0, k, j-1, i))/(2.0*dx2);
        }
        if (three_d) {
          const Real dx3 = size.d_view(m).dx3;
          gz = (old_temperature(m, 0, k+1, j, i)-
                old_temperature(m, 0, k-1, j, i))/(2.0*dx3);
        }
        const Real gradient = sqrt(gx*gx+gy*gy+gz*gz);
        const Real qmax_cgs = flux_coefficient*ne*boltzmann_cgs*tele_kelvin*
            sqrt(boltzmann_cgs*tele_kelvin/electron_mass_cgs);
        const Real qmax_code = qmax_cgs/heat_flux_unit;
        if (gradient > 0.0 && qmax_code > 0.0) {
          const Real saturation_inverse = gradient/qmax_code;
          if (limiter == 1) {
            spitzer_code = 1.0/(1.0/spitzer_code+saturation_inverse);
          } else if (limiter == 2) {
            spitzer_code = fmin(spitzer_code, 1.0/saturation_inverse);
          } else {
            spitzer_code = 1.0/sqrt(1.0/(spitzer_code*spitzer_code)+
                                     saturation_inverse*saturation_inverse);
          }
        }
      }
      coefficient += spitzer_code;
    }
    kappa(m, 0, k, j, i) = fmax(coefficient, 0.0);
  });

  ApplyDiffusionOperator(temperature_old_, explicit_laplacian_);

  iterations_last_solve = 0;
  nonlinear_iterations_last_solve = 0;
  residual_last_solve = std::numeric_limits<Real>::infinity();
  bool nonlinear_converged = false;
  const bool tabular = has_materials && pmaterials_->UsesTabularEOS();
  auto capacity = capacity_;
  auto residual = residual_;
  auto old_laplacian = explicit_laplacian_;
  auto ion_temperature = ptwo_temp->temperature;
  const Real theta = theta_;
  const Real gm1 = gamma_minus_one_;
  const Real fixed_fe = electron_heat_capacity_fraction_;
  Real residual_scale = GlobalDot(energy_old_, energy_old_);
  const Real laplacian_scale = dt*dt*GlobalDot(explicit_laplacian_,
                                               explicit_laplacian_);
  residual_scale = fmax(residual_scale, laplacian_scale);
  residual_scale = fmax(residual_scale,
      std::numeric_limits<Real>::min()*static_cast<Real>(
          pmy_pack->nmb_thispack*indcs.nx1*indcs.nx2*indcs.nx3));

  for (int nonlinear_iteration = 0;
       nonlinear_iteration < max_nonlinear_iterations_; ++nonlinear_iteration) {
    nonlinear_iterations_last_solve = nonlinear_iteration+1;
    ExchangeSolverField(temperature_new_, pbval, false);
    ApplyDiffusionOperator(temperature_new_, operator_direction_);
    auto new_laplacian = operator_direction_;

    par_for("cond_impl_eos_residual", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real density = u(m, IDN, k, j, i);
      const Real tele = new_temperature(m, 0, k, j, i);
      Real eele = 0.0;
      Real cv = 0.0;
      if (has_materials) {
        const Real y0 = mixture.Material0MassFractionFromConserved(u, m, k, j, i);
        const Real tion = ion_temperature(m, 0, k, j, i);
        const auto state = mixture.StateFromRhoTemperaturesNoSound(
            density, tion, tele, y0);
        eele = density*state.electron_specific_internal_energy;
        if (tabular) {
          const Real tmin = mixture.MinimumTransportTemperature(y0);
          const Real tmax = mixture.MaximumTransportTemperature(y0);
          const Real step = fmax(fabs(tele)*1.0e-4,
                                 fmax((tmax-tmin)*1.0e-8, 1.0e-12));
          const Real tlo = fmax(tmin, tele-step);
          const Real thi = fmin(tmax, tele+step);
          const auto elo = mixture.StateFromRhoTemperaturesNoSound(
              density, tion, tlo, y0);
          const auto ehi = mixture.StateFromRhoTemperaturesNoSound(
              density, tion, thi, y0);
          const Real span = thi-tlo;
          cv = (span > 0.0)
              ? density*(ehi.electron_specific_internal_energy-
                         elo.electron_specific_internal_energy)/span
              : 0.0;
        } else {
          cv = density*mixture.ElectronHeatCapacityFraction(y0)/gm1;
        }
      } else {
        cv = density*fixed_fe/gm1;
        eele = cv*tele;
      }
      const Real cv_floor = fmax(fabs(eele)/fmax(fabs(tele), 1.0)*1.0e-12,
                                 1.0e-30);
      capacity(m, 0, k, j, i) = fmax(cv, cv_floor);
      residual(m, 0, k, j, i) = eele-old_energy(m, 0, k, j, i)-dt*(
          theta*new_laplacian(m, 0, k, j, i)+
          (1.0-theta)*old_laplacian(m, 0, k, j, i));
    });

    const Real residual_norm = sqrt(GlobalDot(residual_, residual_)/residual_scale);
    residual_last_solve = residual_norm;
    if (!std::isfinite(residual_norm)) {
      ImplicitConductionError("Implicit thermal conduction produced a non-finite "
                              "nonlinear residual");
    }
    if (residual_norm <= nonlinear_tolerance_) {
      nonlinear_converged = true;
      break;
    }

    // Diagonal preconditioner for J = C_v-theta*dt*L.
    auto z = preconditioned_;
    auto p = direction_;
    auto delta = correction_;
    auto jacobian_p = operator_direction_;
    const Real jacobian_factor = theta_*dt;
    par_for("cond_impl_pcg_init", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real kc = kappa(m, 0, k, j, i);
      const Real dx1 = size.d_view(m).dx1;
      Real diffusion_diagonal =
          (0.5*(kc+kappa(m, 0, k, j, i+1))+
           0.5*(kc+kappa(m, 0, k, j, i-1)))/(dx1*dx1);
      if (multi_d) {
        const Real dx2 = size.d_view(m).dx2;
        diffusion_diagonal +=
            (0.5*(kc+kappa(m, 0, k, j+1, i))+
             0.5*(kc+kappa(m, 0, k, j-1, i)))/(dx2*dx2);
      }
      if (three_d) {
        const Real dx3 = size.d_view(m).dx3;
        diffusion_diagonal +=
            (0.5*(kc+kappa(m, 0, k+1, j, i))+
             0.5*(kc+kappa(m, 0, k-1, j, i)))/(dx3*dx3);
      }
      const Real diagonal = capacity(m, 0, k, j, i)+
                            jacobian_factor*diffusion_diagonal;
      residual(m, 0, k, j, i) = -residual(m, 0, k, j, i);
      z(m, 0, k, j, i) = residual(m, 0, k, j, i)/diagonal;
      p(m, 0, k, j, i) = z(m, 0, k, j, i);
      delta(m, 0, k, j, i) = 0.0;
    });

    Real rz = GlobalDot(residual_, preconditioned_);
    const Real rr0 = GlobalDot(residual_, residual_);
    bool linear_converged = (rr0 == 0.0);
    for (int iteration = 0; iteration < max_iterations_ && !linear_converged;
         ++iteration) {
      ExchangeSolverField(direction_, pbval, true);
      ApplyJacobian(direction_, operator_direction_, dt);
      const Real p_ap = GlobalDot(direction_, operator_direction_);
      if (!(p_ap > 0.0) || !std::isfinite(p_ap) || !std::isfinite(rz)) {
        ImplicitConductionError("Implicit-conduction PCG lost positive definiteness");
      }
      const Real alpha = rz/p_ap;
      par_for("cond_impl_pcg_update", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        delta(m, 0, k, j, i) += alpha*p(m, 0, k, j, i);
        residual(m, 0, k, j, i) -= alpha*jacobian_p(m, 0, k, j, i);
      });
      ++iterations_last_solve;
      const Real rr = GlobalDot(residual_, residual_);
      if (rr <= linear_tolerance_*linear_tolerance_*rr0) {
        linear_converged = true;
        break;
      }

      par_for("cond_impl_pcg_precondition", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real kc = kappa(m, 0, k, j, i);
        const Real dx1 = size.d_view(m).dx1;
        Real diffusion_diagonal =
            (0.5*(kc+kappa(m, 0, k, j, i+1))+
             0.5*(kc+kappa(m, 0, k, j, i-1)))/(dx1*dx1);
        if (multi_d) {
          const Real dx2 = size.d_view(m).dx2;
          diffusion_diagonal +=
              (0.5*(kc+kappa(m, 0, k, j+1, i))+
               0.5*(kc+kappa(m, 0, k, j-1, i)))/(dx2*dx2);
        }
        if (three_d) {
          const Real dx3 = size.d_view(m).dx3;
          diffusion_diagonal +=
              (0.5*(kc+kappa(m, 0, k+1, j, i))+
               0.5*(kc+kappa(m, 0, k-1, j, i)))/(dx3*dx3);
        }
        const Real diagonal = capacity(m, 0, k, j, i)+
                              jacobian_factor*diffusion_diagonal;
        z(m, 0, k, j, i) = residual(m, 0, k, j, i)/diagonal;
      });
      const Real rz_new = GlobalDot(residual_, preconditioned_);
      const Real beta = rz_new/rz;
      par_for("cond_impl_pcg_direction", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        p(m, 0, k, j, i) = z(m, 0, k, j, i)+beta*p(m, 0, k, j, i);
      });
      rz = rz_new;
    }
    if (!linear_converged) {
      ImplicitConductionError("Implicit-conduction PCG exceeded conduction_max_iterations");
    }

    Real damping = 1.0;
    if (tabular) {
      Real local_limit = 1.0;
      auto correction = correction_;
      Kokkos::parallel_reduce(
          "cond_impl_newton_bound",
          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
              {0, ks, js, is}, {pmy_pack->nmb_thispack, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(int m, int k, int j, int i, Real &minimum) {
        const Real y0 = mixture.Material0MassFractionFromConserved(u, m, k, j, i);
        const Real tmin = mixture.MinimumTransportTemperature(y0);
        const Real tmax = mixture.MaximumTransportTemperature(y0);
        const Real change = correction(m, 0, k, j, i);
        Real bound = 1.0;
        if (change < 0.0) {
          bound = (new_temperature(m, 0, k, j, i)-tmin)/(-change);
        } else if (change > 0.0) {
          bound = (tmax-new_temperature(m, 0, k, j, i))/change;
        }
        minimum = fmin(minimum, bound);
      }, Kokkos::Min<Real>(local_limit));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &local_limit, 1, MPI_ATHENA_REAL,
                    MPI_MIN, MPI_COMM_WORLD);
#endif
      damping = (local_limit >= 1.0)
          ? 1.0 : 0.99*fmax(local_limit, 0.0);
      if (!(damping > 0.0)) {
        ImplicitConductionError("Implicit-conduction Newton update reached an EOS-table "
                                "temperature bound");
      }
    }
    auto correction = correction_;
    par_for("cond_impl_newton_update", DevExeSpace(), 0, nmb1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      new_temperature(m, 0, k, j, i) +=
          damping*correction(m, 0, k, j, i);
    });
  }

  if (!nonlinear_converged) {
    ImplicitConductionError(
        "Implicit thermal conduction exceeded conduction_max_nonlinear_iterations; "
        "last relative residual="+std::to_string(residual_last_solve));
  }

  // Recompute the converged flux divergence and use it for the conservative update.
  ExchangeSolverField(temperature_new_, pbval, false);
  ApplyDiffusionOperator(temperature_new_, operator_direction_);
  auto new_laplacian = operator_direction_;
  Real net_energy_change = 0.0;
  auto mbsize = pmy_pack->pmb->mb_size;
  auto w = prim;
  const int iion = ptwo_temp->iion;
  Kokkos::parallel_reduce(
      "cond_impl_conservative_update",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, ks, js, is}, {pmy_pack->nmb_thispack, ke+1, je+1, ie+1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum) {
    const Real delta_energy = dt*(
        theta*new_laplacian(m, 0, k, j, i)+
        (1.0-theta)*old_laplacian(m, 0, k, j, i));
    const Real electron_energy = old_energy(m, 0, k, j, i)+delta_energy;
    if (!(electron_energy > 0.0) || !Kokkos::isfinite(electron_energy)) {
      Kokkos::abort("Implicit conduction produced invalid electron energy");
    }
    u(m, iele, k, j, i) = electron_energy;
    u(m, IEN, k, j, i) += delta_energy;
    w(m, iele, k, j, i) = electron_energy/u(m, IDN, k, j, i);
    w(m, IEN, k, j, i) += delta_energy;
    // Ion energy is unchanged by electron conduction.
    w(m, iion, k, j, i) = u(m, iion, k, j, i)/u(m, IDN, k, j, i);
    const Real volume = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*
                        mbsize.d_view(m).dx3;
    sum += volume*delta_energy;
  }, net_energy_change);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &net_energy_change, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif

  // Rebuild component temperatures, pressures, floors, and table diagnostics from the
  // authoritative conservative energies before exchange/radiation consume them.
  ptwo_temp->Sync(cons, prim, is, ie, js, je, ks, ke);

  if (report_ && global_variable::my_rank == 0) {
    std::cout << "# implicit conduction: nonlinear_iterations="
              << nonlinear_iterations_last_solve
              << " pcg_iterations=" << iterations_last_solve
              << " relative_residual=" << residual_last_solve
              << " net_energy_change=" << net_energy_change << std::endl;
  }
}
