//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation.cpp
//! \brief Explicit multigroup FLD and electron-radiation energy exchange.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "two_temperature/opacity_table.hpp"
#include "two_temperature/thermal_radiation.hpp"

namespace two_temperature {
namespace {

constexpr Real kPlanckIntegralInfinity = 6.4939394022668291491;  // pi^4/15

#if SINGLE_PRECISION_ENABLED
constexpr Real kRealEpsilon = FLT_EPSILON;
#else
constexpr Real kRealEpsilon = DBL_EPSILON;
#endif

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

// Integral_0^x t^3/(exp(t)-1) dt.  The small-x expansion avoids cancellation, while
// the exponentially convergent complementary series is accurate over the rest of the
// range and is suitable for device execution.
KOKKOS_INLINE_FUNCTION
Real PlanckIntegral(Real x) {
  if (x <= 0.0) return 0.0;
  if (x >= 50.0) return kPlanckIntegralInfinity;
  if (x < 0.5) {
    Real x2 = x*x;
    Real x3 = x2*x;
    return x3/3.0 - x3*x/8.0 + x3*x2/60.0
           - x3*x2*x2/5040.0 + x3*x2*x2*x2/272160.0
           - x3*x2*x2*x2*x2/13305600.0;
  }

  // exp(-n*x) is a geometric sequence: one exp plus a running multiply replaces the 64
  // independent transcendentals this loop used to evaluate.  The terms fall off like
  // e^(-n*x) with x >= 0.5 here, so the series is also truncated as soon as a term can no
  // longer change the double-precision sum -- typically after a handful of steps.
  Real tail = 0.0;
  const Real q = exp(-x);
  const Real x2 = x*x;
  const Real x3 = x2*x;
  Real qn = q;
  for (int n = 1; n <= 64; ++n) {
    const Real invn = 1.0/static_cast<Real>(n);
    const Real invn2 = invn*invn;
    const Real term = qn*(x3*invn + 3.0*x2*invn2
                          + 6.0*x*invn2*invn + 6.0*invn2*invn2);
    tail += term;
    if (term <= 1.0e-17*tail) break;
    qn *= q;
  }
  return fmin(fmax(kPlanckIntegralInfinity - tail, 0.0),
              kPlanckIntegralInfinity);
}

KOKKOS_INLINE_FUNCTION
Real PlanckGroupFraction(Real lower_bound, Real upper_bound, Real temperature) {
  if (temperature <= 0.0) return 0.0;
  Real fraction = (PlanckIntegral(upper_bound/temperature)
                   - PlanckIntegral(lower_bound/temperature))
                  /kPlanckIntegralInfinity;
  return fmin(fmax(fraction, 0.0), 1.0);
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
  state.gradient = sqrt(grad1*grad1+grad2*grad2+grad3*grad3);
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
  state.gradient = sqrt(grad1*grad1+grad2*grad2+grad3*grad3);
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
  state.gradient = sqrt(grad1*grad1+grad2*grad2+grad3*grad3);
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
    kappa_emission_("thermal-radiation-kappa-emission", 1) {
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
}

//----------------------------------------------------------------------------------------

ThermalRadiation::~ThermalRadiation() {
  if (opacity_table_ != nullptr) delete opacity_table_;
  if (mixed_opacity_table_ != nullptr) delete mixed_opacity_table_;
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
    // Roll the lower boundary forward instead of re-evaluating it: a 20-group cell needs
    // 21 Planck integrals, not 40.  This is the same construction the source-limit
    // reducer already uses below.
    Real lower_planck = (trad > 0.0) ? PlanckIntegral(bounds(0)/trad) : 0.0;
    for (int g = 0; g < ng; ++g) {
      Real fraction = 0.0;
      if (trad > 0.0) {
        const Real upper_planck = PlanckIntegral(bounds(g+1)/trad);
        fraction = fmin(fmax(
            (upper_planck-lower_planck)/kPlanckIntegralInfinity, 0.0), 1.0);
        lower_planck = upper_planck;
      }
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
                                 DvceFaceFld5D<Real> &flx) {
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

  auto flx1 = flx.x1f;
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

  auto flx2 = flx.x2f;
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

  auto flx3 = flx.x3f;
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
//! Apply FLASH-style time-lagged Planck emission and implicit group absorption.
//!
//! The sum of radiation changes is removed from the electron and material total energies.
//! Positive emission is scaled only when necessary to prevent a negative electron energy.

void ThermalRadiation::Couple(Real dt, DvceArray5D<Real> &cons,
    DvceArray5D<Real> &prim, DvceArray5D<Real> &temperature,
    Real material_pressure_floor, Real material_temperature_floor,
    int il, int iu, int jl, int ju, int kl, int ku) {
  if (!couple_matter_ || dt <= 0.0) {
    UpdateDiagnostics(cons, prim, il, iu, jl, ju, kl, ku);
    return;
  }

  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  int ie = iele_;
  Real gm1 = gamma_minus_one_;
  Real fe = cv_e_fraction_;
  Real arad = arad_;
  Real chat = chat_;
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

  par_for("thermal_rad_couple", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real density = cons(m, IDN, k, j, i);
    Real eele_old = fmax(cons(m, ie, k, j, i), 0.0);
    materials::MaterialComposition composition;
    Real tele;
    if (use_materials) {
      composition = mixture.CompositionFromConserved(cons, m, k, j, i);
      tele = temperature(m, 1, k, j, i);
    } else {
      tele = gm1*eele_old/(density*fe);
    }
    OpacityTableLocation opacity_location;
    if (use_table) opacity_location = opacity.Locate(density, tele);
    MixedOpacityTableLocation mixed_location;
    if (use_mixed_table) {
      mixed_location = mixed_opacity.Locate(density, tele, composition);
    }
    Real blackbody = arad*tele*tele*tele*tele;
    Real positive = 0.0;
    Real negative = 0.0;
    // Each group boundary is shared with its neighbour, so rolling the lower Planck
    // integral forward evaluates 21 integrals for 20 groups instead of 40.  Identical
    // construction to the source-limit reducer further down this file.
    Real lower_planck = (tele > 0.0) ? PlanckIntegral(bounds(0)/tele) : 0.0;

    for (int g = 0; g < ng; ++g) {
      Real old = fmax(cons(m, i0+g, k, j, i), 0.0);
      Real kappaa = use_mixed_table ? mixed_opacity.Get(
          opacity_absorption, g, mixed_location) : (use_table ? opacity.Get(
          opacity_absorption, g, opacity_location) : ka(g));
      Real kappae = use_mixed_table ? mixed_opacity.Get(
          opacity_emission, g, mixed_location) : (use_table ? opacity.Get(
          opacity_emission, g, opacity_location) : ke(g));
      Real siga = density*kappaa;
      Real sige = density*kappae;
      Real group_fraction = 0.0;
      if (tele > 0.0) {
        const Real upper_planck = PlanckIntegral(bounds(g+1)/tele);
        group_fraction = fmin(fmax(
            (upper_planck-lower_planck)/kPlanckIntegralInfinity, 0.0), 1.0);
        lower_planck = upper_planck;
      }
      Real source = sige*blackbody*group_fraction;
      Real updated = (old + dt*chat*source)/(1.0 + dt*chat*siga);
      // Cache the unscaled update in the primitive slot.  Radiation-group
      // primitives are not read in Couple(), and every slot is overwritten
      // with its final specific energy in the second loop below.
      prim(m, i0+g, k, j, i) = updated;
      Real delta = updated-old;
      if (delta > 0.0) positive += delta;
      if (delta < 0.0) negative += delta;
    }

    Real eele_floor = 0.0;
    if (use_materials && mixture.UsesTabularEOS()) {
      const materials::MaterialPressureEnergyState floor_state =
          mixture.MinimumPressureEnergyState(
              density, composition, pressure_floor, temperature_floor);
      eele_floor = density*floor_state.electron_specific_internal_energy;
    }
    // Absorbed radiation is immediately available, but tabular emission may not draw
    // the electron component below the same table/pressure/temperature floor as Sync.
    Real available = fmax(eele_old-eele_floor-negative, 0.0);
    Real emission_scale = (positive > available && positive > 0.0)
        ? available/positive : 1.0;
    Real total_delta = 0.0;
    Real total_radiation = 0.0;
    for (int g = 0; g < ng; ++g) {
      Real old = fmax(cons(m, i0+g, k, j, i), 0.0);
      Real updated = prim(m, i0+g, k, j, i);
      Real delta = updated-old;
      if (delta > 0.0) delta *= emission_scale;
      Real value = old+delta;
      cons(m, i0+g, k, j, i) = value;
      prim(m, i0+g, k, j, i) = value/density;
      total_delta += delta;
      total_radiation += value;
    }

    Real eele_new = fmax(eele_old-total_delta, eele_floor);
    Real matter_delta = eele_new-eele_old;
    cons(m, ie, k, j, i) = eele_new;
    prim(m, ie, k, j, i) = eele_new/density;
    cons(m, IEN, k, j, i) += matter_delta;
    prim(m, IEN, k, j, i) += matter_delta;
    if (!use_materials) {
      temperature(m, 1, k, j, i) = gm1*eele_new/(density*fe);
    } else if (!mixture.UsesTabularEOS()) {
      temperature(m, 1, k, j, i) =
          mixture.ElectronTemperature(density, eele_new/density, composition);
    }
    diag(m, 0, k, j, i) = total_radiation/density;
    diag(m, 1, k, j, i) = pow(total_radiation/arad, 0.25);
  });
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
  Real transport_dt = (transport_rate > 0.0) ? 1.0/transport_rate : FLT_MAX;

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
      Real lower_planck = 0.0;
      if (tele > 0.0) lower_planck = PlanckIntegral(bounds(0)/tele);
      OpacityTableLocation opacity_location;
      if (use_table) opacity_location = opacity.Locate(density, tele);
      MixedOpacityTableLocation mixed_location;
      if (use_mixed_table) {
        mixed_location = mixed_opacity.Locate(density, tele, composition);
      }

      for (int g = 0; g < ng; ++g) {
        int n = i0+g;
        Real energy = density*w0(m, n, k, j, i);
        Real fraction = 0.0;
        if (tele > 0.0) {
          Real upper_planck = PlanckIntegral(bounds(g+1)/tele);
          fraction = fmin(fmax(
              (upper_planck-lower_planck)/kPlanckIntegralInfinity, 0.0), 1.0);
          lower_planck = upper_planck;
        }
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
