//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dci_3d.cpp
//! \brief Three-dimensional laser drive of an open CH/Au target in a He chamber.
//!
//! The deck selects the carrier fluid: a <mhd> block runs the full 3T MHD problem with
//! the Biermann battery, while a <hydro> block runs exactly the same target, laser, and
//! radiation physics with no magnetic field at all.  Everything below is written against
//! whichever module the input file created.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "hydro/hydro.hpp"
#include "laser/laser.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/biermann_battery.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"

namespace {

constexpr int kHistoryFields = 22;

void DCIHistory(HistoryData *pdata, Mesh *pm);
void VacuumRadiationBoundary(Mesh *pm);

[[noreturn]] void ProblemError(const char *message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

bool NearlyEqual(const Real lhs, const Real rhs, const Real scale = 1.0) {
  return std::abs(lhs-rhs) <= 1.0e-12*std::fmax(scale, 1.0);
}

//----------------------------------------------------------------------------------------
//! Carrier-fluid view of the problem: the members below are the only fluid state this
//! problem generator touches, so the rest of the file never branches on <hydro>/<mhd>.
//! Face and cell-centred magnetic fields are left empty in the hydrodynamic case; every
//! use is guarded by is_mhd.

struct DCIFluid {
  bool is_mhd = false;
  EquationOfState *peos = nullptr;
  two_temperature::TwoTemperature *ptwo = nullptr;
  materials::MaterialMixture *pmaterials = nullptr;
  mhd::BiermannBattery *pbiermann = nullptr;
  bool use_dual_energy = false;
  int nuser_scalars = 0;
  DvceArray5D<Real> u0, w0;
  DvceArray5D<Real> flx1, flx2, flx3;
  DvceArray5D<Real> bcc0;
  DvceArray4D<Real> b0x1f, b0x2f, b0x3f;
};

DCIFluid SelectFluid(MeshBlockPack *pmbp) {
  DCIFluid fluid;
  if (pmbp->pmhd != nullptr && pmbp->phydro != nullptr) {
    ProblemError("dci_3d requires exactly one of the <mhd> and <hydro> blocks");
  }
  if (pmbp->pmhd != nullptr) {
    auto *pmhd = pmbp->pmhd;
    fluid.is_mhd = true;
    fluid.peos = pmhd->peos;
    fluid.ptwo = pmhd->ptwo_temp;
    fluid.pmaterials = pmhd->pmaterials;
    fluid.pbiermann = pmhd->pbiermann;
    fluid.use_dual_energy = pmhd->use_dual_energy;
    fluid.nuser_scalars = pmhd->nuser_scalars;
    fluid.u0 = pmhd->u0;
    fluid.w0 = pmhd->w0;
    fluid.flx1 = pmhd->uflx.x1f;
    fluid.flx2 = pmhd->uflx.x2f;
    fluid.flx3 = pmhd->uflx.x3f;
    fluid.bcc0 = pmhd->bcc0;
    fluid.b0x1f = pmhd->b0.x1f;
    fluid.b0x2f = pmhd->b0.x2f;
    fluid.b0x3f = pmhd->b0.x3f;
  } else if (pmbp->phydro != nullptr) {
    auto *phydro = pmbp->phydro;
    fluid.peos = phydro->peos;
    fluid.ptwo = phydro->ptwo_temp;
    fluid.pmaterials = phydro->pmaterials;
    fluid.use_dual_energy = phydro->use_dual_energy;
    fluid.nuser_scalars = phydro->nuser_scalars;
    fluid.u0 = phydro->u0;
    fluid.w0 = phydro->w0;
    fluid.flx1 = phydro->uflx.x1f;
    fluid.flx2 = phydro->uflx.x2f;
    fluid.flx3 = phydro->uflx.x3f;
  } else {
    ProblemError("dci_3d requires an <mhd> or <hydro> block");
  }
  return fluid;
}

} // namespace

//----------------------------------------------------------------------------------------
//! Initialize the CH spherical cap and its conservative material tracer.
//!
//! The smooth geometry variable alpha is a volume fraction.  The passive scalar is a
//! mass fraction: rho*X_CH = alpha*rho_CH.  This distinction is important across the
//! smoothed solid/ambient interface where the component densities differ by five orders
//! of magnitude.  The material closure evaluates the CH, Au, and He tables at their partial
//! densities and initializes every component at the requested common physical
//! temperature.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  static_assert(kHistoryFields <= NHISTORY_VARIABLES,
                "DCI history exceeds AthenaK's reduction capacity");
  user_hist_func = DCIHistory;
  user_bcs_func = VacuumRadiationBoundary;
  // Fluid faces use ordinary outflow flags.  Force the post-physical-BC callback so it
  // can replace only the embedded FLD group ghost cells with a vacuum state.
  user_bcs = true;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->plaser == nullptr) {
    ProblemError("dci_3d requires a <laser> block");
  }
  DCIFluid fluid = SelectFluid(pmbp);
  auto *ptwo = fluid.ptwo;
  if (ptwo == nullptr || ptwo->pradiation == nullptr) {
    ProblemError("dci_3d requires a two-temperature fluid and thermal radiation");
  }
  // The Biermann battery is a magnetic-field source, so it exists only in the MHD deck.
  // The hydrodynamic deck is the same problem with B identically zero.
  if (fluid.is_mhd && fluid.pbiermann == nullptr) {
    ProblemError("dci_3d MHD mode requires the Biermann battery");
  }
  if (!fluid.use_dual_energy) {
    ProblemError("dci_3d requires dual energy");
  }
  if (!pmy_mesh_->three_d) {
    ProblemError("dci_3d requires a three-dimensional Cartesian mesh");
  }
  // Each material is explicit: rho*Y_CH, rho*Y_Au, and rho*Y_He are all advected.
  if (fluid.nuser_scalars != 3) {
    ProblemError("dci_3d requires exactly three user scalars for CH, Au, and He");
  }
  if (fluid.pmaterials == nullptr) {
    ProblemError("dci_3d requires the three-material CH/Au/He closure");
  }
  if (fluid.pmaterials->NumberOfMaterials() != 3) {
    ProblemError("dci_3d requires <materials>/nmaterials=3 (CH shell, Au cone, He)");
  }
  if (!fluid.pmaterials->UsesTabularEOS()) {
    ProblemError("dci_3d requires the audited CH, Au, and He two-temperature EOS tables");
  }
  if (ptwo->pradiation->ngroups != 20) {
    ProblemError("dci_3d requires the 20 reference radiation groups");
  }
  if (!fluid.peos->eos_data.is_gamma_law) {
    ProblemError("dci_3d tabular material mode requires its gamma-law carrier");
  }
  if (fluid.is_mhd &&
      !NearlyEqual(fluid.pbiermann->coefficient,
                   2.7875722321043606e-4,
                   2.7875722321043606e-4)) {
    ProblemError("dci_3d requires the material-number-density Biermann normalization");
  }

  // Target geometry transcribed from 3d_zb/Simulation_initBlock.F90 with the FLASH cone
  // axis +z mapped to +x1.  Lengths are code mm (0.1 cm), so the archive's cm constants
  // are multiplied by ten:
  //
  //   CH shell (FOAM_SPEC): 0.52 < R <= 0.55 mm spherical cap, |polar| <= 50 deg
  //   Au cone (CONE_SPEC):  hollow cone on the same 50-degree surface, x1 from 0.085 mm
  //                         up to the cap edge, wall 0.031 mm thick measured in
  //                         cylindrical
  //                         radius (the archive's lxa offset)
  //   He ambient (CHAM_SPEC): everything else
  const Real ambient_density =
      pin->GetOrAddReal("problem", "ambient_density", 9.090909090909091e-6);
  const Real solid_density =
      pin->GetOrAddReal("problem", "solid_density", 1.0);
  const Real cone_density =
      pin->GetOrAddReal("problem", "cone_density", 17.454545454545453);
  const Real temperature_kelvin =
      pin->GetOrAddReal("problem", "temperature_kelvin", 11606.0);
  const Real inner_radius =
      pin->GetOrAddReal("problem", "inner_radius", 0.52);
  const Real outer_radius =
      pin->GetOrAddReal("problem", "outer_radius", 0.55);
  const Real opening_half_angle_deg =
      pin->GetOrAddReal("problem", "opening_half_angle_deg", 50.0);
  const Real cone_base_x1 =
      pin->GetOrAddReal("problem", "cone_base_x1", 0.085);
  const Real cone_wall_offset =
      pin->GetOrAddReal("problem", "cone_wall_offset", 0.031);
  const Real transition_width =
      pin->GetOrAddReal("problem", "transition_width", 5.0e-3);
  const Real angular_transition =
      pin->GetOrAddReal("problem", "angular_transition", 1.0e-2);
  if (!(ambient_density > 0.0) || !(solid_density > ambient_density) ||
      !(cone_density > solid_density) ||
      !(temperature_kelvin > 0.0) || !(inner_radius > 0.0) ||
      !(outer_radius > inner_radius) || !(opening_half_angle_deg > 0.0) ||
      !(opening_half_angle_deg < 90.0) || !(transition_width > 0.0) ||
      !(angular_transition > 0.0) || !(cone_base_x1 > 0.0) ||
      !(cone_wall_offset > 0.0)) {
    ProblemError("dci_3d geometry and thermodynamic parameters are invalid");
  }
  if (!NearlyEqual(solid_density, 1.0) ||
      !NearlyEqual(cone_density, 17.454545454545453, 17.454545454545453) ||
      !NearlyEqual(ambient_density, 9.090909090909091e-6) ||
      !NearlyEqual(temperature_kelvin, 11606.0, 11606.0) ||
      !NearlyEqual(inner_radius, 0.52) || !NearlyEqual(outer_radius, 0.55) ||
      !NearlyEqual(opening_half_angle_deg, 50.0, 50.0) ||
      !NearlyEqual(cone_base_x1, 0.085) ||
      !NearlyEqual(cone_wall_offset, 0.031)) {
    ProblemError("dci_3d target must be the archived 1.1 g/cc CH cap from 0.52--0.55 mm, "
                 "a 19.2 g/cc Au cone on the same 50-degree surface, 1e-5 g/cc He, "
                 "and 11606 K");
  }

  // The drive is the audited FLASH 3d_zb/ParDir/1l_4beam_BB.par laser mapped with the
  // cone axis +z -> +x1: four collimated 0.275 mm uniform 3-omega beams at the four
  // diagonal azimuths, 50 degrees off axis, sharing the archived 13-section picket
  // pulse whose per-beam integral is 1.774550 kJ (7.0982 kJ total).
  const int nbeams = pin->GetInteger("laser", "nbeams");
  const int npulses = pin->GetInteger("laser", "npulses");
  const int max_reflections =
      pin->GetInteger("laser", "max_reflections_per_ray");
  const int max_mpi_waves = pin->GetInteger("laser", "max_mpi_waves");
  const Real reflection_offset =
      pin->GetReal("laser", "reflection_offset_fraction");
  const Real reflection_hysteresis =
      pin->GetReal("laser", "reflection_hysteresis_fraction");
  const bool allow_laser_transport_variants = pin->GetOrAddBoolean(
      "problem", "allow_laser_transport_variants", false);
  if (nbeams != 4 || npulses != 1 ||
      pin->GetInteger("laser", "pulse0_nsections") != 13) {
    ProblemError("dci_3d requires the archived four-beam single-pulse FLASH drive");
  }
  Real pulse_energy = 0.0;  // erg, from code-ns knots and erg/s powers
  Real peak_power = 0.0;
  Real previous_time = 0.0;
  Real previous_power = 0.0;
  for (int s = 0; s < 13; ++s) {
    const Real knot_time =
        pin->GetReal("laser", "pulse0_time_" + std::to_string(s));
    const Real knot_power =
        pin->GetReal("laser", "pulse0_power_" + std::to_string(s));
    if (s > 0) {
      pulse_energy += 0.5*(knot_power+previous_power)*(knot_time-previous_time)*1.0e-9;
    }
    peak_power = std::fmax(peak_power, knot_power);
    previous_time = knot_time;
    previous_power = knot_power;
  }
  const Real reference_pulse_energy = 1.774550e10;  // erg per beam
  const bool pulse_matches =
      std::abs(pulse_energy-reference_pulse_energy) <=
          1.0e-6*reference_pulse_energy &&
      NearlyEqual(peak_power, 6.875e18, 6.875e18) &&
      NearlyEqual(previous_time, 4.71, 4.71) &&
      NearlyEqual(previous_power, 0.0);
  bool beams_match = true;
  bool quadrant_used[2][2] = {{false, false}, {false, false}};
  for (int b = 0; b < nbeams; ++b) {
    const std::string beam = "beam" + std::to_string(b) + "_";
    const Real lens_x2 = pin->GetReal("laser", beam + "lens_x2");
    const Real lens_x3 = pin->GetReal("laser", beam + "lens_x3");
    beams_match = beams_match &&
        pin->GetString("laser", beam + "geometry") == "lens" &&
        pin->GetString("laser", beam + "profile") == "uniform" &&
        NearlyEqual(pin->GetReal("laser", beam + "wavelength"),
                    3.51e-5, 3.51e-5) &&
        NearlyEqual(pin->GetReal("laser", beam + "lens_x1"), 2.037, 2.037) &&
        NearlyEqual(std::abs(lens_x2), 1.4142, 1.4142) &&
        NearlyEqual(std::abs(lens_x3), 1.4142, 1.4142) &&
        NearlyEqual(pin->GetReal("laser", beam + "target_x1"), 0.359) &&
        NearlyEqual(pin->GetReal("laser", beam + "target_x2"), 0.0) &&
        NearlyEqual(pin->GetReal("laser", beam + "target_x3"), 0.0) &&
        NearlyEqual(pin->GetReal("laser", beam + "aperture_radius"), 0.275) &&
        NearlyEqual(pin->GetReal("laser", beam + "target_radius"), 0.275) &&
        pin->GetInteger("laser", beam + "pulse_number") == 0 &&
        pin->GetInteger("laser", beam + "nrays") >= 4096 &&
        NearlyEqual(pin->GetReal("laser", beam + "start_time"), 0.0) &&
        NearlyEqual(pin->GetReal("laser", beam + "end_time"), 5.0, 5.0);
    quadrant_used[lens_x2 > 0.0 ? 1 : 0][lens_x3 > 0.0 ? 1 : 0] = true;
  }
  beams_match = beams_match &&
      quadrant_used[0][0] && quadrant_used[0][1] &&
      quadrant_used[1][0] && quadrant_used[1][1];
  const Real pi = std::acos(-1.0);
  const bool production_transport =
      max_reflections == 64 && max_mpi_waves >= 64 &&
      NearlyEqual(reflection_offset, 1.0e-5) &&
      NearlyEqual(reflection_hysteresis, 1.0e-2);
  const bool diagnostic_transport =
      max_reflections > 0 && max_mpi_waves >= 64 &&
      reflection_offset >= 1.0e-7 && reflection_offset <= 1.0e-3 &&
      reflection_hysteresis >= 0.0 && reflection_hysteresis < 0.1;
  if (!beams_match || !pulse_matches ||
      !(allow_laser_transport_variants ? diagnostic_transport
                                      : production_transport)) {
    ProblemError("dci_3d requires the archived FLASH 3d_zb four-beam collimated "
                 "uniform 3-omega picket drive and chatter-controlled "
                 "ray-transport limits");
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int scalar_index = fluid.pmaterials->ScalarIndex();
  const int cone_scalar_index = scalar_index+1;
  const int helium_scalar_index = scalar_index+2;
  const auto material_mixture = fluid.pmaterials->DeviceData();
  const Real code_temperature =
      material_mixture.CodeTemperatureFromKelvin(temperature_kelvin);
  auto size = pmbp->pmb->mb_size;
  auto w0 = fluid.w0;
  const Real half_angle = opening_half_angle_deg*pi/180.0;
  const Real cos_half_angle = std::cos(half_angle);
  // The archive writes the cone surface as |z| = tan(90-angle/2)*rr, i.e. the same
  // 50-degree half-angle as the CH cap edge, and truncates it at the cap's own edge.
  const Real cone_slope = std::tan(0.5*pi-half_angle);
  const Real cone_tip_x1 = outer_radius*cos_half_angle;

  Kokkos::deep_copy(w0, 0.0);
  if (fluid.is_mhd) {
    Kokkos::deep_copy(fluid.bcc0, 0.0);
    Kokkos::deep_copy(fluid.b0x1f, 0.0);
    Kokkos::deep_copy(fluid.b0x2f, 0.0);
    Kokkos::deep_copy(fluid.b0x3f, 0.0);
  }

  par_for("pgen_dci_3d", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
    const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
    const Real radius = sqrt(x1*x1 + x2*x2 + x3*x3);
    const Real cylindrical_radius = sqrt(x2*x2 + x3*x3);
    const Real axis_cosine = (radius > 0.0) ? x1/radius : -1.0;

    // CH spherical cap: a shell in radius intersected with the polar cone.
    const Real inside_outer =
        0.5*(1.0 - tanh((radius-outer_radius)/transition_width));
    const Real outside_inner =
        0.5*(1.0 + tanh((radius-inner_radius)/transition_width));
    const Real inside_cap = 0.5*(1.0 +
        tanh((axis_cosine-cos_half_angle)/angular_transition));
    const Real alpha_ch = inside_outer*outside_inner*inside_cap;

    // Au cone: the wall between the cap's own 50-degree surface and a parallel surface
    // offset inward by cone_wall_offset in cylindrical radius, truncated below the CH
    // cap edge and above the archive's flat base.
    const Real inside_cone_outer = 0.5*(1.0 -
        tanh((x1-cone_slope*cylindrical_radius)/transition_width));
    const Real outside_cone_inner = 0.5*(1.0 +
        tanh((x1-cone_slope*(cylindrical_radius-cone_wall_offset))/
             transition_width));
    const Real above_cone_base =
        0.5*(1.0 + tanh((x1-cone_base_x1)/transition_width));
    const Real below_cone_tip =
        0.5*(1.0 - tanh((x1-cone_tip_x1)/transition_width));
    // The cap and the cone touch along the same conical surface, so remove any overlap
    // from the cone rather than double-counting solid mass there.
    const Real alpha_cone = fmax(
        inside_cone_outer*outside_cone_inner*above_cone_base*below_cone_tip-
            alpha_ch, 0.0);

    const Real ch_partial_density = alpha_ch*solid_density;
    const Real cone_partial_density = alpha_cone*cone_density;
    const Real ambient_fraction = fmax(1.0-alpha_ch-alpha_cone, 0.0);
    const Real ambient_partial_density = ambient_fraction*ambient_density;
    const Real density =
        ch_partial_density + cone_partial_density + ambient_partial_density;
    const Real ch_mass_fraction = ch_partial_density/density;
    const Real cone_mass_fraction = cone_partial_density/density;
    const Real helium_mass_fraction = ambient_partial_density/density;
    Real fractions[3] = {
        ch_mass_fraction, cone_mass_fraction, helium_mass_fraction};
    const auto composition = material_mixture.CompositionFromFractions(fractions);
    const auto thermo = material_mixture.StateFromRhoTemperatures(
        density, code_temperature, code_temperature, composition);
    const Real internal_energy = density*(thermo.ion_specific_internal_energy +
                                          thermo.electron_specific_internal_energy);

    w0(m, IDN, k, j, i) = density;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = internal_energy;
    w0(m, scalar_index, k, j, i) = ch_mass_fraction;
    w0(m, cone_scalar_index, k, j, i) = cone_mass_fraction;
    w0(m, helium_scalar_index, k, j, i) = helium_mass_fraction;
  });

  if (fluid.is_mhd) {
    pmbp->pmhd->peos->PrimToCons(
        w0, fluid.bcc0, fluid.u0, is, ie, js, je, ks, ke);
  } else {
    pmbp->phydro->peos->PrimToCons(w0, fluid.u0, is, ie, js, je, ks, ke);
  }
}

namespace {

//----------------------------------------------------------------------------------------
//! Apply a zero-energy (Dirichlet vacuum) boundary to embedded FLD groups only.
//!
//! AthenaK first applies the ordinary outflow condition to all MHD variables.  This
//! callback then replaces radiation group ghost cells at global faces.  A zero-gradient
//! outflow condition would instead be an insulated FLD wall.

void VacuumRadiationBoundary(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  const DCIFluid fluid = SelectFluid(pmbp);
  auto *prad = fluid.ptwo->pradiation;
  auto u0 = fluid.u0;
  auto mb_bcs = pmbp->pmb->mb_bcs;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = indcs.nx2 + 2*ng;
  const int n3 = indcs.nx3 + 2*ng;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int ifirst = prad->ifirst;
  const int ngroups = prad->ngroups;

  par_for("dci_vacuum_rad_x1", DevExeSpace(), 0, nmb1, 0, ngroups-1,
          0, n3-1, 0, n2-1,
  KOKKOS_LAMBDA(int m, int g, int k, int j) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, j, is-n-1) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, j, ie+n+1) = 0.0;
    }
  });

  par_for("dci_vacuum_rad_x2", DevExeSpace(), 0, nmb1, 0, ngroups-1,
          0, n3-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int g, int k, int i) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, js-n-1, i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, je+n+1, i) = 0.0;
    }
  });

  par_for("dci_vacuum_rad_x3", DevExeSpace(), 0, nmb1, 0, ngroups-1,
          0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int g, int j, int i) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, ks-n-1, j, i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, ke+n+1, j, i) = 0.0;
    }
  });
}

//----------------------------------------------------------------------------------------
//! Integrated physics and conservation diagnostics (MPI reduction is done by writer).

void DCIHistory(HistoryData *pdata, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  DCIFluid fluid = SelectFluid(pmbp);
  auto *ptwo = fluid.ptwo;
  auto *prad = ptwo->pradiation;

  pdata->nhist = kHistoryFields;
  pdata->label[0] = "laser_Edep";
  pdata->label[1] = "laser_Pdep";
  pdata->label[2] = "rad_Pesc";
  pdata->label[3] = "mass";
  pdata->label[4] = "CH_mass";
  pdata->label[5] = "mat_E";
  pdata->label[6] = "kin_E";
  // The magnetic columns are retained (as identical zeros) when the deck runs the
  // hydrodynamic carrier so that the two modes produce comparable history files.
  pdata->label[7] = "mag_E";
  pdata->label[8] = "eion_E";
  pdata->label[9] = "eele_E";
  pdata->label[10] = "erad_E";
  pdata->label[11] = "erad_soft";
  pdata->label[12] = "erad_mid";
  pdata->label[13] = "erad_hard";
  pdata->label[14] = "chain_E";
  pdata->label[15] = "abs_B";
  pdata->label[16] = "laser_x";
  pdata->label[17] = "rad_x";
  pdata->label[18] = "eos_floor";
  pdata->label[19] = "eos_bad";
  pdata->label[20] = "divB";
  pdata->label[21] = "Au_mass";

  const bool have_magnetic = fluid.is_mhd;
  auto u0 = fluid.u0;
  auto bcc0 = fluid.bcc0;
  auto b0x1f = fluid.b0x1f;
  auto b0x2f = fluid.b0x2f;
  auto b0x3f = fluid.b0x3f;
  auto flx1 = fluid.flx1;
  auto flx2 = fluid.flx2;
  auto flx3 = fluid.flx3;
  auto mb_bcs = pmbp->pmb->mb_bcs;
  auto laser_data = pmbp->plaser->cell_data;
  auto thermodynamics = ptwo->thermodynamics;
  auto size = pmbp->pmb->mb_size;
  const int scalar_index = fluid.pmaterials->ScalarIndex();
  const int cone_scalar_index = scalar_index+1;
  const int iion = ptwo->iion;
  const int iele = ptwo->iele;
  const int ifirst = prad->ifirst;
  const int ngroups = prad->ngroups;
  constexpr int disallowed_eos_flags =
      materials::ionmix_density_above_table |
      materials::ionmix_temperature_below_table |
      materials::ionmix_temperature_above_table |
      materials::ionmix_energy_above_table;
  const bool have_flux = pm->ncycle > 0;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  array_sum::GlobalSum sum_this_rank;

  Kokkos::parallel_reduce(
      "dci_3d_history", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, array_sum::GlobalSum &sum) {
        const int m = idx/nkji;
        const int k = (idx-m*nkji)/nji + ks;
        const int j = (idx-m*nkji-(k-ks)*nji)/nx1 + js;
        const int i = idx-m*nkji-(k-ks)*nji-(j-js)*nx1 + is;
        const Real dx1 = size.d_view(m).dx1;
        const Real dx2 = size.d_view(m).dx2;
        const Real dx3 = size.d_view(m).dx3;
        const Real volume = dx1*dx2*dx3;
        const Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min,
                                    size.d_view(m).x1max);
        const Real density = u0(m, IDN, k, j, i);
        const Real xch = u0(m, scalar_index, k, j, i)/density;
        const Real momentum_squared = SQR(u0(m, IM1, k, j, i)) +
                                      SQR(u0(m, IM2, k, j, i)) +
                                      SQR(u0(m, IM3, k, j, i));
        const Real b1 = have_magnetic ? bcc0(m, IBX, k, j, i) : 0.0;
        const Real b2 = have_magnetic ? bcc0(m, IBY, k, j, i) : 0.0;
        const Real b3 = have_magnetic ? bcc0(m, IBZ, k, j, i) : 0.0;
        const Real b2sum = b1*b1 + b2*b2 + b3*b3;
        Real erad = 0.0;
        Real erad_soft = 0.0;
        Real erad_mid = 0.0;
        Real erad_hard = 0.0;
        Real radiation_escape_power = 0.0;
        for (int g = 0; g < ngroups; ++g) {
          const Real group_energy = u0(m, ifirst+g, k, j, i);
          erad += group_energy;
          if (g <= 6) {
            erad_soft += group_energy;
          } else if (g <= 13) {
            erad_mid += group_energy;
          } else {
            erad_hard += group_energy;
          }
          if (have_flux) {
            if (i == is &&
                mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::outflow) {
              radiation_escape_power -= dx2*dx3*flx1(m, ifirst+g, k, j, is);
            }
            if (i == is+nx1-1 &&
                mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow) {
              radiation_escape_power += dx2*dx3*flx1(m, ifirst+g, k, j, is+nx1);
            }
            if (j == js &&
                mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow) {
              radiation_escape_power -= dx1*dx3*flx2(m, ifirst+g, k, js, i);
            }
            if (j == js+nx2-1 &&
                mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow) {
              radiation_escape_power += dx1*dx3*flx2(m, ifirst+g, k, js+nx2, i);
            }
            if (k == ks &&
                mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::outflow) {
              radiation_escape_power -= dx1*dx2*flx3(m, ifirst+g, ks, j, i);
            }
            if (k == ks+nx3-1 &&
                mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::outflow) {
              radiation_escape_power += dx1*dx2*flx3(m, ifirst+g, ks+nx3, j, i);
            }
          }
        }
        array_sum::GlobalSum local;
        local.the_array[0] = volume*laser_data(m, 1, k, j, i);
        local.the_array[1] = volume*laser_data(m, 0, k, j, i);
        local.the_array[2] = radiation_escape_power;
        local.the_array[3] = volume*density;
        local.the_array[4] = volume*u0(m, scalar_index, k, j, i);
        local.the_array[5] = volume*u0(m, IEN, k, j, i);
        local.the_array[6] = volume*0.5*momentum_squared/density;
        local.the_array[7] = volume*0.5*b2sum;
        local.the_array[8] = volume*u0(m, iion, k, j, i);
        local.the_array[9] = volume*u0(m, iele, k, j, i);
        local.the_array[10] = volume*erad;
        local.the_array[11] = volume*erad_soft;
        local.the_array[12] = volume*erad_mid;
        local.the_array[13] = volume*erad_hard;
        local.the_array[14] = volume*(u0(m, IEN, k, j, i)+erad);
        local.the_array[15] = volume*sqrt(b2sum);
        local.the_array[16] = volume*x1*laser_data(m, 1, k, j, i);
        local.the_array[17] = volume*x1*erad;
        const int eos_flags = static_cast<int>(thermodynamics(
            m, two_temperature::TwoTemperature::eos_query_flags, k, j, i));
        local.the_array[18] =
            ((eos_flags & materials::ionmix_energy_below_table) != 0) ? 1.0 : 0.0;
        local.the_array[19] = ((eos_flags & disallowed_eos_flags) != 0) ? 1.0 : 0.0;
        // Volume-integrated |div B| from the face fields.  Constrained transport keeps
        // this at roundoff, so it is the diagnostic that shows whether a closure or
        // subcycling change has quietly broken the CT invariant.  Identically zero for
        // the hydrodynamic carrier, which stores no magnetic field at all.
        Real divb = 0.0;
        if (have_magnetic) {
          divb = (b0x1f(m, k, j, i+1)-b0x1f(m, k, j, i))/dx1 +
                 (b0x2f(m, k, j+1, i)-b0x2f(m, k, j, i))/dx2 +
                 (b0x3f(m, k+1, j, i)-b0x3f(m, k, j, i))/dx3;
        }
        local.the_array[20] = volume*fabs(divb);
        local.the_array[21] = volume*u0(m, cone_scalar_index, k, j, i);
        sum += local;
      }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_rank));
  Kokkos::fence();

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_rank.the_array[n];
  }
}

} // namespace
