#ifndef MHD_BIERMANN_BATTERY_HPP_
#define MHD_BIERMANN_BATTERY_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file biermann_battery.hpp
//! \brief FLASH-style flux formulation of the Biermann battery for 2T MHD.

#include "athena.hpp"
#include "materials/material_mixture.hpp"

class MeshBlockPack;
class ParameterInput;

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class BiermannBattery
//! \brief Adds E_B = -C_B grad(p_e)/n_e to constrained transport and
//! consistently evolves the two-temperature electron energy.

class BiermannBattery {
 public:
  BiermannBattery(MeshBlockPack *ppack, ParameterInput *pin, int electron_index,
                  Real electron_heat_capacity_fraction, Real gamma,
                  Real density_floor, Real pressure_floor,
                  materials::MaterialMixture *material_mixture = nullptr);
  ~BiermannBattery() = default;

  Real coefficient;
  Real dtnew;
  bool suppress_in_shocks;
  Real shock_threshold;
  Real shock_compression_threshold;

  // Construct face-centered Biermann electric fields and add the corresponding
  // Poynting/electron-energy fluxes to the finite-volume update.
  void AddFluxes(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                 DvceFaceFld5D<Real> &flx);

  // Flux-CT average of the face electric fields onto the staggered CT edges.
  void AddEMFs(DvceEdgeFld4D<Real> &efld);

  // Production composite-grid mortar used by the dedicated subcycle.  Fine edge
  // segments are shifted so their exact FC restriction equals one coarse-grid
  // Biermann edge field evaluated from synchronized coarse conserved state.  This
  // covers all 2-D/3-D edge orientations, material EOS closures, neutral activation,
  // and face/edge refinement neighbors.
  void ReconcileCompositeAMREMFs(DvceEdgeFld4D<Real> &efld);

  // Add E_B x B on every active face after the CT edge field has been reconciled
  // and communicated.  AddFluxes supplies only electron drift/enthalpy transport
  // on the dedicated subcycle path; ordinary CC reflux sees the complete result.
  void AddPoyntingFluxFromEdgeEMF(
      const DvceArray5D<Real> &bcc, const DvceEdgeFld4D<Real> &efld,
      DvceFaceFld5D<Real> &flx);

  // Non-conservative p_e div(v_e-v) work paired with electron internal-energy
  // drift.
  void ApplyElectronWork(Real dt, DvceArray5D<Real> &cons,
                         DvceArray5D<Real> &prim);

  // Add -p_e div(v_e-v) as an explicit RHS contribution.  This is the form required
  // by the SSPRK2 subcycle; the legacy exponential map above is retained unchanged for
  // the ordinary stage-coupled update.
  void AddElectronWorkRHS(Real dt, DvceArray5D<Real> &cons,
                          const DvceArray5D<Real> &prim);

  // Direct coarse/fine correction of v_e-v.  It is refluxed as its own face field;
  // inferring it from the electron-energy flux is invalid when subface upwind states
  // differ.
  bool DirectDriftCorrectionEnabled() const;
  DvceFaceFld5D<Real> *DriftCorrectionFlux();
  void UseCorrectedDriftFlux();

  // Electron-drift and FLASH thermal-magnetic-wave stability limit.
  void NewTimeStep(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc);

  // Implementation kernel.  This is public because CUDA forbids extended device
  // lambdas in private member functions.
  void ComputeShockMask(const DvceArray5D<Real> &prim);

 private:
  MeshBlockPack *pmy_pack_;
  int iele_;
  Real electron_fraction_;
  Real gamma_;
  Real gamma_minus_one_;
  Real density_floor_;
  Real pressure_floor_;
  Real temperature_floor_;
  Real minimum_electron_fraction_;
  bool use_material_mixture_;
  materials::MaterialMixtureDevice material_mixture_;

  // Diagonal shock-suppression operator.  Each component is one in smooth flow
  // and attenuates only the matching pressure-gradient/current component in a
  // shock, reducing unrelated cross-component mask coupling.  A local diagonal
  // mask is not curl-free for general curved or oblique shock geometry.
  DvceArray4D<Real> smooth_x1_, smooth_x2_, smooth_x3_;

  // Biermann electric fields on coordinate faces.  The names match MHD's ideal
  // face-EMF scratch arrays (e2x1 means E2 on x1 faces, and so on).
  DvceArray4D<Real> e3x1_, e2x1_;
  DvceArray4D<Real> e1x2_, e3x2_;
  DvceArray4D<Real> e2x3_, e1x3_;

  // Normal electron-ion drift velocity on each coordinate face.
  DvceArray4D<Real> vd1_, vd2_, vd3_;
  DvceFaceFld5D<Real> vd_flux_;

  // Dimension-independent endpoint state for the path-conservative edge
  // integral -C*int(dp_e/n_e).
  DvceArray4D<Real> pressure_vertex_, electron_density_vertex_;
};

} // namespace mhd

#endif // MHD_BIERMANN_BATTERY_HPP_
