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

  // Construct face-centered Biermann electric fields and add the corresponding
  // Poynting/electron-energy fluxes to the finite-volume update.
  void AddFluxes(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                 DvceFaceFld5D<Real> &flx);

  // Flux-CT average of the face electric fields onto the staggered CT edges.
  void AddEMFs(DvceEdgeFld4D<Real> &efld);

  // Non-conservative p_e div(v_e-v) work paired with electron internal-energy
  // drift.
  void ApplyElectronWork(Real dt, DvceArray5D<Real> &cons,
                         DvceArray5D<Real> &prim);

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
  Real minimum_electron_fraction_;
  bool use_material_mixture_;
  materials::MaterialMixtureDevice material_mixture_;

  // Cell mask is one in smooth flow and zero in symmetrically detected shocks.
  DvceArray4D<Real> smooth_cell_;

  // Biermann electric fields on coordinate faces.  The names match MHD's ideal
  // face-EMF scratch arrays (e2x1 means E2 on x1 faces, and so on).
  DvceArray4D<Real> e3x1_, e2x1_;
  DvceArray4D<Real> e1x2_, e3x2_;
  DvceArray4D<Real> e2x3_, e1x3_;

  // Normal electron-ion drift velocity on each coordinate face.
  DvceArray4D<Real> vd1_, vd2_, vd3_;
};

} // namespace mhd

#endif // MHD_BIERMANN_BATTERY_HPP_
