//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic mhd

#include "athena.hpp"
#include "materials/material_mixture.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "two_temperature/two_temperature.hpp"

namespace {

// Duplicated in mhd/mhd_dual_energy.cpp (MHDInternalEnergyFloor) and
// bvals/prolong_prims.cpp (BoundaryInternalEnergyFloor); keep the three in sync.
// Unlike the mhd_dual_energy.cpp copy, dens is used as passed (callers here have
// already applied the density floor).
KOKKOS_INLINE_FUNCTION
Real MHDInternalEnergyFloor(const EOS_Data &eos, const Real dens) {
  Real eint_floor = eos.pfloor/(eos.gamma - 1.0);
  if (eos.tfloor > 0.0) {
    eint_floor = fmax(eint_floor, dens*eos.tfloor/(eos.gamma - 1.0));
  }
  if (eos.sfloor > 0.0) {
    eint_floor = fmax(eint_floor, dens*eos.sfloor*pow(dens, eos.gamma - 1.0)/
                                  (eos.gamma - 1.0));
  }
  return eint_floor;
}

} // namespace

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealMHD::IdealMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.sigma_max = pin->GetOrAddReal("mhd","sigma_max",(FLT_MAX));  // sigma ceiling
}

//----------------------------------------------------------------------------------------
//! \!fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.

void IdealMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  const bool use_dual = pmy_pack->pmhd->use_dual_energy;
  const int iion = use_dual ? pmy_pack->pmhd->ptwo_temp->iion : -1;
  const int iele = use_dual ? pmy_pack->pmhd->ptwo_temp->iele : -1;
  const Real dual_eta1 = pmy_pack->pmhd->dual_energy_eta1;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  const bool use_materials = pmy_pack->pmhd->pmaterials != nullptr;
  materials::MaterialMixtureDevice material_mixture;
  if (use_materials) {
    material_mixture = pmy_pack->pmhd->pmaterials->DeviceData();
  }

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  Kokkos::parallel_reduce("mhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // call c2p function
    // (inline function in ideal_c2p_mhd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    if (!use_dual) {
      SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used);
    } else {
      const Real b2 = SQR(u.bx) + SQR(u.by) + SQR(u.bz);
      const Real dfloor = fmax(eos.dfloor, b2/eos.sigma_max);
      if (u.d < dfloor) {
        u.d = dfloor;
        dfloor_used = true;
      }
      w.d = u.d;
      const Real di = 1.0/u.d;
      w.vx = di*u.mx;
      w.vy = di*u.my;
      w.vz = di*u.mz;
      const Real e_k = 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
      const Real e_m = 0.5*b2;
      const Real eint_cons = u.e - e_k - e_m;
      const Real eint_aux = fmax(cons(m, iion, k, j, i), 0.0) +
                            fmax(cons(m, iele, k, j, i), 0.0);
      const Real eint_floor = MHDInternalEnergyFloor(eos, w.d);
      const bool use_cons_e =
          (eint_cons > 0.0) &&
          ((dual_eta1 <= 0.0) ||
           (eint_cons > dual_eta1*fmax(u.e, 1.0e-18)));
      w.e = use_cons_e ? eint_cons : eint_aux;
      if (w.e < eint_floor) {
        w.e = eint_floor;
        efloor_used = true;
      }
      // Keep conservative total energy unless a physical floor was required.  The
      // auxiliary energy supplies pressure without sacrificing total-energy conservation.
      u.e = w.e + e_k + e_m;
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || tfloor_used) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      // update counter, reset conserved if floor was hit
      if (dfloor_used) {
        cons(m,IDN,k,j,i) = u.d;
        sumd++;
      }
      if (efloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sume++;
      }
      if (tfloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sumt++;
      }
      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;
      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        bool is_material_fraction = false;
        if (use_materials) {
          for (int q=0; q<material_mixture.nmaterials; ++q) {
            if (n == material_mixture.scalar_indices(q)) is_material_fraction = true;
          }
        }
        if (is_material_fraction) {
          // Composition scalars are conservative rho*Y_s values.
          cons(m,n,k,j,i) = fmin(fmax(cons(m,n,k,j,i), 0.0), u.d);
        } else {
          // Legacy positivity floor for every other advected scalar.
          if (cons(m,n,k,j,i) < 0.0) {
            cons(m,n,k,j,i) = 0.0;
          }
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
      // All material densities are explicit. Normalize them to the bulk density after
      // clamping; an all-zero state deterministically becomes the final material.
      if (use_materials) {
        Real material_density_sum = 0.0;
        for (int q=0; q<material_mixture.nmaterials; ++q) {
          material_density_sum += cons(
              m, material_mixture.scalar_indices(q), k, j, i);
        }
        if (material_density_sum > 0.0) {
          const Real scale = u.d/material_density_sum;
          for (int q=0; q<material_mixture.nmaterials; ++q) {
            const int n = material_mixture.scalar_indices(q);
            cons(m,n,k,j,i) *= scale;
            prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
          }
        } else {
          const int n = material_mixture.scalar_indices(
              material_mixture.nmaterials-1);
          cons(m,n,k,j,i) = u.d;
          prim(m,n,k,j,i) = 1.0;
        }
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void PrimToCons()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.  Does not change cell- or face-centered magnetic fields.

void IdealMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons, const int il, const int iu,
                          const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  const bool use_materials = pmy_pack->pmhd->pmaterials != nullptr;
  materials::MaterialMixtureDevice material_mixture;
  if (use_materials) {
    material_mixture = pmy_pack->pmhd->pmaterials->DeviceData();
  }

  par_for("mhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // load single state primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealMHD(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any), always stored at end of cons and prim arrays.
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
    if (use_materials) {
      const materials::MaterialComposition composition =
          material_mixture.CompositionFromPrimitive(prim, m, k, j, i);
      for (int q=0; q<material_mixture.nmaterials; ++q) {
        cons(m,material_mixture.scalar_indices(q),k,j,i) =
            u.d*composition[q];
      }
    }
  });

  return;
}
