//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_hyd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic hydro

#include "athena.hpp"
#include "hydro/hydro.hpp"
#include "materials/material_mixture.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "two_temperature/two_temperature.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealHydro::IdealHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables. Operates over range of cells given
//! in argument list. Number of times floors used stored into event counters.

void IdealHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  const bool use_dual = pmy_pack->phydro->use_dual_energy;
  const int iion = use_dual ? pmy_pack->phydro->ptwo_temp->iion : -1;
  const int iele = use_dual ? pmy_pack->phydro->ptwo_temp->iele : -1;
  const Real dual_eta1 = pmy_pack->phydro->dual_energy_eta1;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->phydro->fofc;
  const bool use_materials = pmy_pack->phydro->pmaterials != nullptr;
  materials::MaterialMixtureDevice material_mixture;
  if (use_materials) {
    material_mixture = pmy_pack->phydro->pmaterials->DeviceData();
  }

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  Kokkos::parallel_reduce("hyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // call c2p function
    // (inline function in ideal_c2p_hyd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    if (!use_dual) {
      SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);
    } else {
      // Dual-energy variant: select between the conservative internal energy and the
      // advected ion+electron sum, then restore total energy from the selection.
      if (u.d < eos.dfloor) {
        u.d = eos.dfloor;
        dfloor_used = true;
      }
      w.d = u.d;
      const Real di = 1.0/u.d;
      w.vx = di*u.mx;
      w.vy = di*u.my;
      w.vz = di*u.mz;
      const Real e_k = 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
      const Real eint_cons = u.e - e_k;
      const Real eint_aux = fmax(cons(m, iion, k, j, i), 0.0) +
                            fmax(cons(m, iele, k, j, i), 0.0);
      const Real eint_floor = eos.HydroInternalEnergyDensityFloor(w.d);
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
      u.e = w.e + e_k;
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
      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        if (use_materials && n == material_mixture.scalar_index) {
          // The material scalar is conservative rho*Y0. Clamp only its mass fraction;
          // the complementary material remains exactly 1-Y0.
          cons(m,n,k,j,i) = fmin(fmax(cons(m,n,k,j,i), 0.0), u.d);
        } else {
          // Legacy positivity floor for every other advected scalar.
          if (cons(m,n,k,j,i) < 0.0) {
            cons(m,n,k,j,i) = 0.0;
          }
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
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
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables. Operates over range of cells given
//! in argument list.  Floors never needed.

void IdealHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  const bool use_materials = pmy_pack->phydro->pmaterials != nullptr;
  materials::MaterialMixtureDevice material_mixture;
  if (use_materials) {
    material_mixture = pmy_pack->phydro->pmaterials->DeviceData();
  }

  par_for("hyd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // load single state primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealHyd(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      if (use_materials && n == material_mixture.scalar_index) {
        cons(m,n,k,j,i) = u.d*material_mixture.ClampMassFraction(
            prim(m,n,k,j,i));
      } else {
        cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
      }
    }
  });

  return;
}
