//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.cpp
//! \brief create multigrid solver for gravity

// C headers

// C++ headers
#include <algorithm>
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"
#include "../driver/driver.hpp"

class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::MGGravityDriver(Mesh *pm, ParameterInput *pin)
//! \brief MGGravityDriver constructor

MGGravityDriver::MGGravityDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
    four_pi_G_ = pin->GetOrAddReal("gravity", "four_pi_G", -1.0);
    omega_ = pin->GetOrAddReal("gravity", "omega", 1.15);
    eps_ = pin->GetOrAddReal("gravity", "threshold", -1.0);
    niter_ = pin->GetOrAddInteger("gravity", "niteration", -1);
    npresmooth_ = pin->GetOrAddReal("gravity", "npresmooth", npresmooth_);
    npostsmooth_ = pin->GetOrAddReal("gravity", "npostsmooth", npostsmooth_);
    full_multigrid_ = pin->GetOrAddBoolean("gravity", "full_multigrid", false);
    fmg_ncycle_ = pin->GetOrAddInteger("gravity", "fmg_ncycle", 1);
    fshowdef_ = pin->GetOrAddBoolean("gravity", "show_defect", false);
    fsubtract_average_ = pin->GetOrAddBoolean("gravity", "subtract_average", true);
    if (eps_ < 0.0 && niter_ < 0) {
        std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <gravity> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
        exit(EXIT_FAILURE);
  }
  if (four_pi_G_ < 0.0) {
    std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
    exit(EXIT_FAILURE);
  }
  // Allocate the root multigrid
  int nghost = pin->GetOrAddInteger("gravity", "mg_nghost", 1);
  bool root_on_host = pin->GetOrAddBoolean("gravity", "root_on_host", false);
  mgroot_ = new MGGravity(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGGravity(this, pmbp, nghost);
  // allocate boundary buffers
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers((nvar_));
  mglevels_->pbval->RemapIndicesForMG();
}


//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::~MGGravityDriver()
//! \brief MGGravityDriver destructor

MGGravityDriver::~MGGravityDriver() {
  delete mgroot_;
  delete mglevels_;
}

void MGGravityDriver::SetFourPiG(Real four_pi_G) {
  four_pi_G_ = four_pi_G;
}

//----------------------------------------------------------------------------------------
//! \fn MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb)
//! \brief MGGravity constructor

MGGravity::MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}


//----------------------------------------------------------------------------------------
//! \fn MGGravity::~MGGravity()
//! \brief MGGravity deconstructor

MGGravity::~MGGravity() {
  //delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::Solve(int stage, Real dt)
//! \brief load the data and solve

void MGGravityDriver::Solve(Driver *pdriver, int stage, Real dt) {
  RegionIndcs &indcs_ = pmy_pack_->pmesh->mb_indcs;

  // Reallocate MG arrays and phi if AMR has changed the mesh
  PrepareForAMR();
  {
    int nmb = pmy_pack_->nmb_thispack;
    if (static_cast<int>(pmy_pack_->pgrav->phi.extent_int(0)) != nmb) {
      int ncells1 = indcs_.nx1 + 2*indcs_.ng;
      int ncells2 = (indcs_.nx2 > 1) ? (indcs_.nx2 + 2*indcs_.ng) : 1;
      int ncells3 = (indcs_.nx3 > 1) ? (indcs_.nx3 + 2*indcs_.ng) : 1;
      Kokkos::realloc(pmy_pack_->pgrav->phi, nmb, 1, ncells3, ncells2, ncells1);
    }
  }

  // mglevels_ points to the Multigrid object for all MeshBlocks
  // The MG smoother solves -∇²u = src (note the minus sign from the Laplacian
  // convention: Laplacian(u) = 6u - neighbors = -dx²∇²u).  To obtain the
  // standard Poisson equation ∇²φ = 4πGρ we must load the source with a
  // negative sign so that -∇²φ = -4πGρ, i.e. ∇²φ = +4πGρ.
  mglevels_->LoadSource(pmy_pack_->phydro->u0, IDN, indcs_.ng, -four_pi_G_);

  // iterative mode - load initial guess
  if(!full_multigrid_) 
    mglevels_->LoadFinestData(pmy_pack_->pgrav->phi, 0, indcs_.ng);
  
  SetupMultigrid(dt, false);

  if (full_multigrid_)
    SolveFMG(pdriver);
  else
    SolveMG(pdriver);

  if (fshowdef_) {
    Real norm = CalculateDefectNorm(MGNormType::l2, 0);
    if (global_variable::my_rank == 0) {
      std::cout << "MGGravityDriver::Solve: Final defect norm = " << norm << std::endl;
    }
  }

  mglevels_->RetrieveResult(pmy_pack_->pgrav->phi, 0, indcs_.ng);

  return;
}

void MGGravity::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{static_cast<MGGravityDriver*>(pmy_driver_)->omega_/6.0};
  if (on_host_) {
    Smooth(u_[current_level_].h_view, src_[current_level_].h_view,
           coeff_[current_level_].h_view, matrix_[current_level_].h_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  } else {
    Smooth(u_[current_level_].d_view, src_[current_level_].d_view,
           coeff_[current_level_].d_view, matrix_[current_level_].d_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  }
}

void MGGravity::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateDefect(def_[current_level_].h_view, u_[current_level_].h_view,
                    src_[current_level_].h_view, coeff_[current_level_].h_view,
                    matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateDefect(def_[current_level_].d_view, u_[current_level_].d_view,
                    src_[current_level_].d_view, coeff_[current_level_].d_view,
                    matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}

void MGGravity::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateFASRHS(src_[current_level_].h_view, u_[current_level_].h_view,
                    coeff_[current_level_].h_view, matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateFASRHS(src_[current_level_].d_view, u_[current_level_].d_view,
                    coeff_[current_level_].d_view, matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}


//----------------------------------------------------------------------------------------
// Host-side octet physics for MGGravityDriver

static inline Real OctLaplacian(const MGOctet &o, int v, int k, int j, int i) {
  return (6.0*o.U(v,k,j,i) - o.U(v,k+1,j,i) - o.U(v,k,j+1,i)
          - o.U(v,k,j,i+1) - o.U(v,k-1,j,i) - o.U(v,k,j-1,i)
          - o.U(v,k,j,i-1));
}

void MGGravityDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  Real isix = omega_ / 6.0;
  int c = color ^ coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
        Real lap = OctLaplacian(oct, 0, k, j, i);
        oct.U(0,k,j,i) -= (lap - oct.Src(0,k,j,i)*dx2)*isix;
      }
    }
  }
}

void MGGravityDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Def(0,k,j,i) = oct.Src(0,k,j,i) - OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}

void MGGravityDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Src(0,k,j,i) += OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}
