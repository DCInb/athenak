//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of Hydro class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "materials/material_mixture.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "two_temperature/two_temperature.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    u1("cons1",1,1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    dual_vf("dual_vf",1,1,1,1,1),
    dual_etot_max("dual_etot_max",1,1,1,1),
    fofc("fofc",1,1,1,1),
    utest("utest",1,1,1,1,1),
    pmy_pack(ppack) {
  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("hydro","eos");
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRHydro(ppack, pin);
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRHydro(ppack, pin);
    } else {
      peos = new IdealHydro(ppack, pin);
    }
    nhydro = 5;
  // density-temperature table EOS
  } else if (eqn_of_state == "table" || eqn_of_state == "tabulated") {
    peos = new TabulatedHydro(ppack, pin);
    nhydro = 5;
  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "<hydro>/eos = isothermal cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalHydro(ppack, pin);
      nhydro = 4;
    }
  // EOS string not recognized
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro>/eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, two-temperature model, diffusion, and source terms
  nuser_scalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
  nscalars = nuser_scalars;
  if (pin->DoesBlockExist("materials")) {
    if (!peos->eos_data.is_gamma_law ||
        pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic ||
        pmy_pack->pcoord->is_dynamical_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<materials> currently requires Newtonian ideal-gas "
                << "hydrodynamics" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pmaterials = new materials::MaterialMixture(
        pin, "hydro", nhydro, nuser_scalars, peos->eos_data.gamma, ppack->punit);
    use_tabular_material_eos = pmaterials->UsesTabularEOS();
  }
  bool use_two_temperature =
      pin->GetOrAddBoolean("hydro", "two_temperature", false);
  if (use_tabular_material_eos && !use_two_temperature) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tabular <materials> EOS requires "
              << "<hydro>/two_temperature=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (use_two_temperature) {
    if (!peos->eos_data.is_gamma_law || pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic ||
        pmy_pack->pcoord->is_dynamical_relativistic ||
        pin->DoesBlockExist("ion-neutral") || pin->DoesBlockExist("radiation")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/two_temperature currently requires standalone "
                << "Newtonian ideal-gas hydrodynamics" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ptwo_temp = new two_temperature::TwoTemperature(
        "hydro", ppack, pin, nhydro + nuser_scalars, pmaterials);
    nscalars += 2 + ptwo_temp->NumberOfRadiationGroups();
    // 2T scalar advection through the shear boundary is validated, but the FLD
    // diffusive-flux interplay with the shearing-box remap is not — forbid the
    // radiation+shear combination until it is tested.
    if (ptwo_temp->pradiation != nullptr && pin->DoesBlockExist("shearing_box")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<thermal_radiation> with a <shearing_box> block "
                << "is not yet validated" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else if (pin->DoesBlockExist("thermal_radiation") &&
             pin->GetOrAddBoolean("thermal_radiation", "enabled", true)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<thermal_radiation> requires "
              << "<hydro>/two_temperature=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Hydrodynamics has no magnetic energy to subtract, so total-minus-kinetic is usually
  // a well-conditioned internal energy and the ordinary 2T pressure partition suffices.
  // Laser-driven coronae do reach kinetic-energy-dominated states where the subtraction
  // loses the gas pressure entirely; those decks opt in explicitly.  Unlike MHD this
  // therefore defaults off, so existing 2T hydro results are unchanged.
  use_dual_energy = pin->GetOrAddBoolean("hydro", "dual_energy", false);
  if (use_dual_energy) {
    if (ptwo_temp == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy requires "
                << "<hydro>/two_temperature=true" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    dual_energy_eta1 = pin->GetOrAddReal("hydro", "dual_energy_eta1", 1.0e-3);
    dual_energy_eta2 = pin->GetOrAddReal("hydro", "dual_energy_eta2", 1.0e-4);
    if (dual_energy_eta1 < 0.0 || dual_energy_eta2 < 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy_eta1 and dual_energy_eta2 must be "
                << "non-negative" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // The Biermann battery is a magnetic-field source term, so it exists only in <mhd>.
  // Decks that switch a 3T problem between the two carriers keep the parameters in
  // place; say plainly that they do nothing here rather than dropping them silently.
  if (pin->GetOrAddBoolean("hydro", "biermann_battery", false) &&
      global_variable::my_rank == 0) {
    std::cout << "### WARNING: <hydro>/biermann_battery is ignored; the Biermann "
              << "battery generates magnetic field and requires an <mhd> block. "
              << "This run evolves no magnetic field." << std::endl;
  }

  // Viscosity (if requested in input file)
  if (pin->DoesParameterExist("hydro","nu_iso") ||
      pin->DoesParameterExist("hydro","nu_aniso")) {
    pvisc = new Viscosity("hydro", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Thermal conduction (if requested in input file)
  if (pin->DoesParameterExist("hydro","alpha_iso") ||
      pin->DoesParameterExist("hydro","alpha_aniso") ||
      pin->DoesParameterExist("hydro","alpha_spitzer")) {
    if (peos->eos_data.is_gamma_law) {
      pcond = new Conduction("hydro", ppack, pin, ptwo_temp, pmaterials);
    } else {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Thermal conduction in hydro requires ideal gas EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    pcond = nullptr;
  }

  // Source terms (if needed)
  if (pin->DoesBlockExist("hydro_srcterms")) {
    if (peos->eos_data.is_table &&
        (pin->GetOrAddBoolean("hydro_srcterms", "ism_cooling", false) ||
         pin->GetOrAddBoolean("hydro_srcterms", "rel_cooling", false))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Tabulated EOS does not support gamma-law cooling "
                << "source terms" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    psrc = new SourceTerms("hydro_srcterms", ppack, pin);
  }

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");
  if (use_dual_energy) {
    if (evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy requires dynamic evolution"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pvisc != nullptr || (pcond != nullptr && !pcond->IsImplicit()) ||
        psrc != nullptr || pin->DoesBlockExist("shearing_box")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy is not yet compatible with "
                << "viscosity, explicit thermal conduction, hydro source terms, or "
                << "shearing box" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // allocate memory for conserved and primitive variables
  // With AMR, maximum size of Views are limited by total device memory through an input
  // parameter, which in turn limits max number of MBs that can be created.
  {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(w0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    if (use_dual_energy) {
      Kokkos::realloc(dual_etot_max, nmb, ncells3, ncells2, ncells1);
    }
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nhydro+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb, (nhydro+nscalars), n_ccells3, n_ccells2, n_ccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables.  The dual-energy
  // correction needs one extra refluxed face field (the upwind face velocity).
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers((nhydro+nscalars),
      (nhydro+nscalars) + (use_dual_energy ? 1 : 0));

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, (nhydro+nscalars));
    psbox_u = new ShearingBoxCC(ppack, pin, (nhydro+nscalars));
  } else {
    porb_u = nullptr;
    psbox_u = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // determine if FOFC is enabled
    use_fofc = pin->GetOrAddBoolean("hydro","fofc",false);
    if (use_dual_energy && use_fofc) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy is not yet compatible with FOFC"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (use_fofc && ptwo_temp != nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/two_temperature is not yet compatible with "
                << "FOFC (first-order flux replacement does not include the "
                << "ion/electron/radiation scalar fluxes)" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
      // check that nghost > 2 with PLM+FOFC
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (use_fofc && indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 3 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // check that nghost > 3 with PPM4(or PPMX or WENOZ)+FOFC
      if (use_fofc && indcs.ng < 4) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 4 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> reconstruct = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("hydro","rsolver");
    if (peos->eos_data.is_table && rsolver != "llf") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos=table currently requires rsolver=llf"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Only the LLF flux has a material-EOS variant that takes pressure and sound speed
    // from the tabulated closure rather than from the gamma-law carrier.
    if (use_tabular_material_eos && rsolver != "llf") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "tabular <materials> EOS currently requires "
                << "<hydro>/rsolver=llf" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Special relativistic dynamic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_sr;
        } else if (rsolver.compare("hllc") == 0) {
          rsolver_method = Hydro_RSolver::hllc_sr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic dynamic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_gr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for GR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for GR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic dynamic solvers
    } else if (evolution_t.compare("dynamic") == 0) {
      // LLF solver
      if (rsolver.compare("llf") == 0) {
        rsolver_method = Hydro_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = Hydro_RSolver::hlle;
      // HLLC solver
      } else if (rsolver.compare("hllc") == 0) {
        if (peos->eos_data.is_ideal) {
          rsolver_method = Hydro_RSolver::hllc;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = hllc cannot be used with "
                    << "isothermal EOS" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      // Roe solver
      } else if (rsolver.compare("roe") == 0) {
        rsolver_method = Hydro_RSolver::roe;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solvers
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        rsolver_method = Hydro_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // Final memory allocations
    {
      // allocate second registers, fluxes
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(u1,       nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x1f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x2f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x3f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      if (use_dual_energy) {
        Kokkos::realloc(dual_vf.x1f, nmb, 1, ncells3, ncells2, ncells1);
        Kokkos::realloc(dual_vf.x2f, nmb, 1, ncells3, ncells2, ncells1);
        Kokkos::realloc(dual_vf.x3f, nmb, 1, ncells3, ncells2, ncells1);
      }

      // allocate array of flags used with FOFC
      if (use_fofc) {
        Kokkos::realloc(fofc,  nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest, nmb, nhydro, ncells3, ncells2, ncells1);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

Hydro::~Hydro() {
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_u;
  if (psrc != nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (pvisc != nullptr) {delete pvisc;}
  if (ptwo_temp != nullptr) {delete ptwo_temp;}
  if (pmaterials != nullptr) {delete pmaterials;}
  delete peos;
}

} // namespace hydro
