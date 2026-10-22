#ifndef DIFFUSION_NATIVE_DIFFUSION_SOLVER_HPP_
#define DIFFUSION_NATIVE_DIFFUSION_SOLVER_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <cstddef>
#include <memory>

#include "athena.hpp"

class MeshBlockPack;
namespace diffusion {

// Shared leaf-cell adapter for radiation (mass=1) and conduction (mass=C_v).
// Uses the existing frozen two-point face graph on AMR; coarse rows request
// individual fine-face conductances so both sides use the identical flux.
class NativeDiffusionSolver {
 public:
  struct State;
  NativeDiffusionSolver();
  ~NativeDiffusionSolver();
  NativeDiffusionSolver(const NativeDiffusionSolver &) = delete;
  NativeDiffusionSolver &operator=(const NativeDiffusionSolver &) = delete;
  // Boundary types: 0=Neumann, 1=Dirichlet, 2=vacuum. Empty mass means unity.
  // Uniform grids use arithmetic coefficient faces; AMR uses supplied faces.
  void Build(MeshBlockPack *pack, const DvceArray5D<Real> &coefficient,
             const DvceFaceFld5D<Real> &faces, FaceFldOrigin origin,
             const DvceArray5D<Real> &mass, Real factor, const int boundary[6],
             Real vacuum_cap, bool device_mpi, std::size_t max_bytes);
  void Apply(MeshBlockPack *pack, const DvceArray5D<Real> &residual,
             DvceArray5D<Real> &correction);
  int levels() const;
  std::size_t bytes() const;

 private:
  std::unique_ptr<State> state_;
};

}  // namespace diffusion
#endif  // DIFFUSION_NATIVE_DIFFUSION_SOLVER_HPP_
