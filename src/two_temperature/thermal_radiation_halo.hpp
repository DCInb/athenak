#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_HALO_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_HALO_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! Scalar halo/flux plans shared by radiation and electron conduction.

#include "athena.hpp"

class MeshBlockPack;
class MeshBoundaryValuesCC;

namespace two_temperature {

class ImplicitRadiationHalo {
 public:
  ImplicitRadiationHalo();
  ~ImplicitRadiationHalo();
  ImplicitRadiationHalo(const ImplicitRadiationHalo &) = delete;
  ImplicitRadiationHalo &operator=(const ImplicitRadiationHalo &) = delete;

  // Called once per transport solve, never once per Krylov iteration. Compares
  // actual connectivity/ownership, so same-size regrids invalidate the plans too.
  void Prepare(MeshBlockPack *pack, MeshBoundaryValuesCC *bvals, bool host_staging);
  // width=0 retains the fluid halo. width=1 exchanges the two-point stencil;
  // width=2 keeps two same-level fine layers to restrict one coarse donor layer.
  // Both compact widths retain one coarse/fine donor layer. Faces-only excludes
  // transverse padding; quadratic radiation must include edges and corners.
  void Exchange(DvceArray5D<Real> &field, DvceArray5D<Real> &coarse, bool faces_only,
                int width = 0);
  // Area-average fine fluxes onto coarse faces, with one message per peer.
  void Reflux(DvceFaceFld5D<Real> &flux, FaceFldOrigin origin = {});
  static KOKKOS_INLINE_FUNCTION bool IsFace(int slot) {
    return slot < 16 || (slot >= 24 && slot < 32);
  }

  // Apply the same receiver-space clipping when planning transfers and when
  // injecting coarse donors. Template bounds also accept device buffer indices.
  template<typename Bounds, typename Indices>
  static KOKKOS_INLINE_FUNCTION Bounds CompactBounds(
      Bounds b, const Indices &ind, bool coarse, bool faces_only, int width) {
    ClipAxis(b.bis, b.bie, coarse ? ind.cis : ind.is,
             coarse ? ind.cie : ind.ie, faces_only, width);
    ClipAxis(b.bjs, b.bje, coarse ? ind.cjs : ind.js,
             coarse ? ind.cje : ind.je, faces_only, width);
    ClipAxis(b.bks, b.bke, coarse ? ind.cks : ind.ks,
             coarse ? ind.cke : ind.ke, faces_only, width);
    return b;
  }

 private:
  static KOKKOS_INLINE_FUNCTION void ClipAxis(
      int &lo, int &hi, int first, int last, bool faces_only, int width) {
    const bool normal = hi < first || lo > last;
    const int depth = faces_only && !normal ? 0 : width;
    lo = lo < first-depth ? first-depth : lo;
    hi = hi > last+depth ? last+depth : hi;
  }
  struct State;
  State *state_;
};

}  // namespace two_temperature
#endif  // TWO_TEMPERATURE_THERMAL_RADIATION_HALO_HPP_
