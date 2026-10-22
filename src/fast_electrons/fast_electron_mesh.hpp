#ifndef FAST_ELECTRONS_FAST_ELECTRON_MESH_HPP_
#define FAST_ELECTRONS_FAST_ELECTRON_MESH_HPP_
//========================================================================================
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
//! Direction-aware leaf/cell lookup. Coordinates are never nudged across an interface.
//========================================================================================

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "fast_electrons/fast_electrons.hpp"

namespace fast_electrons {

template <typename Packet, typename Tree, typename Blocks>
KOKKOS_INLINE_FUNCTION
void LocatePacket(Packet &p, const Tree &tree, const Blocks &blocks,
                  const RegionSize &domain, const RegionIndcs &ind, int ndim,
                  bool periodic) {
  const Real lower[3] = {domain.x1min, domain.x2min, domain.x3min};
  const Real upper[3] = {domain.x1max, domain.x2max, domain.x3max};
  const int nc[3] = {ind.nx1, ind.nx2, ind.nx3};
  for (int d = 0; d < ndim; ++d) {
    if ((p.x[d] <= lower[d] && p.n[d] < 0.0) ||
        (p.x[d] >= upper[d] && p.n[d] > 0.0)) {
      if (!periodic) {
        p.status = 2;
        return;
      }
      p.x[d] = p.n[d] < 0.0 ? upper[d] : lower[d];
    }
  }
  int node_index = 0;
  p.gid = -1;
  while (node_index < tree.extent_int(0)) {
    const auto node = tree(node_index);
    bool inside = true;
    for (int d = 0; d < ndim; ++d) {
      // At an exact face choose the cell entered by the velocity. Tangential packets
      // use half-open ownership, except at the physical upper boundary.
      const bool at_upper = p.x[d] == node.upper[d] &&
          (p.n[d] < 0.0 || (p.n[d] == 0.0 && p.x[d] == upper[d]));
      inside = inside &&
          (p.x[d] > node.lower[d] ||
           (p.x[d] == node.lower[d] && p.n[d] >= 0.0)) &&
          (p.x[d] < node.upper[d] || at_upper);
    }
    if (!inside) {
      node_index = node.skip;
    } else if (node.gid >= 0) {
      p.gid = node.gid;
      break;
    } else {
      ++node_index;
    }
  }
  if (p.gid < 0) {
    p.status = 3;
    return;
  }
  const auto block = blocks(p.gid);
  const Real lo[3] = {block.x1min, block.x2min, block.x3min};
  const Real hi[3] = {block.x1max, block.x2max, block.x3max};
  const Real dx[3] = {block.dx1, block.dx2, block.dx3};
  for (int d = 0; d < 3; ++d) {
    p.cell[d] = 0;
    if (d >= ndim) continue;
    int c = static_cast<int>(floor((p.x[d]-lo[d])/dx[d]));
    c = c < 0 ? 0 : (c >= nc[d] ? nc[d]-1 : c);
    const Real left = c == 0 ? lo[d] : lo[d]+c*dx[d];
    const Real right = c == nc[d]-1 ? hi[d] : lo[d]+(c+1)*dx[d];
    if (p.n[d] < 0.0 && p.x[d] <= left && c > 0) --c;
    if (p.n[d] >= 0.0 && p.x[d] >= right && c < nc[d]-1) ++c;
    p.cell[d] = c;
  }
}
}  // namespace fast_electrons
#endif  // FAST_ELECTRONS_FAST_ELECTRON_MESH_HPP_
