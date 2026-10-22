#ifndef LASER_LASER_BLOCK_TREE_HPP_
#define LASER_LASER_BLOCK_TREE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_block_tree.hpp
//! \brief Cached leaf-block bounds and a stackless spatial lookup for device rays.

#include <algorithm>
#include <numeric>
#include <vector>

#include "athena.hpp"

namespace laser {

struct LaserBlockInfo {
  Real x1min, x1max, x2min, x2max, x3min, x3max;
  Real dx1, dx2, dx3;
  int gid, rank;
};

struct LaserBlockTreeNode {
  Real lower[3], upper[3];
  int gid = -1;
  int skip = 0;  // first node after this subtree in preorder
};

// Partition leaf centers along the longest active axis. Bounding boxes retain the
// mesh's exact endpoints, including its half-open ownership convention at interfaces.
inline std::vector<LaserBlockTreeNode> BuildBlockTree(
    const std::vector<LaserBlockInfo> &blocks, bool multi_d, bool three_d) {
  std::vector<LaserBlockTreeNode> tree;
  if (blocks.empty()) return tree;
  tree.reserve(2*blocks.size()-1);
  std::vector<int> order(blocks.size());
  std::iota(order.begin(), order.end(), 0);
  const int dimensions = three_d ? 3 : (multi_d ? 2 : 1);
  const auto build = [&](auto &&self, int begin, int end) -> void {
    const int index = static_cast<int>(tree.size());
    const auto &first = blocks[order[begin]];
    LaserBlockTreeNode node{{first.x1min, first.x2min, first.x3min},
                            {first.x1max, first.x2max, first.x3max}};
    for (int p = begin+1; p < end; ++p) {
      const auto &block = blocks[order[p]];
      const Real lower[3] = {block.x1min, block.x2min, block.x3min};
      const Real upper[3] = {block.x1max, block.x2max, block.x3max};
      for (int d = 0; d < dimensions; ++d) {
        node.lower[d] = std::min(node.lower[d], lower[d]);
        node.upper[d] = std::max(node.upper[d], upper[d]);
      }
    }
    tree.push_back(node);
    if (end-begin == 1) {
      tree[index].gid = first.gid;
    } else {
      int axis = 0;
      for (int d = 1; d < dimensions; ++d) {
        if (node.upper[d]-node.lower[d] > node.upper[axis]-node.lower[axis]) axis = d;
      }
      const auto center = [axis](const LaserBlockInfo &block) {
        if (axis == 0) return 0.5*block.x1min+0.5*block.x1max;
        if (axis == 1) return 0.5*block.x2min+0.5*block.x2max;
        return 0.5*block.x3min+0.5*block.x3max;
      };
      const int middle = begin+(end-begin)/2;
      std::nth_element(order.begin()+begin, order.begin()+middle, order.begin()+end,
          [&](int a, int b) {
            const Real ca = center(blocks[a]), cb = center(blocks[b]);
            return ca == cb ? blocks[a].gid < blocks[b].gid : ca < cb;
          });
      self(self, begin, middle);
      self(self, middle, end);
    }
    tree[index].skip = static_cast<int>(tree.size());
  };
  build(build, 0, static_cast<int>(blocks.size()));
  return tree;
}

KOKKOS_INLINE_FUNCTION
int FindGlobalBlock(const DvceArray1D<LaserBlockTreeNode> &tree,
                    Real x, Real y, Real z, bool multi_d, bool three_d) {
  int index = 0;
  while (index < tree.extent_int(0)) {
    const auto &node = tree(index);
    const bool inside = x >= node.lower[0] && x < node.upper[0] &&
        (!multi_d || (y >= node.lower[1] && y < node.upper[1])) &&
        (!three_d || (z >= node.lower[2] && z < node.upper[2]));
    if (!inside) {
      index = node.skip;
    } else if (node.gid >= 0) {
      return node.gid;
    } else {
      ++index;
    }
  }
  return -1;
}

KOKKOS_INLINE_FUNCTION
int FindLocalBlock(const DvceArray1D<LaserBlockTreeNode> &tree,
                   int first_gid, int nmb, Real x, Real y, Real z,
                   bool multi_d, bool three_d) {
  const int gid = FindGlobalBlock(tree, x, y, z, multi_d, three_d);
  return gid >= first_gid && gid < first_gid+nmb ? gid-first_gid : -1;
}

} // namespace laser

#endif // LASER_LASER_BLOCK_TREE_HPP_
