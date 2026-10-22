//========================================================================================
//! \file laser_block_tree_driver.cpp
//! \brief Compare device block lookup with exhaustive half-open leaf ownership.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "athena.hpp"
#include "laser/laser_block_tree.hpp"

struct Query {
  Real x, y, z;
  int expected;
};

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    std::mt19937 random(42);
    for (int dimensions = 1; dimensions <= 3; ++dimensions) {
      const bool multi_d = dimensions >= 2, three_d = dimensions == 3;
      const int nx = dimensions == 1 ? 1024 : 8;
      const int ny = multi_d ? 8 : 1, nz = three_d ? 8 : 1;
      std::vector<laser::LaserBlockInfo> blocks;
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            // Nonuniform leaves, with both coarse/fine interfaces and corners.
            const int split = (i+2*j+3*k)%3 == 0 ? 2 : 1;
            for (int ck = 0; ck < (three_d ? split : 1); ++ck) {
              for (int cj = 0; cj < (multi_d ? split : 1); ++cj) {
                for (int ci = 0; ci < split; ++ci) {
                  blocks.push_back({
                      (i+Real(ci)/split)/nx, (i+Real(ci+1)/split)/nx,
                      multi_d ? (j+Real(cj)/split)/ny : 0.0,
                      multi_d ? (j+Real(cj+1)/split)/ny : 1.0,
                      three_d ? (k+Real(ck)/split)/nz : 0.0,
                      three_d ? (k+Real(ck+1)/split)/nz : 1.0,
                      0.0, 0.0, 0.0, 0, 0});
                }
              }
            }
          }
        }
      }
      std::shuffle(blocks.begin(), blocks.end(), random);
      for (std::size_t n = 0; n < blocks.size(); ++n) blocks[n].gid = n;
      const auto host_nodes = laser::BuildBlockTree(blocks, multi_d, three_d);
      DvceArray1D<laser::LaserBlockTreeNode> nodes("lookup-nodes", host_nodes.size());
      auto mirror = Kokkos::create_mirror_view(nodes);
      for (std::size_t n = 0; n < host_nodes.size(); ++n) mirror(n) = host_nodes[n];
      Kokkos::deep_copy(nodes, mirror);

      std::vector<Query> queries;
      for (const auto &block : blocks) {
        queries.push_back({block.x1min, block.x2min, block.x3min, -1});
        queries.push_back({block.x1max, block.x2max, block.x3max, -1});
        queries.push_back({std::nextafter(block.x1max, block.x1min),
                           std::nextafter(block.x2max, block.x2min),
                           std::nextafter(block.x3max, block.x3min), -1});
      }
      std::uniform_real_distribution<Real> coordinate(-0.05, 1.05);
      for (int n = 0; n < 10000; ++n) {
        queries.push_back({coordinate(random), coordinate(random), coordinate(random), -1});
      }
      for (auto &query : queries) {
        for (const auto &block : blocks) {
          if (query.x >= block.x1min && query.x < block.x1max &&
              (!multi_d || (query.y >= block.x2min && query.y < block.x2max)) &&
              (!three_d || (query.z >= block.x3min && query.z < block.x3max))) {
            query.expected = block.gid;
            break;
          }
        }
      }
      DvceArray1D<Query> device_queries("lookup-queries", queries.size());
      auto host_queries = Kokkos::create_mirror_view(device_queries);
      for (std::size_t n = 0; n < queries.size(); ++n) host_queries(n) = queries[n];
      Kokkos::deep_copy(device_queries, host_queries);
      DvceArray1D<laser::LaserBlockInfo> device_blocks("lookup-blocks", blocks.size());
      auto host_blocks = Kokkos::create_mirror_view(device_blocks);
      for (std::size_t n = 0; n < blocks.size(); ++n) host_blocks(n) = blocks[n];
      Kokkos::deep_copy(device_blocks, host_blocks);
      const int first_gid = blocks.size()/3, count = blocks.size()/2;
      int mismatches = 0;
      Kokkos::fence();
      Kokkos::Timer timer;
      Kokkos::parallel_reduce("block-lookup-linear-reference",
          Kokkos::RangePolicy<>(0, queries.size()),
          KOKKOS_LAMBDA(const int n, int &wrong) {
            const auto query = device_queries(n);
            int found = -1;
            for (int b = 0; b < device_blocks.extent_int(0); ++b) {
              const auto block = device_blocks(b);
              if (query.x >= block.x1min && query.x < block.x1max &&
                  (!multi_d || (query.y >= block.x2min && query.y < block.x2max)) &&
                  (!three_d || (query.z >= block.x3min && query.z < block.x3max))) {
                found = block.gid;
                break;
              }
            }
            if (found != query.expected) ++wrong;
          }, mismatches);
      Kokkos::fence();
      const double linear_time = timer.seconds();
      failures += mismatches;
      timer.reset();
      Kokkos::parallel_reduce("block-lookup-regression",
          Kokkos::RangePolicy<>(0, queries.size()),
          KOKKOS_LAMBDA(const int n, int &wrong) {
            const auto query = device_queries(n);
            if (laser::FindGlobalBlock(nodes, query.x, query.y, query.z,
                                      multi_d, three_d) != query.expected) ++wrong;
          }, mismatches);
      Kokkos::fence();
      const double tree_time = timer.seconds();
      failures += mismatches;
      Kokkos::parallel_reduce("block-lookup-local-regression",
          Kokkos::RangePolicy<>(0, queries.size()),
          KOKKOS_LAMBDA(const int n, int &wrong) {
            const auto query = device_queries(n);
            const int local = query.expected >= first_gid &&
                query.expected < first_gid+count ? query.expected-first_gid : -1;
            if (laser::FindLocalBlock(nodes, first_gid, count, query.x, query.y,
                                     query.z, multi_d, three_d) != local) ++wrong;
          }, mismatches);
      failures += mismatches;
      std::cout << dimensions << "D: " << blocks.size() << " leaves, " << queries.size()
                << " queries, " << mismatches << " local mismatches; linear="
                << linear_time << "s tree=" << tree_time << "s\n";
    }
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
