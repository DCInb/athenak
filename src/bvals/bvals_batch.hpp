#ifndef BVALS_BVALS_BATCH_HPP_
#define BVALS_BVALS_BATCH_HPP_
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
// Aggregate uniform-mesh GPU halo buffers into one MPI message per peer.

#include <vector>

#include "athena.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>

class BoundaryMPIBatch {
 public:
  struct Segment {
    Real *data;
    int count, peer, tag;
    bool operator==(const Segment &b) const {
      return data == b.data && count == b.count && peer == b.peer && tag == b.tag;
    }
  };
  struct Span { Real *data; int count, offset; };
  struct Peer { int rank, count, offset; };
  struct Plan {
    std::vector<Segment> segments;
    std::vector<Peer> peers;
    std::vector<MPI_Request> requests;
    DvceArray1D<Span> spans;
    DvceArray1D<Real> buffer;
  };

  void PostRecv(std::vector<Segment> segments, MPI_Comm comm);
  void PackAndSend(std::vector<Segment> segments, MPI_Comm comm);
  bool Receive(bool wait);
  void WaitSend();

 private:
  static void SetPlan(Plan &plan, std::vector<Segment> segments);
  Plan send_, recv_;
  bool unpacked_ = true;
};
#endif
#endif  // BVALS_BVALS_BATCH_HPP_
