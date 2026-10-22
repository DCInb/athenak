// Two-rank, one-node transport measurements using the application's MPI/CUDA stack.
// All allocations, verification, barriers and reporting are outside timed loops.
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int rank_id;
void fail(const std::string &message) {
  std::cerr << "benchmark rank " << rank_id << ": " << message << std::endl;
  MPI_Abort(MPI_COMM_WORLD, 2);
  std::abort();
}
void mpi(int code) {
  if (code != MPI_SUCCESS) fail("MPI operation failed");
}
void gpu(cudaError_t code) {
  if (code != cudaSuccess) fail(cudaGetErrorString(code));
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int ranks, devices;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 2 || argc < 2) fail("requires two ranks and host/device argument");
  const std::string memory = argv[1];
  if (memory != "host" && memory != "device") fail("unknown memory type");
  const bool device = memory == "device";
  gpu(cudaGetDeviceCount(&devices));
  if (devices < 2) fail("two visible GPUs required");
  gpu(cudaSetDevice(rank_id));
  gpu(cudaFree(nullptr));
  constexpr int window = 8;
  constexpr std::size_t maximum = 16UL*1024*1024;
  unsigned char *send, *recv;
  if (device) {
    gpu(cudaMalloc(reinterpret_cast<void **>(&send), maximum*window));
    gpu(cudaMalloc(reinterpret_cast<void **>(&recv), maximum*window));
    gpu(cudaMemset(send, rank_id+37, maximum*window));
  } else {
    gpu(cudaMallocHost(reinterpret_cast<void **>(&send), maximum*window));
    gpu(cudaMallocHost(reinterpret_cast<void **>(&recv), maximum*window));
    std::memset(send, rank_id+37, maximum*window);
  }
  gpu(cudaDeviceSynchronize());
  const int peer = 1-rank_id;
  auto check = [&](std::size_t bytes, bool receive = true) {
    if (!receive) return;
    std::vector<unsigned char> values(bytes);
    if (device) gpu(cudaMemcpy(values.data(), recv, bytes, cudaMemcpyDeviceToHost));
    else std::memcpy(values.data(), recv, bytes);
    if (!std::all_of(values.begin(), values.end(),
        [&](unsigned char value) { return value == peer+37; })) fail("payload mismatch");
  };
  auto finish = [&](const char *kind, std::size_t bytes, int iterations, int repetition,
                    double start, int transfers, int batch = 1) {
    // MPI completion already owns the buffers; the final fence also checks that
    // any device work enqueued by the transport has completed.
    if (device) gpu(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime()-start, maximum_elapsed;
    mpi(MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank_id == 0) {
      const double per_iteration = maximum_elapsed/iterations;
      const double factor = std::string(kind) == "pingpong" ? 2.0 : 1.0;
      std::cout << std::setprecision(10) << "{\"kind\":\"" << kind
        << "\",\"memory\":\"" << memory << "\",\"bytes\":" << bytes
        << ",\"iterations\":" << iterations << ",\"window\":" << batch
        << ",\"repetition\":" << repetition << ",\"seconds\":" << maximum_elapsed
        << ",\"time_us\":" << per_iteration*1e6/factor
        << ",\"per_direction_gbps\":" << bytes*batch*factor/per_iteration/1e9
        << ",\"aggregate_gbps\":" << transfers*bytes*batch/per_iteration/1e9
        << "}" << std::endl;
    }
  };
  for (std::size_t bytes : {8UL, 64UL, 256UL, 1024UL, 4096UL, 16384UL,
                           65536UL, 262144UL, 1048576UL, 4194304UL, maximum}) {
    const int count = static_cast<int>(bytes);
    const int iterations = bytes <= 16384 ? 2000 : (bytes <= 1048576 ? 200 : 24);
    auto pingpong = [&]() {
      if (rank_id == 0) {
        mpi(MPI_Send(send, count, MPI_BYTE, peer, 11, MPI_COMM_WORLD));
        mpi(MPI_Recv(recv, count, MPI_BYTE, peer, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      } else {
        mpi(MPI_Recv(recv, count, MPI_BYTE, peer, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        mpi(MPI_Send(send, count, MPI_BYTE, peer, 11, MPI_COMM_WORLD));
      }
    };
    auto exchange = [&]() {
      MPI_Request request[2];
      mpi(MPI_Irecv(recv, count, MPI_BYTE, peer, 12, MPI_COMM_WORLD, &request[0]));
      mpi(MPI_Isend(send, count, MPI_BYTE, peer, 12, MPI_COMM_WORLD, &request[1]));
      mpi(MPI_Waitall(2, request, MPI_STATUSES_IGNORE));
    };
    for (int repetition = 0; repetition < 3; ++repetition) {
      for (int i = 0; i < 10; ++i) pingpong();
      mpi(MPI_Barrier(MPI_COMM_WORLD));
      double start = MPI_Wtime();
      for (int i = 0; i < iterations; ++i) pingpong();
      finish("pingpong", bytes, iterations, repetition, start, 2);
      check(bytes);
      for (int i = 0; i < 10; ++i) exchange();
      mpi(MPI_Barrier(MPI_COMM_WORLD));
      start = MPI_Wtime();
      for (int i = 0; i < iterations; ++i) exchange();
      finish("exchange", bytes, iterations, repetition, start, 2);
      check(bytes);
    }
    if (bytes < 65536) continue;
    const int batches = std::max(16, std::min(128, static_cast<int>(32UL*1024*1024/(bytes*window))));
    auto bandwidth = [&](bool bidirectional) {
      MPI_Request requests[2*window];
      int nr = 0;
      if (bidirectional || rank_id == 1) {
        for (int w = 0; w < window; ++w)
          mpi(MPI_Irecv(recv+w*bytes, count, MPI_BYTE, peer, 13, MPI_COMM_WORLD, &requests[nr++]));
      }
      if (bidirectional || rank_id == 0) {
        for (int w = 0; w < window; ++w)
          mpi(MPI_Isend(send+w*bytes, count, MPI_BYTE, peer, 13, MPI_COMM_WORLD, &requests[nr++]));
      }
      mpi(MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE));
      if (!bidirectional) {
        int ack = 1;
        if (rank_id == 1) mpi(MPI_Send(&ack, 1, MPI_INT, peer, 14, MPI_COMM_WORLD));
        else mpi(MPI_Recv(&ack, 1, MPI_INT, peer, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      }
    };
    for (bool bidirectional : {false, true}) {
      for (int repetition = 0; repetition < 3; ++repetition) {
        for (int i = 0; i < 3; ++i) bandwidth(bidirectional);
        mpi(MPI_Barrier(MPI_COMM_WORLD));
        double start = MPI_Wtime();
        for (int i = 0; i < batches; ++i) bandwidth(bidirectional);
        finish(bidirectional ? "bidir_bw" : "oneway_bw", bytes, batches, repetition,
               start, bidirectional ? 2 : 1, window);
        check(bytes*window, bidirectional || rank_id == 1);
      }
    }
  }
  if (!device) {
    double value = rank_id+1, result = 0;
    for (int repetition = 0; repetition < 3; ++repetition) {
      for (int i = 0; i < 20; ++i) mpi(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
      mpi(MPI_Barrier(MPI_COMM_WORLD));
      double start = MPI_Wtime();
      for (int i = 0; i < 10000; ++i) mpi(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
      finish("allreduce", 8, 10000, repetition, start, 0);
      if (result != 3) fail("allreduce mismatch");
    }
  }
  if (device) { gpu(cudaFree(send)); gpu(cudaFree(recv)); }
  else { gpu(cudaFreeHost(send)); gpu(cudaFreeHost(recv)); }
  if (rank_id == 0) std::cout << "# payload validation passed" << std::endl;
  MPI_Finalize();
}
