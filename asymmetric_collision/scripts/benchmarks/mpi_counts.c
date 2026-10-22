/* Diagnostic-only PMPI counters. Payload sizes are API arguments, not wire bytes.
 * Use separate runs; these counters are never loaded for quiet performance tests.
 */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { ISEND, IRECV, ALLREDUCE, ALLGATHERV, NKINDS };
static const char *names[NKINDS] = {"Isend", "Irecv", "Allreduce", "Allgatherv"};
static struct Counts {
  uint64_t calls, bytes, small, maximum, bins[9];
} counts[NKINDS];
static const uint64_t limits[8] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};

static void record(int kind, int n, MPI_Datatype type) {
  int width = 0;
  if (n < 0 || PMPI_Type_size(type, &width) != MPI_SUCCESS || width < 0) return;
  const uint64_t bytes = (uint64_t)n * (uint64_t)width;
  struct Counts *c = &counts[kind];
  ++c->calls;
  c->bytes += bytes;
  c->small += bytes <= 16384;
  if (bytes > c->maximum) c->maximum = bytes;
  int bin = 0;
  while (bin < 8 && bytes > limits[bin]) ++bin;
  ++c->bins[bin];
}

int MPI_Isend(const void *buf, int n, MPI_Datatype type, int dest, int tag,
              MPI_Comm comm, MPI_Request *request) {
  record(ISEND, n, type);
  return PMPI_Isend(buf, n, type, dest, tag, comm, request);
}
int MPI_Irecv(void *buf, int n, MPI_Datatype type, int src, int tag,
              MPI_Comm comm, MPI_Request *request) {
  record(IRECV, n, type);
  return PMPI_Irecv(buf, n, type, src, tag, comm, request);
}
int MPI_Allreduce(const void *send, void *recv, int n, MPI_Datatype type,
                  MPI_Op op, MPI_Comm comm) {
  record(ALLREDUCE, n, type);
  return PMPI_Allreduce(send, recv, n, type, op, comm);
}
int MPI_Allgatherv(const void *send, int n, MPI_Datatype type, void *recv,
                   const int *sizes, const int *offsets, MPI_Datatype recv_type,
                   MPI_Comm comm) {
  if (send == MPI_IN_PLACE) {
    int rank = 0;
    PMPI_Comm_rank(comm, &rank);
    record(ALLGATHERV, sizes[rank], recv_type);
  } else {
    record(ALLGATHERV, n, type);
  }
  return PMPI_Allgatherv(send, n, type, recv, sizes, offsets, recv_type, comm);
}
int MPI_Finalize(void) {
  const char *directory = getenv("ATHENA_MPI_COUNTS_DIR");
  if (directory != NULL) {
    int rank = 0;
    char path[4096];
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    snprintf(path, sizeof(path), "%s/rank-%d.json", directory, rank);
    FILE *file = fopen(path, "w");
    if (file != NULL) {
      fprintf(file, "{\n  \"rank\": %d,\n  \"scope\": \"whole process\",\n", rank);
      fprintf(file, "  \"bin_upper_bytes\": [64,256,1024,4096,16384,65536,262144,1048576,null],\n");
      fprintf(file, "  \"calls\": {\n");
      for (int k = 0; k < NKINDS; ++k) {
        struct Counts *c = &counts[k];
        fprintf(file, "    \"%s\": {\"count\": %llu, \"payload_bytes\": %llu, "
            "\"at_most_16_kib\": %llu, \"maximum_bytes\": %llu, \"histogram\": [",
            names[k], (unsigned long long)c->calls, (unsigned long long)c->bytes,
            (unsigned long long)c->small, (unsigned long long)c->maximum);
        for (int b = 0; b < 9; ++b) {
          fprintf(file, "%s%llu", b ? "," : "", (unsigned long long)c->bins[b]);
        }
        fprintf(file, "]}%s\n", k + 1 == NKINDS ? "" : ",");
      }
      fprintf(file, "  }\n}\n");
      fclose(file);
    }
  }
  return PMPI_Finalize();
}
