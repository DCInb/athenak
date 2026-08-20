#!/bin/bash
#SBATCH --job-name=athenak_run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8

source ~/mengqi/bashrc_athenak

# The qualified same-node transport list avoids the pathological cuda_ipc path while
# retaining CUDA-aware copies and shared-memory host transport.
export UCX_TLS=self,sm,tcp,cuda_copy

mpirun -n 4 src/athena --kokkos-map-device-id-by=mpi_rank \
  -i dci_3d.athinput
