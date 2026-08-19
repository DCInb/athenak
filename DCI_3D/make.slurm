#!/bin/bash
#SBATCH --job-name=athenak_make
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

source ~/mengqi/bashrc_athenak

cmake ..\
 -DAthena_ENABLE_MPI=ON \
 -DKokkos_ENABLE_CUDA=ON \
 -DKokkos_ARCH_AMPERE80=ON \
 -DPROBLEM=../../DCI_3D/dci_3d

make -j20