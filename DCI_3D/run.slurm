#!/bin/bash
#SBATCH --job-name=athenak_run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=2

source ~/mengqi/bashrc_athenak

mpirun -n 4 src/athena -i dci_3d.athinput