#!/bin/bash
#SBATCH --job-name=mpi_G10_v5_n8_p256
#SBATCH --output=mpi_G10_v5_n8_p256_%j.out
#SBATCH --error=mpi_G10_v5_n8_p256_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# No OpenMP for v5
export OMP_NUM_THREADS=1

# Run benchmark
echo "=== G10 v5 Pure MPI - 8 nodes, 256 procs ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v5 10 --csv results/romeo/mpi_G10_v5_n8_p256.csv
