#!/bin/bash
#SBATCH --job-name=mpi_G11_v5_n2_p64
#SBATCH --output=mpi_G11_v5_n2_p64_%j.out
#SBATCH --error=mpi_G11_v5_n2_p64_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# No OpenMP for v5
export OMP_NUM_THREADS=1

# Run benchmark
echo "=== G11 v5 Pure MPI - 2 nodes, 64 procs ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v5 11 --csv results/romeo/mpi_G11_v5_n2_p64.csv
