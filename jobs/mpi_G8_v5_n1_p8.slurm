#!/bin/bash
#SBATCH --job-name=mpi_G8_v5_n1_p8
#SBATCH --output=mpi_G8_v5_n1_p8_%j.out
#SBATCH --error=mpi_G8_v5_n1_p8_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# No OpenMP for v5
export OMP_NUM_THREADS=1

# Run benchmark
echo "=== G8 v5 Pure MPI - 1 nodes, 8 procs ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v5 8 --csv results/romeo/mpi_G8_v5_n1_p8.csv
