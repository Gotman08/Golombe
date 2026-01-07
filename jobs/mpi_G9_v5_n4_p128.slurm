#!/bin/bash
#SBATCH --job-name=mpi_G9_v5_n4_p128
#SBATCH --output=mpi_G9_v5_n4_p128_%j.out
#SBATCH --error=mpi_G9_v5_n4_p128_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# No OpenMP for v5
export OMP_NUM_THREADS=1

# Run benchmark
echo "=== G9 v5 Pure MPI - 4 nodes, 128 procs ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v5 9 --csv results/romeo/mpi_G9_v5_n4_p128.csv
