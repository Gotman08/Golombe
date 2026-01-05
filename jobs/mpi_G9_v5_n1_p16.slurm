#!/bin/bash
#SBATCH --job-name=mpi_G9_v5_n1_p16
#SBATCH --output=mpi_G9_v5_n1_p16_%j.out
#SBATCH --error=mpi_G9_v5_n1_p16_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# No OpenMP for v5
export OMP_NUM_THREADS=1

# Run benchmark
echo "=== G9 v5 Pure MPI - 1 nodes, 16 procs ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v5 9 --csv results/romeo/mpi_G9_v5_n1_p16.csv
