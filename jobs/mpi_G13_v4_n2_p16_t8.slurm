#!/bin/bash
#SBATCH --job-name=mpi_G13_v4_n2_p16_t8
#SBATCH --output=mpi_G13_v4_n2_p16_t8_%j.out
#SBATCH --error=mpi_G13_v4_n2_p16_t8_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G13 v4 - 2 nodes, 16 procs, 8 threads ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v4 13 --threads 8 --csv results/romeo/mpi_G13_v4_n2_p16_t8.csv
