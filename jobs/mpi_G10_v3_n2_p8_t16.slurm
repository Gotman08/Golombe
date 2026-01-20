#!/bin/bash
#SBATCH --job-name=mpi_G10_v3_n2_p8_t16
#SBATCH --output=mpi_G10_v3_n2_p8_t16_%j.out
#SBATCH --error=mpi_G10_v3_n2_p8_t16_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G10 v3 - 2 nodes, 8 procs, 16 threads ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v3 10 --threads 16 --csv results/romeo/mpi_G10_v3_n2_p8_t16.csv
