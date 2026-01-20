#!/bin/bash
#SBATCH --job-name=mpi_G14_v4_n8_p64_t8
#SBATCH --output=mpi_G14_v4_n8_p64_t8_%j.out
#SBATCH --error=mpi_G14_v4_n8_p64_t8_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL
#SBATCH --mem=8G

# OpenMP configuration
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G14 v4 - 8 nodes, 64 procs, 8 threads ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v4 14 --threads 8 --csv results/romeo/mpi_G14_v4_n8_p64_t8.csv
