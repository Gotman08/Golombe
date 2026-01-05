#!/bin/bash
#SBATCH --job-name=mpi_G11_v3_n1_p4_t8
#SBATCH --output=mpi_G11_v3_n1_p4_t8_%j.out
#SBATCH --error=mpi_G11_v3_n1_p4_t8_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
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
echo "=== G11 v3 - 1 nodes, 4 procs, 8 threads ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v3 11 --threads 8 --csv results/romeo/mpi_G11_v3_n1_p4_t8.csv
