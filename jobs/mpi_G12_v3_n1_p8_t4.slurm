#!/bin/bash
#SBATCH --job-name=mpi_G12_v3_n1_p8_t4
#SBATCH --output=mpi_G12_v3_n1_p8_t4_%j.out
#SBATCH --error=mpi_G12_v3_n1_p8_t4_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G12 v3 - 1 nodes, 8 procs, 4 threads ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v3 12 --threads 4 --csv results/romeo/mpi_G12_v3_n1_p8_t4.csv
