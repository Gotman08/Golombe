#!/bin/bash
#SBATCH --job-name=seq_G10_v1_c1_t1
#SBATCH --output=seq_G10_v1_c1_t1_%j.out
#SBATCH --error=seq_G10_v1_c1_t1_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G10 v1 - 1 cores, 1 threads ==="
cd ~/golomb
./build/golomb_v1 10 --csv results/romeo/seq_G10_v1_c1_t1.csv
