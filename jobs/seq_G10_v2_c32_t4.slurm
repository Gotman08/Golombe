#!/bin/bash
#SBATCH --job-name=seq_G10_v2_c32_t4
#SBATCH --output=seq_G10_v2_c32_t4_%j.out
#SBATCH --error=seq_G10_v2_c32_t4_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G10 v2 - 32 cores, 4 threads ==="
cd ~/golomb
./build/golomb_v2 10 --csv results/romeo/seq_G10_v2_c32_t4.csv
