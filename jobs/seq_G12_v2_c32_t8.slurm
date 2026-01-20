#!/bin/bash
#SBATCH --job-name=seq_G12_v2_c32_t8
#SBATCH --output=seq_G12_v2_c32_t8_%j.out
#SBATCH --error=seq_G12_v2_c32_t8_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G12 v2 - 32 cores, 8 threads ==="
cd ~/golomb
./build/golomb_v2 12 --csv results/romeo/seq_G12_v2_c32_t8.csv
