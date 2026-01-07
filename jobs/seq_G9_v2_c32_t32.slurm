#!/bin/bash
#SBATCH --job-name=seq_G9_v2_c32_t32
#SBATCH --output=seq_G9_v2_c32_t32_%j.out
#SBATCH --error=seq_G9_v2_c32_t32_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G9 v2 - 32 cores, 32 threads ==="
cd ~/golomb
./build/golomb_v2 9 --csv results/romeo/seq_G9_v2_c32_t32.csv
