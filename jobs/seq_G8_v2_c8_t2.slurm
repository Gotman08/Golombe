#!/bin/bash
#SBATCH --job-name=seq_G8_v2_c8_t2
#SBATCH --output=seq_G8_v2_c8_t2_%j.out
#SBATCH --error=seq_G8_v2_c8_t2_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G8 v2 - 8 cores, 2 threads ==="
cd ~/golomb
./build/golomb_v2 8 --csv results/romeo/seq_G8_v2_c8_t2.csv
