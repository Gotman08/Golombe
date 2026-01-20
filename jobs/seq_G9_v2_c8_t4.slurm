#!/bin/bash
#SBATCH --job-name=seq_G9_v2_c8_t4
#SBATCH --output=seq_G9_v2_c8_t4_%j.out
#SBATCH --error=seq_G9_v2_c8_t4_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G9 v2 - 8 cores, 4 threads ==="
cd ~/golomb
./build/golomb_v2 9 --csv results/romeo/seq_G9_v2_c8_t4.csv
