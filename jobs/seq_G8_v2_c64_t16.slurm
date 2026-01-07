#!/bin/bash
#SBATCH --job-name=seq_G8_v2_c64_t16
#SBATCH --output=seq_G8_v2_c64_t16_%j.out
#SBATCH --error=seq_G8_v2_c64_t16_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G8 v2 - 64 cores, 16 threads ==="
cd ~/golomb
./build/golomb_v2 8 --csv results/romeo/seq_G8_v2_c64_t16.csv
