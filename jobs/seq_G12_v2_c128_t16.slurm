#!/bin/bash
#SBATCH --job-name=seq_G12_v2_c128_t16
#SBATCH --output=seq_G12_v2_c128_t16_%j.out
#SBATCH --error=seq_G12_v2_c128_t16_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G12 v2 - 128 cores, 16 threads ==="
cd ~/golomb
./build/golomb_v2 12 --csv results/romeo/seq_G12_v2_c128_t16.csv
