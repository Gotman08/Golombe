#!/bin/bash
#SBATCH --job-name=seq_G14_v2_c128_t1
#SBATCH --output=seq_G14_v2_c128_t1_%j.out
#SBATCH --error=seq_G14_v2_c128_t1_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL

# OpenMP configuration
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G14 v2 - 128 cores, 1 threads ==="
cd ~/golomb
./build/golomb_v2 14 --csv results/romeo/seq_G14_v2_c128_t1.csv
