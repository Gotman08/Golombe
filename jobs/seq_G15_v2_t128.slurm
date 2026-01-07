#!/bin/bash
#SBATCH --job-name=golomb_G15_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --output=results/romeo/G15_v2_%j.out
#SBATCH --error=results/romeo/G15_v2_%j.err

source $HOME/golomb/setup_env.sh
cd $HOME/golomb

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "================================================"
echo "Golomb G15 - v2 OpenMP (128 threads)"
echo "Started at $(date)"
echo "Host: $(hostname)"
echo "================================================"

./build/golomb_v2 15 --csv results/romeo/G15_v2_t128.csv

echo "================================================"
echo "Finished at $(date)"
echo "================================================"
