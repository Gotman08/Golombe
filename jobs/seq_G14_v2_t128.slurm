#!/bin/bash
#SBATCH --job-name=golomb_seq_G14_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --output=results/romeo/seq_G14_v2_t128_%j.out
#SBATCH --error=results/romeo/seq_G14_v2_t128_%j.err

source $HOME/golomb/setup_env.sh
cd $HOME/golomb

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== Golomb G14 - v2 OpenMP (128 threads) ==="
echo "Started: $(date)"
./build/golomb_v2 14 --csv results/romeo/seq_G14_v2_t128.csv
echo "Finished: $(date)"
