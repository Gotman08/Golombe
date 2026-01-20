#!/bin/bash
#SBATCH --job-name=golomb_G15_v4
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --output=results/romeo/G15_v4_%j.out
#SBATCH --error=results/romeo/G15_v4_%j.err

source $HOME/golomb/setup_env.sh
cd $HOME/golomb

echo "================================================"
echo "Golomb G15 - v4 Hypercube (8 nodes, 128 procs)"
echo "Started at $(date)"
echo "Hosts: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "================================================"

mpirun -np 128 ./build/golomb_v4 15 --csv results/romeo/G15_v4_p128.csv

echo "================================================"
echo "Finished at $(date)"
echo "================================================"
