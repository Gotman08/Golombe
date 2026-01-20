#!/bin/bash
#SBATCH --job-name=golomb_G15_v3
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --output=results/romeo/G15_v3_%j.out
#SBATCH --error=results/romeo/G15_v3_%j.err

source $HOME/golomb/setup_env.sh
cd $HOME/golomb

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "================================================"
echo "Golomb G15 - v3 MPI (4 nodes, 32 procs x 4 threads)"
echo "Started at $(date)"
echo "Hosts: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "================================================"

mpirun -np 32 ./build/golomb_v3 15 --threads 4 --csv results/romeo/G15_v3_p32.csv

echo "================================================"
echo "Finished at $(date)"
echo "================================================"
