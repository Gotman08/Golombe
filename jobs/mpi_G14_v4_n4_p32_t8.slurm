#!/bin/bash
#SBATCH --job-name=mpi_G14_v4_n4_p32_t8
#SBATCH --output=mpi_G14_v4_n4_p32_t8_%j.out
#SBATCH --error=mpi_G14_v4_n4_p32_t8_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x64cpu
#SBATCH --account=r250127
#SBATCH --export=ALL
#SBATCH --mem=8G

# OpenMP configuration
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G14 v4 - 4 nodes, 32 procs, 8 threads (256 CPUs) ==="
cd ~/golomb
srun --mpi=pmix ./build/golomb_v4 14 --threads 8 --csv results/romeo/mpi_G14_v4_n4_p32_t8.csv
