# HPC Deployment Guide

## Quick Start

### Local Execution
```bash
# Build all versions
make all

# Sequential
./build/golomb_v1 10

# OpenMP (8 threads)
OMP_NUM_THREADS=8 ./build/golomb_v2 11

# MPI+OpenMP (4 ranks x 4 threads)
OMP_NUM_THREADS=4 mpirun -np 4 ./build/golomb_v3 12

# Hypercube (8 ranks, power of 2)
OMP_NUM_THREADS=4 mpirun -np 8 ./build/golomb_v4 12
```

## Cluster Deployment

### 1. Deploy to Cluster
```bash
bash scripts/hpc/deploy.sh
```

### 2. Submit Job
```bash
ssh user@cluster
cd ~/golomb
sbatch jobs/hybrid_benchmark.slurm
```

## SLURM Configuration

### Sequential Job
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
```

### MPI Job
```bash
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
```

### CPU Affinity (Important!)
```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

## Scaling Guidelines

| Order | Recommended Version | Configuration |
|-------|---------------------|---------------|
| ≤ 9 | v1 or v2 | 1-8 threads |
| 10-11 | v2 | 8-32 threads |
| 12-13 | v3 or v4 | 4-16 ranks × 8 threads |
| ≥ 14 | v4 | 32+ ranks × 8 threads |

## Performance Tips

1. **Use power-of-2 ranks for v4** (hypercube topology)
2. **Match threads to physical cores** (avoid hyperthreading)
3. **Use fast scratch storage** for I/O
4. **Set proper CPU affinity** for NUMA systems

## Troubleshooting

### MPI not found
```bash
module load openmpi  # or mpich
```

### Poor scaling
- Check CPU affinity settings
- Reduce thread count if oversubscribed
- Use v4 instead of v3 for >8 ranks
