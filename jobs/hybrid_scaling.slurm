#!/bin/bash
# =============================================================================
# SLURM Job Script - Hybrid MPI+OpenMP Scaling Study (Romeo 2025)
# =============================================================================
# This script tests different MPI/OpenMP ratios to find optimal configuration
# =============================================================================
#SBATCH --job-name=golomb_scaling
#SBATCH --output=results/romeo/scaling_%j.out
#SBATCH --error=results/romeo/scaling_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --account=r250127

# Load Romeo 2025 environment
source $HOME/golomb/setup_env.sh

# =============================================================================
# Setup
# =============================================================================
SCRATCH_DIR="/scratch_p/${USER}/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
cd "${SCRATCH_DIR}"
cp -r $HOME/golomb/build "${SCRATCH_DIR}/"

mkdir -p results/romeo
RESULT_FILE="results/romeo/hybrid_scaling.csv"

echo "============================================================"
echo "  Hybrid MPI+OpenMP Scaling Study"
echo "  Romeo 2025 - ${SLURM_NNODES} nodes x ${SLURM_CPUS_PER_TASK} CPUs"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================================"
echo ""

# CSV header
echo "config,order,nodes,mpi_ranks,threads,total_workers,time_ms,speedup,efficiency" > "${RESULT_FILE}"

# Baseline: Sequential time for efficiency calculation
ORDER=11
echo "=== Getting baseline (sequential) for G${ORDER} ==="
OMP_NUM_THREADS=1 ./build/golomb_v6 $ORDER --benchmark > baseline.txt 2>&1
BASELINE_TIME=$(grep "Time:" baseline.txt | awk '{print $2}')
echo "Baseline sequential: ${BASELINE_TIME} ms"
echo ""

# =============================================================================
# Test different hybrid configurations (constant total workers = 128)
# =============================================================================
echo "=== Testing Different Hybrid Ratios ==="
echo ""

# Configuration: nodes x ranks_per_node x threads = total
# All configs use ~128 total workers

# Config 1: 4 nodes x 1 rank x 32 threads = 128 (max OpenMP)
echo "--- Config 1: 4 MPI x 32 OpenMP (max shared memory) ---"
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close
mpirun --map-by node -np 4 ./build/golomb_mpi_v4 $ORDER --threads 32 --depth 5 > config1.txt 2>&1
TIME1=$(grep "Time:" config1.txt | awk '{print $2}')
SPEEDUP1=$(echo "scale=2; $BASELINE_TIME / $TIME1" | bc)
EFF1=$(echo "scale=2; $SPEEDUP1 / 128 * 100" | bc)
echo "Time: ${TIME1} ms, Speedup: ${SPEEDUP1}x, Efficiency: ${EFF1}%"
echo "4xMPI_32xOMP,$ORDER,4,4,32,128,$TIME1,$SPEEDUP1,$EFF1" >> "${RESULT_FILE}"

# Config 2: 4 nodes x 2 ranks x 16 threads = 128
echo "--- Config 2: 8 MPI x 16 OpenMP (balanced) ---"
export OMP_NUM_THREADS=16
mpirun --map-by node:PE=16 -np 8 ./build/golomb_mpi_v4 $ORDER --threads 16 --depth 5 > config2.txt 2>&1
TIME2=$(grep "Time:" config2.txt | awk '{print $2}')
SPEEDUP2=$(echo "scale=2; $BASELINE_TIME / $TIME2" | bc)
EFF2=$(echo "scale=2; $SPEEDUP2 / 128 * 100" | bc)
echo "Time: ${TIME2} ms, Speedup: ${SPEEDUP2}x, Efficiency: ${EFF2}%"
echo "8xMPI_16xOMP,$ORDER,4,8,16,128,$TIME2,$SPEEDUP2,$EFF2" >> "${RESULT_FILE}"

# Config 3: 4 nodes x 4 ranks x 8 threads = 128
echo "--- Config 3: 16 MPI x 8 OpenMP ---"
export OMP_NUM_THREADS=8
mpirun --map-by node:PE=8 -np 16 ./build/golomb_mpi_v4 $ORDER --threads 8 --depth 5 > config3.txt 2>&1
TIME3=$(grep "Time:" config3.txt | awk '{print $2}')
SPEEDUP3=$(echo "scale=2; $BASELINE_TIME / $TIME3" | bc)
EFF3=$(echo "scale=2; $SPEEDUP3 / 128 * 100" | bc)
echo "Time: ${TIME3} ms, Speedup: ${SPEEDUP3}x, Efficiency: ${EFF3}%"
echo "16xMPI_8xOMP,$ORDER,4,16,8,128,$TIME3,$SPEEDUP3,$EFF3" >> "${RESULT_FILE}"

# Config 4: 4 nodes x 8 ranks x 4 threads = 128
echo "--- Config 4: 32 MPI x 4 OpenMP ---"
export OMP_NUM_THREADS=4
mpirun --map-by node:PE=4 -np 32 ./build/golomb_mpi_v4 $ORDER --threads 4 --depth 5 > config4.txt 2>&1
TIME4=$(grep "Time:" config4.txt | awk '{print $2}')
SPEEDUP4=$(echo "scale=2; $BASELINE_TIME / $TIME4" | bc)
EFF4=$(echo "scale=2; $SPEEDUP4 / 128 * 100" | bc)
echo "Time: ${TIME4} ms, Speedup: ${SPEEDUP4}x, Efficiency: ${EFF4}%"
echo "32xMPI_4xOMP,$ORDER,4,32,4,128,$TIME4,$SPEEDUP4,$EFF4" >> "${RESULT_FILE}"

# Config 5: Pure MPI (128 ranks x 1 thread)
echo "--- Config 5: 128 MPI x 1 OpenMP (pure MPI) ---"
export OMP_NUM_THREADS=1
mpirun -np 128 ./build/golomb_mpi_v3 $ORDER --depth 5 > config5.txt 2>&1
TIME5=$(grep "Time:" config5.txt | awk '{print $2}')
SPEEDUP5=$(echo "scale=2; $BASELINE_TIME / $TIME5" | bc)
EFF5=$(echo "scale=2; $SPEEDUP5 / 128 * 100" | bc)
echo "Time: ${TIME5} ms, Speedup: ${SPEEDUP5}x, Efficiency: ${EFF5}%"
echo "128xMPI_pure,$ORDER,4,128,1,128,$TIME5,$SPEEDUP5,$EFF5" >> "${RESULT_FILE}"

echo ""

# =============================================================================
# Strong Scaling Test (G12)
# =============================================================================
ORDER=12
echo "=== Strong Scaling Test for G${ORDER} ==="

for NODES in 1 2 4; do
    RANKS=$((NODES * 4))
    THREADS=8
    TOTAL=$((RANKS * THREADS))

    echo "--- ${NODES} node(s), ${RANKS} MPI x ${THREADS} OMP = ${TOTAL} workers ---"
    export OMP_NUM_THREADS=$THREADS
    mpirun --map-by node:PE=$THREADS -np $RANKS ./build/golomb_mpi_v4 $ORDER --threads $THREADS --depth 5 > strong_${NODES}.txt 2>&1
    TIME=$(grep "Time:" strong_${NODES}.txt | awk '{print $2}')
    echo "Time: ${TIME} ms"
    echo "strong_scaling,$ORDER,$NODES,$RANKS,$THREADS,$TOTAL,$TIME,0,0" >> "${RESULT_FILE}"
done

echo ""

# =============================================================================
# Results
# =============================================================================
echo "=== Final Results ==="
cat "${RESULT_FILE}"

# Copy back
mkdir -p $HOME/golomb/results/romeo
cp -v results/romeo/*.csv $HOME/golomb/results/romeo/
cp -v *.txt $HOME/golomb/results/romeo/

cd $HOME
rm -rf "${SCRATCH_DIR}"

echo ""
echo "============================================================"
echo "  Scaling Study Complete"
echo "  End: $(date)"
echo "============================================================"
