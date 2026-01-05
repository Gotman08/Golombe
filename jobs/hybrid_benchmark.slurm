#!/bin/bash
# =============================================================================
# SLURM Job Script - Hybrid MPI+OpenMP Golomb Solver (Romeo 2025)
# =============================================================================
# This script demonstrates HPC hybrid parallelism:
# - MPI for inter-node communication (distributed memory)
# - OpenMP for intra-node parallelism (shared memory)
# =============================================================================
#SBATCH --job-name=golomb_hybrid
#SBATCH --output=results/romeo/hybrid_%j.out
#SBATCH --error=results/romeo/hybrid_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --account=r250127

# Configuration: 2 nodes x 2 MPI ranks x 16 OpenMP threads = 64 workers
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Load Romeo 2025 environment
source $HOME/golomb/setup_env.sh

# =============================================================================
# Setup scratch directory for fast I/O
# =============================================================================
SCRATCH_DIR="/scratch_p/${USER}/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
cd "${SCRATCH_DIR}"

# Copy necessary files
cp -r $HOME/golomb/build "${SCRATCH_DIR}/"
cp -r $HOME/golomb/scripts "${SCRATCH_DIR}/"

# =============================================================================
# Job Information
# =============================================================================
echo "============================================================"
echo "  Golomb Ruler Solver - Hybrid MPI+OpenMP Benchmark"
echo "  Romeo 2025 - HPC Architecture"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "MPI Ranks: $SLURM_NTASKS (${SLURM_NTASKS_PER_NODE} per node)"
echo "OpenMP Threads per rank: $OMP_NUM_THREADS"
echo "Total workers: $((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "Scratch: ${SCRATCH_DIR}"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Create results directory
mkdir -p results/romeo

# =============================================================================
# Benchmark different configurations
# =============================================================================
RESULT_FILE="results/romeo/hybrid_benchmark.csv"
echo "version,order,nodes,mpi_ranks,threads_per_rank,total_workers,time_ms,solution,length" > "${RESULT_FILE}"

# Orders to benchmark
ORDERS="11 12"

for ORDER in $ORDERS; do
    echo ""
    echo "=============================================="
    echo "  Benchmarking G${ORDER}"
    echo "=============================================="

    # Calculate optimal depth
    if [ $ORDER -le 10 ]; then
        DEPTH=4
    else
        DEPTH=5
    fi

    OUTPUT_FILE="results/romeo/hybrid_G${ORDER}_raw.txt"

    echo "Running: mpirun ./build/golomb_mpi_v4 $ORDER --threads $OMP_NUM_THREADS --depth $DEPTH"
    echo ""

    # Run hybrid solver
    mpirun ./build/golomb_mpi_v4 $ORDER --threads $OMP_NUM_THREADS --depth $DEPTH > "${OUTPUT_FILE}" 2>&1

    # Show output
    cat "${OUTPUT_FILE}"

    # Parse results
    TIME=$(grep "Time:" "${OUTPUT_FILE}" | awk '{print $2}')
    LENGTH=$(grep "Length:" "${OUTPUT_FILE}" | awk '{print $2}')
    SOLUTION=$(grep "Solution:" "${OUTPUT_FILE}" | cut -d: -f2 | tr -d ' ')

    # Append to CSV
    echo "4,$ORDER,$SLURM_NNODES,$SLURM_NTASKS,$OMP_NUM_THREADS,$((SLURM_NTASKS * OMP_NUM_THREADS)),$TIME,\"$SOLUTION\",$LENGTH" >> "${RESULT_FILE}"

    echo ""
    echo "G${ORDER} completed: ${TIME} ms, Length: ${LENGTH}"
    echo ""
done

# =============================================================================
# Compare with pure MPI (same total workers)
# =============================================================================
echo ""
echo "=============================================="
echo "  Comparison: Pure MPI vs Hybrid"
echo "=============================================="

# Pure MPI: Use all CPUs as separate MPI ranks
PURE_MPI_RANKS=$((SLURM_NTASKS * OMP_NUM_THREADS))

for ORDER in $ORDERS; do
    echo ""
    echo "--- G${ORDER}: Pure MPI ($PURE_MPI_RANKS ranks) ---"

    OUTPUT_FILE="results/romeo/pure_mpi_G${ORDER}_raw.txt"

    # Adjust threads for pure MPI
    OMP_NUM_THREADS=1 mpirun -np $PURE_MPI_RANKS ./build/golomb_mpi_v3 $ORDER --depth 5 > "${OUTPUT_FILE}" 2>&1

    TIME=$(grep "Time:" "${OUTPUT_FILE}" | awk '{print $2}')
    LENGTH=$(grep "Length:" "${OUTPUT_FILE}" | awk '{print $2}')

    echo "Pure MPI: ${TIME} ms, Length: ${LENGTH}"
    echo "3,$ORDER,$SLURM_NNODES,$PURE_MPI_RANKS,1,$PURE_MPI_RANKS,$TIME,\"..\",$LENGTH" >> "${RESULT_FILE}"
done

# =============================================================================
# Copy results back and cleanup
# =============================================================================
echo ""
echo "=== Results Summary ==="
cat "${RESULT_FILE}"

echo ""
echo "=== Copying results to home ==="
mkdir -p $HOME/golomb/results/romeo
cp -v results/romeo/*.csv $HOME/golomb/results/romeo/ 2>/dev/null || true
cp -v results/romeo/*_raw.txt $HOME/golomb/results/romeo/ 2>/dev/null || true

# Cleanup
cd $HOME
rm -rf "${SCRATCH_DIR}"

echo ""
echo "============================================================"
echo "  Benchmark Complete"
echo "  Results: $HOME/golomb/results/romeo/hybrid_benchmark.csv"
echo "  End time: $(date)"
echo "============================================================"
