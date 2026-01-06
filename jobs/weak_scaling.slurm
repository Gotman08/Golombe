#!/bin/bash
# =============================================================================
# SLURM Job Script - Weak Scaling Benchmark (Romeo 2025)
# =============================================================================
# Weak Scaling Analysis: Problem size scales with processor count
#
# Configuration:
#   Job 1: G10, 1 node,  32 cores
#   Job 2: G11, 2 nodes, 64 cores
#   Job 3: G12, 4 nodes, 128 cores
#
# Usage:
#   sbatch --export=ORDER=10,NODES=1,PROCS=32 jobs/weak_scaling.slurm
#   sbatch --export=ORDER=11,NODES=2,PROCS=64 jobs/weak_scaling.slurm
#   sbatch --export=ORDER=12,NODES=4,PROCS=128 jobs/weak_scaling.slurm
#
# Or use the helper script:
#   bash scripts/romeo/submit_weak_scaling.sh
# =============================================================================

#SBATCH --job-name=golomb_weak_G${ORDER}
#SBATCH --output=results/romeo/weak_G%ORDER%_p%PROCS%_%j.out
#SBATCH --error=results/romeo/weak_G%ORDER%_p%PROCS%_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=${NODES:-1}
#SBATCH --ntasks=${PROCS:-32}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --constraint=rome
# CPU affinity is handled by srun, not sbatch
#SBATCH --distribution=block:block
# Uncomment for specific partition/account:
# #SBATCH --partition=cpu
# #SBATCH --account=your_account

# =============================================================================
# Environment Setup
# =============================================================================
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Default values if not provided
ORDER=${ORDER:-10}
NODES=${NODES:-1}
PROCS=${PROCS:-32}
MPI_VERSION=${MPI_VERSION:-3}
DEPTH=${DEPTH:-5}

# Load Romeo 2025 environment
if [[ -f "$HOME/golomb/setup_env.sh" ]]; then
    source "$HOME/golomb/setup_env.sh"
fi

# =============================================================================
# Setup scratch directory for fast I/O (Romeo best practice)
# =============================================================================
SCRATCH_DIR="/scratch_p/${USER}/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
cd "${SCRATCH_DIR}"

# Copy necessary files to scratch
cp -r $HOME/golomb/build "${SCRATCH_DIR}/"
cp -r $HOME/golomb/scripts "${SCRATCH_DIR}/"

# =============================================================================
# Job Information
# =============================================================================
echo "=============================================="
echo "  Golomb Ruler Solver - Weak Scaling Benchmark"
echo "  Romeo 2025"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Scratch: ${SCRATCH_DIR}"
echo "Start time: $(date)"
echo ""
echo "Weak Scaling Configuration:"
echo "  Order: G${ORDER}"
echo "  Nodes: ${NODES}"
echo "  MPI Processes: ${PROCS}"
echo "  MPI Version: v${MPI_VERSION}"
echo "  Prefix Depth: ${DEPTH}"
echo "=============================================="
echo ""

# Create results directory
mkdir -p results/romeo

# =============================================================================
# Run MPI Solver
# =============================================================================
RESULT_FILE="results/romeo/weak_G${ORDER}_v${MPI_VERSION}_p${PROCS}.csv"
OUTPUT_FILE="results/romeo/weak_G${ORDER}_v${MPI_VERSION}_p${PROCS}_raw.txt"

case $MPI_VERSION in
    3)
        echo "Running: mpirun -np ${PROCS} ./build/golomb_mpi_v3 ${ORDER} --depth ${DEPTH}"
        mpirun -np ${PROCS} ./build/golomb_mpi_v3 ${ORDER} --depth ${DEPTH} > "${OUTPUT_FILE}" 2>&1
        ;;
    2)
        echo "Running: mpirun -np ${PROCS} ./build/golomb_mpi_v2 ${ORDER} ${DEPTH}"
        mpirun -np ${PROCS} ./build/golomb_mpi_v2 ${ORDER} ${DEPTH} > "${OUTPUT_FILE}" 2>&1
        ;;
    1)
        echo "Running: mpirun -np ${PROCS} ./build/golomb_mpi_v1 ${ORDER} ${DEPTH}"
        mpirun -np ${PROCS} ./build/golomb_mpi_v1 ${ORDER} ${DEPTH} > "${OUTPUT_FILE}" 2>&1
        ;;
    *)
        echo "ERROR: Unknown MPI version ${MPI_VERSION}"
        exit 1
        ;;
esac

# =============================================================================
# Parse and Display Results
# =============================================================================
echo ""
echo "=== Raw Output ==="
cat "${OUTPUT_FILE}"

# Extract metrics for CSV
TIME_MS=$(grep -oP "Time:\s*\K[\d.]+" "${OUTPUT_FILE}" || echo "0")
NODES_EXPLORED=$(grep -oP "Total nodes:\s*\K[\d]+" "${OUTPUT_FILE}" || echo "0")
SOLUTION=$(grep -oP "Solution:\s*\K.*" "${OUTPUT_FILE}" || echo "")
LENGTH=$(grep -oP "Length:\s*\K[\d]+" "${OUTPUT_FILE}" || echo "0")

# Write CSV
echo "order,nodes,procs,mpi_version,depth,time_ms,nodes_explored,length,solution" > "${RESULT_FILE}"
echo "${ORDER},${NODES},${PROCS},${MPI_VERSION},${DEPTH},${TIME_MS},${NODES_EXPLORED},${LENGTH},\"${SOLUTION}\"" >> "${RESULT_FILE}"

echo ""
echo "=== CSV Result ==="
cat "${RESULT_FILE}"

# =============================================================================
# Copy Results and Cleanup
# =============================================================================
echo ""
echo "=== Copying results to home directory ==="
mkdir -p $HOME/golomb/results/romeo/weak_scaling
cp -v results/romeo/weak_*.csv $HOME/golomb/results/romeo/weak_scaling/ 2>/dev/null || true
cp -v results/romeo/weak_*_raw.txt $HOME/golomb/results/romeo/weak_scaling/ 2>/dev/null || true

echo ""
echo "=== Cleaning up scratch ==="
cd $HOME
rm -rf "${SCRATCH_DIR}"

echo ""
echo "=============================================="
echo "Weak Scaling Job Complete"
echo "Results: $HOME/golomb/results/romeo/weak_scaling/"
echo "End time: $(date)"
echo "=============================================="
