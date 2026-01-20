#!/bin/bash
# =============================================================================
# SLURM Job Script - MPI Parallel Golomb Solver (Romeo 2025)
# =============================================================================
#SBATCH --job-name=golomb_mpi_G13_v3_p16
#SBATCH --output=results/romeo/mpi_G13_v3_p16_%j.out
#SBATCH --error=results/romeo/mpi_G13_v3_p16_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --constraint=x64cpu
#SBATCH --distribution=block:block
#SBATCH --partition=short
#SBATCH --account=r250127

# =============================================================================
# CPU Affinity Settings (Performance Optimization)
# =============================================================================
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Load Romeo 2025 environment
source $HOME/golomb/setup_env.sh

# =============================================================================
# Setup scratch directory for fast I/O (Romeo best practice)
# =============================================================================
SCRATCH_DIR="/scratch_p/${USER}/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
cd "${SCRATCH_DIR}"

# Copy necessary files to scratch
cp -r $HOME/golomb/build "${SCRATCH_DIR}/"
cp -r $HOME/golomb/scripts "${SCRATCH_DIR}/"

# Job info
echo "=============================================="
echo "  Golomb Ruler Solver - MPI Benchmark"
echo "  Romeo 2025 - Architecture: x64cpu"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Scratch: ${SCRATCH_DIR}"
echo "Start time: $(date)"
echo "Order: 13"
echo "MPI Version: v3"
echo "Processes: 16"
echo "Nodes requested: 1"
echo "=============================================="
echo ""

# Create results directory in scratch
mkdir -p results/romeo

# Define file paths
RESULT_FILE="results/romeo/mpi_G13_v3_p16.csv"
OUTPUT_FILE="results/romeo/mpi_G13_v3_p16_raw.txt"

# Determine optimal prefix depth based on order and process count
if [ 13 -le 8 ]; then
    DEPTH=3
elif [ 13 -le 10 ]; then
    DEPTH=4
else
    DEPTH=5
fi

echo "Using prefix depth: $DEPTH"
echo ""

# =============================================================================
# Run MPI solver based on version
# =============================================================================
# MPI v1: Basic master/worker - args: <order> [prefix_depth]
# MPI v2: Hypercube - args: <order> [prefix_depth]
# MPI v3: Optimized - args: <order> [--depth N] [--csv FILE]
# =============================================================================

case 3 in
    3)
        # MPI v3: Has --depth and --csv flags
        echo "Running: mpirun -np 16 ./build/golomb_mpi_v3 13 --depth $DEPTH"
        mpirun -np 16 ./build/golomb_mpi_v3 13 --depth $DEPTH > "${OUTPUT_FILE}" 2>&1

        # Parse output to CSV
        source scripts/romeo/parse_output.sh
        write_mpi_header > "${RESULT_FILE}"
        parse_mpi "${OUTPUT_FILE}" 3 13 16 >> "${RESULT_FILE}"
        ;;

    2)
        # MPI v2: Hypercube topology - positional args only
        echo "Running: mpirun -np 16 ./build/golomb_mpi_v2 13 $DEPTH"
        mpirun -np 16 ./build/golomb_mpi_v2 13 $DEPTH > "${OUTPUT_FILE}" 2>&1

        # Parse output to CSV
        source scripts/romeo/parse_output.sh
        write_mpi_header > "${RESULT_FILE}"
        parse_mpi "${OUTPUT_FILE}" 3 13 16 >> "${RESULT_FILE}"
        ;;

    1)
        # MPI v1: Basic master/worker - positional args only
        echo "Running: mpirun -np 16 ./build/golomb_mpi_v1 13 $DEPTH"
        mpirun -np 16 ./build/golomb_mpi_v1 13 $DEPTH > "${OUTPUT_FILE}" 2>&1

        # Parse output to CSV
        source scripts/romeo/parse_output.sh
        write_mpi_header > "${RESULT_FILE}"
        parse_mpi "${OUTPUT_FILE}" 3 13 16 >> "${RESULT_FILE}"
        ;;

    *)
        echo "ERROR: Unknown MPI version 3"
        exit 1
        ;;
esac

# Show raw output
echo ""
echo "=== Raw Output ==="
cat "${OUTPUT_FILE}"

# Show CSV result
echo ""
echo "=== CSV Result ==="
cat "${RESULT_FILE}"

# =============================================================================
# Copy results back to home and cleanup scratch
# =============================================================================
echo ""
echo "=== Copying results to home directory ==="
mkdir -p $HOME/golomb/results/romeo
cp -v results/romeo/*.csv $HOME/golomb/results/romeo/ 2>/dev/null || true
cp -v results/romeo/*_raw.txt $HOME/golomb/results/romeo/ 2>/dev/null || true

# Cleanup scratch directory
echo ""
echo "=== Cleaning up scratch ==="
cd $HOME
rm -rf "${SCRATCH_DIR}"

echo ""
echo "=============================================="
echo "Results saved to: $HOME/golomb/${RESULT_FILE}"
echo "End time: $(date)"
echo "=============================================="
