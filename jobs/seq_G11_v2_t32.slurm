#!/bin/bash
# =============================================================================
# SLURM Job Script - Sequential Golomb Solver (Romeo 2025)
# =============================================================================
#SBATCH --job-name=golomb_seq_G11_v2
#SBATCH --output=results/romeo/seq_G11_v2_%j.out
#SBATCH --error=results/romeo/seq_G11_v2_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --constraint=x64cpu
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
echo "  Golomb Ruler Solver - Sequential Benchmark"
echo "  Romeo 2025 - Architecture: x64cpu"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Scratch: ${SCRATCH_DIR}"
echo "Start time: $(date)"
echo "Order: 11"
echo "Version: v2"
echo "Threads: 32"
echo "=============================================="
echo ""

# Set OpenMP threads for v6
export OMP_NUM_THREADS=32

# Create results directory in scratch
mkdir -p results/romeo

# Define file paths
RESULT_FILE="results/romeo/seq_G11_v2_t32.csv"
OUTPUT_FILE="results/romeo/seq_G11_v2_t32_raw.txt"

# =============================================================================
# Run solver based on version
# =============================================================================
# v1-v4: No --csv support, need to parse output
# v5: Has --csv support
# v6: Has --csv support with threads column
# =============================================================================

case 2 in
    5)
        # v5: Native CSV support
        echo "Running: ./build/golomb_v5 11"
        ./build/golomb_v5 11 > "${OUTPUT_FILE}" 2>&1

        # Parse output to CSV (v5 stdout doesn't write CSV directly without --csv flag to file)
        source scripts/romeo/parse_output.sh
        write_seq_header > "${RESULT_FILE}"
        parse_sequential "${OUTPUT_FILE}" 2 11 >> "${RESULT_FILE}"
        ;;

    6)
        # v6: Native CSV support with threads
        echo "Running: OMP_NUM_THREADS=32 ./build/golomb_v6 11 --threads 32"
        ./build/golomb_v6 11 --threads 32 > "${OUTPUT_FILE}" 2>&1

        # Parse output to CSV
        source scripts/romeo/parse_output.sh
        write_v6_header > "${RESULT_FILE}"
        parse_v6 "${OUTPUT_FILE}" 2 11 32 >> "${RESULT_FILE}"
        ;;

    1)
        # v1: Brute force - args: <order> [max_length]
        echo "Running: ./build/golomb_v1 11"
        ./build/golomb_v1 11 > "${OUTPUT_FILE}" 2>&1

        source scripts/romeo/parse_output.sh
        write_seq_header > "${RESULT_FILE}"
        parse_sequential "${OUTPUT_FILE}" 2 11 >> "${RESULT_FILE}"
        ;;

    2)
        # v2: Backtracking - args: <order> [max_length] [--no-symmetry]
        echo "Running: ./build/golomb_v2 11"
        ./build/golomb_v2 11 > "${OUTPUT_FILE}" 2>&1

        source scripts/romeo/parse_output.sh
        write_seq_header > "${RESULT_FILE}"
        parse_sequential "${OUTPUT_FILE}" 2 11 >> "${RESULT_FILE}"
        ;;

    3)
        # v3: Branch & bound - args: <order> [--no-symmetry]
        echo "Running: ./build/golomb_v3 11"
        ./build/golomb_v3 11 > "${OUTPUT_FILE}" 2>&1

        source scripts/romeo/parse_output.sh
        write_seq_header > "${RESULT_FILE}"
        parse_sequential "${OUTPUT_FILE}" 2 11 >> "${RESULT_FILE}"
        ;;

    4)
        # v4: Optimized - args: <order> [--no-symmetry]
        echo "Running: ./build/golomb_v4 11"
        ./build/golomb_v4 11 > "${OUTPUT_FILE}" 2>&1

        source scripts/romeo/parse_output.sh
        write_seq_header > "${RESULT_FILE}"
        parse_sequential "${OUTPUT_FILE}" 2 11 >> "${RESULT_FILE}"
        ;;

    *)
        echo "ERROR: Unknown version 2"
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
