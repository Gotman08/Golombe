#!/bin/bash
# =============================================================================
# Weak Scaling Benchmark Script for Golomb Ruler Solver
# =============================================================================
# Weak Scaling: Problem size increases with processor count
# Maintains approximately constant work per processor
#
# Configuration:
#   1 node  (32 cores)  -> G10 (~2.5M nodes)
#   2 nodes (64 cores)  -> G11 (~25M nodes)
#   4 nodes (128 cores) -> G12 (~250M nodes)
#
# Ideal weak scaling: execution time remains constant as both problem
# size and processor count increase proportionally.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_DIR}/results/weak_scaling"

# Default values
MPI_VERSION=3
DEPTH=5
DRY_RUN=false
VERBOSE=false

usage() {
    echo "Weak Scaling Benchmark for Golomb Ruler Solver"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --version N    MPI version to use (1, 2, or 3) [default: 3]"
    echo "  --depth N      Prefix depth for work distribution [default: 5]"
    echo "  --dry-run      Print commands without executing"
    echo "  --verbose      Show detailed output"
    echo "  --help         Show this help message"
    echo ""
    echo "Weak Scaling Configuration:"
    echo "  1 node  (32 procs) -> G10"
    echo "  2 nodes (64 procs) -> G11"
    echo "  4 nodes (128 procs) -> G12"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            MPI_VERSION="$2"
            shift 2
            ;;
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  Weak Scaling Benchmark"
echo "  MPI Version: v${MPI_VERSION}"
echo "  Results: ${RESULTS_DIR}"
echo "=============================================="
echo ""

# Define weak scaling configurations
# Format: "order:nodes:procs"
CONFIGS=(
    "10:1:32"
    "11:2:64"
    "12:4:128"
)

# CSV output file
CSV_FILE="${RESULTS_DIR}/weak_scaling_v${MPI_VERSION}.csv"

# Write CSV header
echo "order,nodes,procs,time_ms,nodes_explored,speedup,efficiency" > "$CSV_FILE"

# Run benchmarks
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r order nodes procs <<< "$config"

    echo "----------------------------------------------"
    echo "G${order}: ${nodes} node(s), ${procs} processes"
    echo "----------------------------------------------"

    EXECUTABLE="${PROJECT_DIR}/build/golomb_mpi_v${MPI_VERSION}"

    if [[ ! -f "$EXECUTABLE" ]]; then
        echo "ERROR: Executable not found: $EXECUTABLE"
        echo "Please run 'make mpi_v${MPI_VERSION}' first."
        exit 1
    fi

    CMD="mpirun -np ${procs} ${EXECUTABLE} ${order} --depth ${DEPTH}"

    if $DRY_RUN; then
        echo "Would run: $CMD"
        continue
    fi

    if $VERBOSE; then
        echo "Running: $CMD"
    fi

    # Run and capture output
    OUTPUT_FILE="${RESULTS_DIR}/weak_G${order}_v${MPI_VERSION}_p${procs}.txt"

    if $VERBOSE; then
        $CMD 2>&1 | tee "$OUTPUT_FILE"
    else
        $CMD > "$OUTPUT_FILE" 2>&1
    fi

    # Parse results from output
    TIME_MS=$(grep -oP "Time:\s*\K[\d.]+" "$OUTPUT_FILE" || echo "0")
    NODES_EXPLORED=$(grep -oP "Total nodes:\s*\K[\d]+" "$OUTPUT_FILE" || echo "0")

    # For weak scaling, efficiency is measured relative to the single-node case
    # We'll compute this after all runs
    echo "${order},${nodes},${procs},${TIME_MS},${NODES_EXPLORED},," >> "$CSV_FILE"

    echo "  Time: ${TIME_MS} ms"
    echo "  Nodes explored: ${NODES_EXPLORED}"
    echo ""
done

echo "=============================================="
echo "Weak Scaling Complete"
echo "Results saved to: $CSV_FILE"
echo "=============================================="

# Post-process to compute efficiency
# In weak scaling, ideal efficiency = 1 (constant time regardless of scale)
if ! $DRY_RUN; then
    echo ""
    echo "Computing weak scaling efficiency..."

    # Get baseline time (first configuration)
    BASELINE_TIME=$(awk -F',' 'NR==2 {print $4}' "$CSV_FILE")

    if [[ -n "$BASELINE_TIME" && "$BASELINE_TIME" != "0" ]]; then
        # Create temporary file with efficiency computed
        TMP_FILE="${CSV_FILE}.tmp"
        head -1 "$CSV_FILE" > "$TMP_FILE"

        tail -n +2 "$CSV_FILE" | while IFS=',' read -r order nodes procs time_ms nodes_explored _ _; do
            if [[ -n "$time_ms" && "$time_ms" != "0" ]]; then
                # Weak scaling efficiency = baseline_time / current_time
                efficiency=$(echo "scale=4; $BASELINE_TIME / $time_ms" | bc)
                echo "${order},${nodes},${procs},${time_ms},${nodes_explored},1,${efficiency}"
            else
                echo "${order},${nodes},${procs},${time_ms},${nodes_explored},1,0"
            fi
        done >> "$TMP_FILE"

        mv "$TMP_FILE" "$CSV_FILE"

        echo "Efficiency computed (baseline: ${BASELINE_TIME} ms)"
    fi
fi
