#!/bin/bash
# =============================================================================
# Submit Weak Scaling Jobs to Romeo Cluster
# =============================================================================
# Submits a series of jobs for weak scaling analysis:
#   G10 with 1 node  (32 cores)
#   G11 with 2 nodes (64 cores)
#   G12 with 4 nodes (128 cores)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
JOB_FILE="${PROJECT_DIR}/jobs/weak_scaling.slurm"

# Default MPI version
MPI_VERSION=${MPI_VERSION:-3}
DEPTH=${DEPTH:-5}

echo "=============================================="
echo "  Submitting Weak Scaling Jobs"
echo "  MPI Version: v${MPI_VERSION}"
echo "=============================================="
echo ""

# Check if job file exists
if [[ ! -f "$JOB_FILE" ]]; then
    echo "ERROR: Job file not found: $JOB_FILE"
    exit 1
fi

# Create results directory
mkdir -p "${PROJECT_DIR}/results/romeo/weak_scaling"

# Weak scaling configurations: ORDER:NODES:PROCS
CONFIGS=(
    "10:1:32"
    "11:2:64"
    "12:4:128"
)

# Submit jobs
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r order nodes procs <<< "$config"

    echo "Submitting G${order}: ${nodes} node(s), ${procs} processes..."

    JOB_ID=$(sbatch \
        --export=ORDER=${order},NODES=${nodes},PROCS=${procs},MPI_VERSION=${MPI_VERSION},DEPTH=${DEPTH} \
        --nodes=${nodes} \
        --ntasks=${procs} \
        "$JOB_FILE" | awk '{print $4}')

    echo "  Job ID: $JOB_ID"
done

echo ""
echo "=============================================="
echo "All weak scaling jobs submitted!"
echo "Monitor with: squeue -u $USER"
echo "=============================================="
