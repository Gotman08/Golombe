#!/bin/bash
# =============================================================================
# Submit Golomb Benchmark Jobs to SLURM on Romeo
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if running on Romeo or locally
if [[ -f "${SCRIPT_DIR}/config.sh" ]]; then
    source "${SCRIPT_DIR}/config.sh"
fi

# Detect if we're on the cluster or local machine
if command -v sbatch &> /dev/null; then
    ON_CLUSTER=true
    BASE_DIR="$HOME/golomb"
else
    ON_CLUSTER=false
    BASE_DIR="${LOCAL_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi

# =============================================================================
# Default configuration (can be overridden via command line)
# =============================================================================
ACCOUNT="${SLURM_ACCOUNT:-}"
MEMORY="${MEMORY_PER_NODE:-8}"
DRY_RUN=false
SUBMIT_REMOTE=false

# Architecture (from config.sh)
ARCH="${ARCHITECTURE:-x64cpu}"
CONSTRAINT="${ARCH}"

# GPU configuration
USE_GPU="${ENABLE_GPU:-false}"
GPU_COUNT="${GPUS_PER_NODE:-0}"

# Sequential versions to test
SEQ_VERSIONS_TO_TEST="${SEQ_VERSIONS_TO_TEST:-1 2 3 4 5 6}"

# MPI versions to test
MPI_VERSIONS_TO_TEST="${MPI_VERSIONS_TO_TEST:-1 2 3}"

# =============================================================================
# Parse arguments
# =============================================================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --seq-versions N,N,...  Sequential versions to test (default: 1,2,3,4,5,6)"
    echo "  --mpi-versions N,N,...  MPI versions to test (default: 1,2,3)"
    echo "  --mpi-procs N,N,...     MPI process counts (default: 8,16,32,64,128)"
    echo "  --arch NAME             Architecture: x64cpu or armgpu (default: x64cpu)"
    echo "  --gpu                   Enable GPU (sets arch to armgpu)"
    echo "  --gpu-count N           Number of GPUs per node (default: 0, max: 4)"
    echo "  --dry-run               Generate scripts but don't submit"
    echo "  --submit                Submit from local machine via SSH"
    echo "  --help                  Show this help"
    echo ""
    echo "Romeo 2025 Architecture:"
    echo "  x64cpu : 44 AMD EPYC servers (128 cores, 2TB RAM each)"
    echo "  armgpu : 58 ARM Grace Hopper servers (288 cores + 4x H100 GPU each)"
    echo ""
    echo "Partitions (auto-selected based on time limit):"
    echo "  instant : <= 1 hour  (high priority)"
    echo "  short   : <= 24 hours"
    echo "  long    : <= 30 days"
    echo ""
    echo "Orders are automatically selected based on version performance:"
    echo "  v1: G5-G7    v2: G6-G8    v3: G7-G9"
    echo "  v4: G8-G11   v5: G10-G13  v6: G11-G14"
    echo "  MPI v1/v2: G9-G11   MPI v3: G10-G14"
    echo ""
    echo "Examples:"
    echo "  # On Romeo cluster (CPU):"
    echo "  ./run_benchmarks.sh"
    echo ""
    echo "  # From local machine:"
    echo "  ./run_benchmarks.sh --submit"
    echo ""
    echo "  # Test on ARM+GPU architecture:"
    echo "  ./run_benchmarks.sh --arch armgpu --gpu-count 2"
    echo ""
    echo "  # Test only optimized versions:"
    echo "  ./run_benchmarks.sh --seq-versions 5,6 --mpi-versions 3 --dry-run"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seq-versions)
            SEQ_VERSIONS_TO_TEST="${2//,/ }"
            shift 2
            ;;
        --mpi-versions)
            MPI_VERSIONS_TO_TEST="${2//,/ }"
            shift 2
            ;;
        --mpi-procs)
            MPI_PROCS="${2//,/ }"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            CONSTRAINT="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            ARCH="armgpu"
            CONSTRAINT="armgpu"
            shift
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            USE_GPU=true
            ARCH="armgpu"
            CONSTRAINT="armgpu"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --submit)
            SUBMIT_REMOTE=true
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

# =============================================================================
# Functions
# =============================================================================

generate_seq_job() {
    local order=$1
    local version=$2
    local threads=$3
    local time_limit="${TIME_LIMITS[$order]:-01:00:00}"

    # Auto-select partition based on time limit (Romeo 2025)
    local partition=$(select_partition "${time_limit}")
    local partition_line=""
    if [[ -n "$partition" ]]; then
        partition_line="#SBATCH --partition=${partition}"
    fi

    local account_line=""
    if [[ -n "$ACCOUNT" ]]; then
        account_line="#SBATCH --account=${ACCOUNT}"
    fi

    # GPU line (only for armgpu architecture)
    local gpu_line=""
    if [[ "$USE_GPU" == "true" && "$GPU_COUNT" -gt 0 ]]; then
        gpu_line="#SBATCH --gpus-per-node=${GPU_COUNT}"
    fi

    local job_file="${BASE_DIR}/jobs/seq_G${order}_v${version}_t${threads}.slurm"

    sed -e "s/%ORDER%/${order}/g" \
        -e "s/%VERSION%/${version}/g" \
        -e "s/%THREADS%/${threads}/g" \
        -e "s/%TIME_LIMIT%/${time_limit}/g" \
        -e "s/%MEMORY%/${MEMORY}/g" \
        -e "s/%CONSTRAINT%/${CONSTRAINT}/g" \
        -e "s/%PARTITION_LINE%/${partition_line}/g" \
        -e "s/%ACCOUNT_LINE%/${account_line}/g" \
        -e "s/%GPU_LINE%/${gpu_line}/g" \
        "${SCRIPT_DIR}/job_sequential.slurm" > "${job_file}"

    echo "${job_file}"
}

generate_mpi_job() {
    local order=$1
    local version=$2
    local procs=$3
    local time_limit="${TIME_LIMITS[$order]:-01:00:00}"

    # Calculate nodes needed based on architecture
    local cores_per_node=$(get_cores_per_node)
    local nodes=$(( (procs + cores_per_node - 1) / cores_per_node ))
    if [[ $nodes -lt 1 ]]; then nodes=1; fi

    # Auto-select partition based on time limit (Romeo 2025)
    local partition=$(select_partition "${time_limit}")
    local partition_line=""
    if [[ -n "$partition" ]]; then
        partition_line="#SBATCH --partition=${partition}"
    fi

    local account_line=""
    if [[ -n "$ACCOUNT" ]]; then
        account_line="#SBATCH --account=${ACCOUNT}"
    fi

    local job_file="${BASE_DIR}/jobs/mpi_G${order}_v${version}_p${procs}.slurm"

    sed -e "s/%ORDER%/${order}/g" \
        -e "s/%VERSION%/${version}/g" \
        -e "s/%PROCS%/${procs}/g" \
        -e "s/%NODES%/${nodes}/g" \
        -e "s/%TIME_LIMIT%/${time_limit}/g" \
        -e "s/%MEMORY%/${MEMORY}/g" \
        -e "s/%CONSTRAINT%/${CONSTRAINT}/g" \
        -e "s/%PARTITION_LINE%/${partition_line}/g" \
        -e "s/%ACCOUNT_LINE%/${account_line}/g" \
        "${SCRIPT_DIR}/job_mpi.slurm" > "${job_file}"

    echo "${job_file}"
}

submit_job() {
    local job_file=$1
    if $DRY_RUN; then
        echo "[DRY-RUN] Would submit: ${job_file}"
    else
        echo "Submitting: ${job_file}"
        sbatch "${job_file}"
    fi
}

# =============================================================================
# Main execution
# =============================================================================

main() {
    echo "=============================================="
    echo "  Golomb Ruler Solver - Benchmark Submission"
    echo "  Romeo 2025 - Architecture: ${ARCH}"
    echo "=============================================="
    echo ""
    echo "Configuration:"
    echo "  Architecture: ${ARCH}"
    echo "  Constraint: ${CONSTRAINT}"
    echo "  GPU enabled: ${USE_GPU} (count: ${GPU_COUNT})"
    echo "  Sequential versions: ${SEQ_VERSIONS_TO_TEST}"
    echo "  MPI versions: ${MPI_VERSIONS_TO_TEST}"
    echo "  MPI processes: ${MPI_PROCS[*]}"
    echo "  OpenMP threads: ${OMP_THREADS[*]}"
    echo "  Partition: auto-selected based on time"
    echo "  Dry run: ${DRY_RUN}"
    echo ""

    # Create directories
    mkdir -p "${BASE_DIR}/jobs"
    mkdir -p "${BASE_DIR}/results/romeo"

    local job_count=0

    # ==========================================================================
    # Generate and submit sequential jobs
    # ==========================================================================
    echo "=== Sequential Jobs ==="

    for version in ${SEQ_VERSIONS_TO_TEST}; do
        # Get orders for this version from config
        local orders="${SEQ_ORDERS[$version]}"

        if [[ -z "$orders" ]]; then
            echo "  [SKIP] v${version}: No orders configured"
            continue
        fi

        echo "  Version v${version}: orders ${orders}"

        for order in $orders; do
            if [[ $version -eq 2 || $version -eq 6 ]]; then
                # v2 and v6: OpenMP versions - test with different thread counts
                for threads in ${OMP_THREADS[*]}; do
                    job_file=$(generate_seq_job $order $version $threads)
                    submit_job "${job_file}"
                    : $((job_count++))
                done
            else
                # Other versions: single-threaded
                job_file=$(generate_seq_job $order $version 1)
                submit_job "${job_file}"
                : $((job_count++))
            fi
        done
    done

    # ==========================================================================
    # Generate and submit MPI jobs
    # ==========================================================================
    echo ""
    echo "=== MPI Jobs ==="

    for version in ${MPI_VERSIONS_TO_TEST}; do
        # Get orders for this MPI version from config
        local orders="${MPI_ORDERS[$version]}"

        if [[ -z "$orders" ]]; then
            echo "  [SKIP] MPI v${version}: No orders configured"
            continue
        fi

        echo "  MPI v${version}: orders ${orders}"

        for order in $orders; do
            for procs in ${MPI_PROCS[*]}; do
                job_file=$(generate_mpi_job $order $version $procs)
                submit_job "${job_file}"
                : $((job_count++))
            done
        done
    done

    echo ""
    echo "=============================================="
    echo "Total jobs: ${job_count}"
    if $DRY_RUN; then
        echo "Dry run - no jobs submitted"
        echo "Review generated scripts in: ${BASE_DIR}/jobs/"
    else
        echo "Jobs submitted! Monitor with: squeue -u \$USER"
    fi
    echo "=============================================="
}

# Handle remote submission
if $SUBMIT_REMOTE && ! $ON_CLUSTER; then
    echo "Submitting jobs remotely via SSH..."
    if ! check_ssh; then
        exit 1
    fi

    # Copy this script to Romeo and execute
    remote_copy "${SCRIPT_DIR}/" "${REMOTE_BASE_DIR}/scripts/romeo/"

    # Build command with current options
    REMOTE_CMD="cd ${REMOTE_BASE_DIR} && bash scripts/romeo/run_benchmarks.sh"
    [[ "${SEQ_VERSIONS_TO_TEST}" != "1 2 3 4 5 6" ]] && REMOTE_CMD+=" --seq-versions '${SEQ_VERSIONS_TO_TEST// /,}'"
    [[ "${MPI_VERSIONS_TO_TEST}" != "1 2 3" ]] && REMOTE_CMD+=" --mpi-versions '${MPI_VERSIONS_TO_TEST// /,}'"
    [[ "${DRY_RUN}" == "true" ]] && REMOTE_CMD+=" --dry-run"

    remote_exec "${REMOTE_CMD}"
else
    main
fi
