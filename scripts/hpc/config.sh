#!/bin/bash
# =============================================================================
# Configuration for Romeo HPC Cluster (Updated for Romeo 2025 / Spack 1.0.1)
# =============================================================================

# SSH Configuration
# Romeo has 4 login nodes: romeo1, romeo2, romeo3, romeo4 (load balanced)
ROMEO_USER="${ROMEO_USER:-nimarano}"
ROMEO_HOST="${ROMEO_HOST:-romeo1.univ-reims.fr}"
ROMEO_SSH="${ROMEO_USER}@${ROMEO_HOST}"

# Remote paths (use ~ for rsync compatibility, $HOME for SSH commands)
REMOTE_BASE_DIR="~/golomb"
REMOTE_BUILD_DIR="~/golomb/build"
REMOTE_RESULTS_DIR="~/golomb/results/romeo"
REMOTE_SCRATCH_DIR="/scratch_p/\$USER"         # Fast scratch storage for jobs

# Local paths
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_RESULTS_DIR="${LOCAL_PROJECT_DIR}/results/romeo"

# =============================================================================
# Architecture Configuration (Romeo 2025)
# =============================================================================
# x64cpu : 44 AMD EPYC servers, 128 cores/server, 2 TB RAM/server
# armgpu : 58 ARM Grace Hopper servers, 288 ARM cores + 4x H100 GPU/server
ARCHITECTURE="${ARCHITECTURE:-x64cpu}"         # x64cpu or armgpu

# GPU Configuration (only for armgpu architecture)
ENABLE_GPU="${ENABLE_GPU:-false}"
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"            # 0-4 GPUs per node (H100)

# =============================================================================
# SLURM Configuration (Romeo 2025)
# =============================================================================
# Valid partitions: instant (<=1h), short (<=24h), long (<=30 days)
# If not specified, SLURM auto-selects based on --time
SLURM_PARTITION="${SLURM_PARTITION:-}"         # Leave empty for auto-selection
SLURM_ACCOUNT="${SLURM_ACCOUNT:-r250127}"      # Account/project (from sacctmgr)

# Compiler variants (for optimized builds)
# x64cpu: gcc or aocc (AMD Optimizing C/C++ Compiler)
# armgpu: gcc or nvhpc (NVIDIA HPC SDK)
COMPILER_X64="${COMPILER_X64:-gcc}"
COMPILER_ARM="${COMPILER_ARM:-nvhpc}"

# =============================================================================
# Test configuration - Orders adapted per version performance
# =============================================================================

# v1: Sequential (single-threaded)
# v2: OpenMP (multi-threaded)
# v3: Hybrid MPI+OpenMP (master/worker)
# v4: Pure Hypercube MPI+OpenMP (all ranks equal)
# v5: Pure MPI (no OpenMP, hypercube topology)

# Sequential versions: orders adapted to each version's performance
declare -A SEQ_ORDERS
SEQ_ORDERS[1]="8 9 10 11"           # v1 sequential: G8-G11 (baseline)
SEQ_ORDERS[2]="8 9 10 11 12 13 14"  # v2 OpenMP: G8-G14

# MPI versions: orders adapted to parallel scaling
declare -A MPI_ORDERS
MPI_ORDERS[3]="8 9 10 11 12 13"     # v3 hybrid MPI+OpenMP: G8-G13
MPI_ORDERS[4]="8 9 10 11 12 13"     # v4 hypercube MPI+OpenMP: G8-G13
MPI_ORDERS[5]="8 9 10 11 12 13"     # v5 pure MPI: G8-G13

# All versions support CSV output
CSV_SEQ_VERSIONS=(1 2)
CSV_MPI_VERSIONS=(3 4 5)

# Process and thread counts
MPI_PROCS=(4 8 16 32)                           # Number of MPI processes for v3/v4
MPI_PROCS_V5=(8 16 32 64 128 256)               # Higher process counts for v5 pure MPI
OMP_THREADS=(1 8 16 32 64 128)                  # OpenMP thread counts for v2-v4

# Time limits per order (format: HH:MM:SS)
declare -A TIME_LIMITS
TIME_LIMITS[5]="00:05:00"
TIME_LIMITS[6]="00:05:00"
TIME_LIMITS[7]="00:10:00"
TIME_LIMITS[8]="00:15:00"
TIME_LIMITS[9]="00:30:00"
TIME_LIMITS[10]="01:00:00"
TIME_LIMITS[11]="02:00:00"
TIME_LIMITS[12]="04:00:00"
TIME_LIMITS[13]="08:00:00"
TIME_LIMITS[14]="16:00:00"

# Memory per node (in GB)
MEMORY_PER_NODE=8

# Cores per node (for calculating node count)
CORES_PER_NODE_X64=128    # AMD EPYC nodes
CORES_PER_NODE_ARM=288    # ARM Grace nodes

# =============================================================================
# Helper functions
# =============================================================================

log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[1;32m[OK]\033[0m $1"
}

log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

# Check SSH connection
check_ssh() {
    log_info "Testing SSH connection to ${ROMEO_SSH}..."
    if ssh -o ConnectTimeout=10 -o BatchMode=yes "${ROMEO_SSH}" "echo 'SSH OK'" 2>/dev/null; then
        log_success "SSH connection successful"
        return 0
    else
        log_error "Cannot connect to ${ROMEO_SSH}"
        log_info "Make sure you have:"
        log_info "  1. Set ROMEO_USER environment variable or edited config.sh"
        log_info "  2. Configured SSH keys for passwordless login"
        return 1
    fi
}

# Execute command on Romeo
remote_exec() {
    ssh "${ROMEO_SSH}" "$@"
}

# Copy files to Romeo
remote_copy() {
    rsync -avz --progress "$1" "${ROMEO_SSH}:$2"
}

# Copy files from Romeo
remote_fetch() {
    rsync -avz --progress "${ROMEO_SSH}:$1" "$2"
}

# =============================================================================
# Romeo 2025 specific functions
# =============================================================================

# Select partition based on time limit (Romeo 2025 partitions)
# instant: <=1h (high priority), short: <=24h, long: <=30 days
select_partition() {
    local time_limit="$1"
    local hours=$(echo "$time_limit" | cut -d: -f1)

    # Remove leading zeros for comparison
    hours=$((10#$hours))

    if [[ $hours -le 1 ]]; then
        echo "instant"
    elif [[ $hours -le 24 ]]; then
        echo "short"
    else
        echo "long"
    fi
}

# Get environment load command based on architecture
get_env_command() {
    case "${ARCHITECTURE}" in
        x64cpu)
            echo "romeo_load_x64cpu_env"
            ;;
        armgpu)
            echo "romeo_load_armgpu_env"
            ;;
        *)
            log_error "Unknown architecture: ${ARCHITECTURE}"
            echo "romeo_load_x64cpu_env"
            ;;
    esac
}

# Get cores per node based on architecture
get_cores_per_node() {
    case "${ARCHITECTURE}" in
        x64cpu)
            echo "${CORES_PER_NODE_X64}"
            ;;
        armgpu)
            echo "${CORES_PER_NODE_ARM}"
            ;;
        *)
            echo "128"
            ;;
    esac
}

# Get constraint for SLURM based on architecture
get_slurm_constraint() {
    echo "${ARCHITECTURE}"
}

# Check storage quotas on Romeo
check_quotas() {
    log_info "Checking storage quotas on Romeo..."
    remote_exec "mmlsquota --block-size auto gpfs 2>/dev/null || echo 'Quota command not available'"
}
