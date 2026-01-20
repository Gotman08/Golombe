#!/bin/bash
# =============================================================================
# Submit ONLY Missing Benchmark Jobs to SLURM on Romeo
# Based on analysis: 34 jobs missing out of 85 total (62% complete)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load config
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

# Configuration
ARCH="${ARCHITECTURE:-x64cpu}"
CONSTRAINT="${ARCH}"
MEMORY="${MEMORY_PER_NODE:-8}"
ACCOUNT="${SLURM_ACCOUNT:-r250127}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --submit)
            # Execute remotely via SSH
            echo "Submitting to Romeo via SSH..."
            ssh romeo1.univ-reims.fr "cd ~/golomb && bash scripts/hpc/run_missing.sh"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--submit]"
            echo "  --dry-run : Generate job files but don't submit"
            echo "  --submit  : Execute this script on Romeo via SSH"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================

select_partition() {
    local time_limit=$1
    local hours=$(echo "$time_limit" | cut -d: -f1)
    if [[ $hours -le 1 ]]; then
        echo "instant"
    elif [[ $hours -le 24 ]]; then
        echo "short"
    else
        echo "long"
    fi
}

get_cores_per_node() {
    case "$ARCH" in
        armgpu) echo 288 ;;
        *)      echo 128 ;;
    esac
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
# Job generation functions
# =============================================================================

generate_seq_job() {
    local order=$1
    local version=$2
    local threads=$3
    local time_limit="${TIME_LIMITS[$order]:-01:00:00}"
    local partition=$(select_partition "${time_limit}")

    local job_file="${BASE_DIR}/jobs/seq_G${order}_v${version}_t${threads}.slurm"

    cat > "${job_file}" << EOF
#!/bin/bash
#SBATCH --job-name=golomb_seq_G${order}_v${version}_t${threads}
#SBATCH --output=${BASE_DIR}/results/romeo/seq_G${order}_v${version}_t${threads}.out
#SBATCH --error=${BASE_DIR}/results/romeo/seq_G${order}_v${version}_t${threads}.err
#SBATCH --time=${time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${threads}
#SBATCH --mem=${MEMORY}G
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --partition=${partition}
#SBATCH --account=${ACCOUNT}

cd ${BASE_DIR}
source setup_env.sh 2>/dev/null || true

export OMP_NUM_THREADS=${threads}

echo "=== Golomb Ruler Benchmark ==="
echo "Version: v${version}"
echo "Order: G${order}"
echo "Threads: ${threads}"
echo "Start: \$(date)"

./build/golomb_v${version} ${order} --threads ${threads} --csv results/romeo/seq_G${order}_v${version}_t${threads}.csv

echo "End: \$(date)"
EOF

    echo "${job_file}"
}

generate_mpi_job() {
    local order=$1
    local version=$2
    local procs=$3
    local time_limit="${TIME_LIMITS[$order]:-01:00:00}"
    local partition=$(select_partition "${time_limit}")

    local cores_per_node=$(get_cores_per_node)
    local nodes=$(( (procs + cores_per_node - 1) / cores_per_node ))
    [[ $nodes -lt 1 ]] && nodes=1

    local omp_threads=4  # Default OMP threads per MPI process

    local job_file="${BASE_DIR}/jobs/mpi_G${order}_v${version}_p${procs}.slurm"

    cat > "${job_file}" << EOF
#!/bin/bash
#SBATCH --job-name=golomb_mpi_G${order}_v${version}_p${procs}
#SBATCH --output=${BASE_DIR}/results/romeo/mpi_G${order}_v${version}_p${procs}.out
#SBATCH --error=${BASE_DIR}/results/romeo/mpi_G${order}_v${version}_p${procs}.err
#SBATCH --time=${time_limit}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${procs}
#SBATCH --cpus-per-task=${omp_threads}
#SBATCH --mem=${MEMORY}G
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --partition=${partition}
#SBATCH --account=${ACCOUNT}

cd ${BASE_DIR}
source setup_env.sh 2>/dev/null || true

export OMP_NUM_THREADS=${omp_threads}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== Golomb Ruler MPI Benchmark ==="
echo "Version: v${version}"
echo "Order: G${order}"
echo "MPI Procs: ${procs}"
echo "OMP Threads: ${omp_threads}"
echo "Nodes: ${nodes}"
echo "Start: \$(date)"

mpirun -np ${procs} ./build/golomb_v${version} ${order} --csv results/romeo/mpi_G${order}_v${version}_p${procs}.csv

echo "End: \$(date)"
EOF

    echo "${job_file}"
}

generate_trace_job() {
    local order=$1
    local version=$2
    local procs=$3
    local time_limit="${TIME_LIMITS[$order]:-01:00:00}"
    local partition=$(select_partition "${time_limit}")

    local cores_per_node=$(get_cores_per_node)
    local nodes=$(( (procs + cores_per_node - 1) / cores_per_node ))
    [[ $nodes -lt 1 ]] && nodes=1

    local omp_threads=4

    local job_file="${BASE_DIR}/jobs/trace_G${order}_v${version}_p${procs}.slurm"

    cat > "${job_file}" << EOF
#!/bin/bash
#SBATCH --job-name=golomb_trace_G${order}_v${version}
#SBATCH --output=${BASE_DIR}/results/romeo/trace_G${order}_v${version}_p${procs}.out
#SBATCH --error=${BASE_DIR}/results/romeo/trace_G${order}_v${version}_p${procs}.err
#SBATCH --time=${time_limit}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${procs}
#SBATCH --cpus-per-task=${omp_threads}
#SBATCH --mem=${MEMORY}G
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --partition=${partition}
#SBATCH --account=${ACCOUNT}

cd ${BASE_DIR}
source setup_env.sh 2>/dev/null || true

export OMP_NUM_THREADS=${omp_threads}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=== Golomb Ruler MPI Trace ==="
echo "Version: v${version}"
echo "Order: G${order}"
echo "MPI Procs: ${procs}"
echo "Start: \$(date)"

mpirun -np ${procs} ./build/golomb_v${version} ${order} --trace results/romeo/mpi_trace_v${version}_G${order}.csv

echo "End: \$(date)"
EOF

    echo "${job_file}"
}

# =============================================================================
# Main - Submit only missing jobs
# =============================================================================

echo "=============================================="
echo "  Golomb Ruler - Submit Missing Benchmarks"
echo "  Romeo 2025 - Architecture: ${ARCH}"
echo "=============================================="
echo ""
echo "Missing benchmarks to submit: 34 total"
echo "  - 1 sequential (v2 G12 t=1)"
echo "  - 7 parallel v3 (G8 redo, G9 incomplete)"
echo "  - 24 parallel v4 (all missing)"
echo "  - 2 MPI traces"
echo ""
echo "Dry run: ${DRY_RUN}"
echo ""

# Create directories
mkdir -p "${BASE_DIR}/jobs"
mkdir -p "${BASE_DIR}/results/romeo"

job_count=0

# ==========================================================================
# 1. Sequential missing: v2 G12 t=1
# ==========================================================================
echo "=== Sequential Missing (1 job) ==="

job_file=$(generate_seq_job 12 2 1)
submit_job "${job_file}"
: $((job_count++))

# ==========================================================================
# 2. V3 missing: G8 (all failed) + G9 (incomplete)
# ==========================================================================
echo ""
echo "=== V3 Missing (7 jobs) ==="

# G8 - all 4 procs failed
for procs in 4 8 16 32; do
    job_file=$(generate_mpi_job 8 3 $procs)
    submit_job "${job_file}"
    : $((job_count++))
done

# G9 - missing p=8,16,32
for procs in 8 16 32; do
    job_file=$(generate_mpi_job 9 3 $procs)
    submit_job "${job_file}"
    : $((job_count++))
done

# ==========================================================================
# 3. V4 missing: ALL (G8-G13 with procs 4,8,16,32)
# ==========================================================================
echo ""
echo "=== V4 Missing (24 jobs) ==="

for order in 8 9 10 11 12 13; do
    for procs in 4 8 16 32; do
        job_file=$(generate_mpi_job $order 4 $procs)
        submit_job "${job_file}"
        : $((job_count++))
    done
done

# ==========================================================================
# 4. MPI Traces missing: v3 G10 and v4 G10
# ==========================================================================
echo ""
echo "=== MPI Traces (2 jobs) ==="

job_file=$(generate_trace_job 10 3 8)
submit_job "${job_file}"
: $((job_count++))

job_file=$(generate_trace_job 10 4 8)
submit_job "${job_file}"
: $((job_count++))

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "=============================================="
echo "  Submitted ${job_count} jobs"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in: ${BASE_DIR}/results/romeo/"
