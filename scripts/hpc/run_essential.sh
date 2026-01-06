#!/bin/bash
# =============================================================================
# Run Essential Benchmarks for Meaningful HPC Graphs
# =============================================================================
# 42 jobs focused on G11, G12, G13 - orders where HPC makes sense
# Avoids G5-G9 where MPI overhead dominates
# =============================================================================

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Results directories
SEQ_DIR="${REMOTE_RESULTS_DIR}/sequential"
PAR_DIR="${REMOTE_RESULTS_DIR}/parallel"
OMP_DIR="${REMOTE_RESULTS_DIR}/openmp"

# SLURM account
ACCOUNT="${SLURM_ACCOUNT:-r250127}"

# =============================================================================
# Helper: Submit a sequential job
# =============================================================================
submit_seq_job() {
    local version=$1
    local order=$2
    local threads=${3:-1}
    local time_limit="${TIME_LIMITS[$order]:-02:00:00}"
    local partition=$(select_partition "$time_limit")

    local job_name="seq_G${order}_v${version}_t${threads}"
    local csv_file="${SEQ_DIR}/${job_name}.csv"

    echo "Submitting: $job_name (partition: $partition, time: $time_limit)"

    sbatch --parsable \
        --job-name="$job_name" \
        --account="$ACCOUNT" \
        --partition="$partition" \
        --constraint="x64cpu" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$threads \
        --time="$time_limit" \
        --mem=8G \
        --output="${REMOTE_RESULTS_DIR}/logs/${job_name}_%j.out" \
        --error="${REMOTE_RESULTS_DIR}/logs/${job_name}_%j.err" \
        --wrap="
            source ~/.bashrc
            romeo_load_x64cpu_env 2>/dev/null || true
            cd ${REMOTE_BASE_DIR}
            export OMP_NUM_THREADS=$threads
            export OMP_PLACES=cores
            export OMP_PROC_BIND=close
            ./build/golomb_v${version} ${order} --threads $threads --csv ${csv_file}
        "
}

# =============================================================================
# Helper: Submit an MPI job
# =============================================================================
submit_mpi_job() {
    local version=$1
    local order=$2
    local procs=$3
    local time_limit="${TIME_LIMITS[$order]:-02:00:00}"
    local partition=$(select_partition "$time_limit")

    local job_name="mpi_G${order}_v${version}_p${procs}"
    local csv_file="${PAR_DIR}/${job_name}.csv"

    # Calculate nodes needed (max 128 cores/node on x64cpu)
    local nodes=$(( (procs + 127) / 128 ))
    [[ $nodes -lt 1 ]] && nodes=1

    echo "Submitting: $job_name (partition: $partition, nodes: $nodes, procs: $procs)"

    sbatch --parsable \
        --job-name="$job_name" \
        --account="$ACCOUNT" \
        --partition="$partition" \
        --constraint="x64cpu" \
        --nodes=$nodes \
        --ntasks=$procs \
        --cpus-per-task=1 \
        --time="$time_limit" \
        --mem-per-cpu=4G \
        --output="${REMOTE_RESULTS_DIR}/logs/${job_name}_%j.out" \
        --error="${REMOTE_RESULTS_DIR}/logs/${job_name}_%j.err" \
        --wrap="
            source ~/.bashrc
            romeo_load_x64cpu_env 2>/dev/null || true
            cd ${REMOTE_BASE_DIR}
            export OMP_NUM_THREADS=1
            export OMP_PLACES=cores
            export OMP_PROC_BIND=close
            mpirun -np $procs ./build/golomb_v${version} ${order} --csv ${csv_file}
        "
}

# =============================================================================
# Helper: Submit an OpenMP job
# =============================================================================
submit_omp_job() {
    local version=$1
    local order=$2
    local threads=$3
    local time_limit="${TIME_LIMITS[$order]:-02:00:00}"
    local partition=$(select_partition "$time_limit")

    local job_name="omp_G${order}_v${version}_t${threads}"
    local csv_file="${OMP_DIR}/${job_name}.csv"

    echo "Submitting: $job_name (partition: $partition, threads: $threads)"

    sbatch --parsable \
        --job-name="$job_name" \
        --account="$ACCOUNT" \
        --partition="$partition" \
        --constraint="x64cpu" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$threads \
        --time="$time_limit" \
        --mem=16G \
        --output="${REMOTE_RESULTS_DIR}/logs/${job_name}_%j.out" \
        --error="${REMOTE_RESULTS_DIR}/logs/${job_name}_%j.err" \
        --wrap="
            source ~/.bashrc
            romeo_load_x64cpu_env 2>/dev/null || true
            cd ${REMOTE_BASE_DIR}
            export OMP_NUM_THREADS=$threads
            export OMP_PLACES=cores
            export OMP_PROC_BIND=close
            ./build/golomb_v${version} ${order} --threads $threads --csv ${csv_file}
        "
}

# =============================================================================
# Main
# =============================================================================

echo "=============================================="
echo " Essential HPC Benchmarks (42 jobs)"
echo " Focus: G11, G12, G13 for meaningful scaling"
echo "=============================================="
echo ""

# Create directories
mkdir -p "${SEQ_DIR}" "${PAR_DIR}" "${OMP_DIR}" "${REMOTE_RESULTS_DIR}/logs"

JOB_COUNT=0

# =============================================================================
# 1. Sequential Baselines (8 jobs) - CRITICAL for T_seq
# =============================================================================
echo ">>> Submitting Sequential Baselines (T_seq references)..."

for order in 10 11 12 13; do
    # v1 baseline
    submit_seq_job 1 $order 1
    ((JOB_COUNT++))

    # v2 backup baseline
    submit_seq_job 2 $order 1
    ((JOB_COUNT++))
done

echo ""

# =============================================================================
# 2. MPI v3 Strong Scaling - G11, G12, G13 (15 jobs)
# =============================================================================
echo ">>> Submitting MPI v3 Strong Scaling..."

for order in 11 12 13; do
    for procs in 2 4 8 16 32; do
        submit_mpi_job 3 $order $procs
        ((JOB_COUNT++))
    done
done

echo ""

# =============================================================================
# 3. MPI v4 Strong Scaling - G11, G12, G13 (15 jobs)
# =============================================================================
echo ">>> Submitting MPI v4 Strong Scaling..."

for order in 11 12 13; do
    for procs in 2 4 8 16 32; do
        submit_mpi_job 4 $order $procs
        ((JOB_COUNT++))
    done
done

echo ""

# =============================================================================
# 4. OpenMP v6 Scaling - G11 (4 jobs)
# =============================================================================
echo ">>> Submitting OpenMP v6 Scaling (for MPI vs OpenMP comparison)..."

for threads in 1 4 8 16; do
    submit_omp_job 6 11 $threads
    ((JOB_COUNT++))
done

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo " Submitted $JOB_COUNT jobs"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel -u \$USER"
echo ""
echo "Results will be in:"
echo "  Sequential: ${SEQ_DIR}/"
echo "  Parallel:   ${PAR_DIR}/"
echo "  OpenMP:     ${OMP_DIR}/"
