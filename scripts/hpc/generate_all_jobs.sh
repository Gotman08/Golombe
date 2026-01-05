#!/bin/bash
# =============================================================================
# Generate All SLURM Jobs for Comprehensive Benchmarks
# =============================================================================

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Output directory for job files
JOBS_DIR="${LOCAL_PROJECT_DIR}/jobs"
mkdir -p "${JOBS_DIR}"

# Counter for generated jobs
JOB_COUNT=0

# =============================================================================
# Helper Functions
# =============================================================================

get_time_limit() {
    local order=$1
    echo "${TIME_LIMITS[$order]:-04:00:00}"
}

# =============================================================================
# Generate Sequential Job
# =============================================================================

generate_seq_job() {
    local version=$1
    local order=$2
    local threads=$3
    local cores=$4
    local ram_gb=$5

    local time_limit=$(get_time_limit $order)
    local nodes=1

    local job_name="seq_G${order}_v${version}_c${cores}_t${threads}"
    local job_file="${JOBS_DIR}/${job_name}.slurm"

    cat > "$job_file" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --time=${time_limit}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cores}
#SBATCH --mem=${ram_gb}G
#SBATCH --constraint=${ARCHITECTURE}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --export=ALL

# Load Spack environment for OpenMPI
source ~/.bashrc
romeo_load_x64cpu_env 2>/dev/null || true
spack load openmpi@4.1.7%gcc@11.4.1 2>/dev/null || true

# OpenMP configuration
export OMP_NUM_THREADS=${threads}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G${order} v${version} - ${cores} cores, ${threads} threads ==="
cd ~/golomb
./build/golomb_v${version} ${order} --csv results/romeo/${job_name}.csv
SLURM_EOF

    chmod +x "$job_file"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# =============================================================================
# Generate MPI Job (v3, v4)
# =============================================================================

generate_mpi_job() {
    local version=$1
    local order=$2
    local mpi_procs=$3
    local omp_threads=$4
    local nodes=$5

    local time_limit=$(get_time_limit $order)
    local tasks_per_node=$((mpi_procs / nodes))
    local mem_per_node=$((omp_threads * tasks_per_node * 4))  # 4GB per thread

    local job_name="mpi_G${order}_v${version}_n${nodes}_p${mpi_procs}_t${omp_threads}"
    local job_file="${JOBS_DIR}/${job_name}.slurm"

    cat > "$job_file" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --time=${time_limit}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${mpi_procs}
#SBATCH --ntasks-per-node=${tasks_per_node}
#SBATCH --cpus-per-task=${omp_threads}
#SBATCH --mem=${mem_per_node}G
#SBATCH --constraint=${ARCHITECTURE}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --export=ALL

# Load Spack environment for OpenMPI
source ~/.bashrc
romeo_load_x64cpu_env 2>/dev/null || true
spack load openmpi@4.1.7%gcc@11.4.1 2>/dev/null || true

# OpenMP configuration
export OMP_NUM_THREADS=${omp_threads}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=64M

# Run benchmark
echo "=== G${order} v${version} - ${nodes} nodes, ${mpi_procs} procs, ${omp_threads} threads ==="
cd ~/golomb
srun ./build/golomb_v${version} ${order} --threads ${omp_threads} --csv results/romeo/${job_name}.csv
SLURM_EOF

    chmod +x "$job_file"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# =============================================================================
# Generate v5 Pure MPI Job
# =============================================================================

generate_v5_job() {
    local order=$1
    local mpi_procs=$2
    local nodes=$3

    local time_limit=$(get_time_limit $order)
    local tasks_per_node=$((mpi_procs / nodes))
    local mem_per_node=$((tasks_per_node * 2))  # 2GB per MPI process

    local job_name="mpi_G${order}_v5_n${nodes}_p${mpi_procs}"
    local job_file="${JOBS_DIR}/${job_name}.slurm"

    cat > "$job_file" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --time=${time_limit}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${mpi_procs}
#SBATCH --ntasks-per-node=${tasks_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --mem=${mem_per_node}G
#SBATCH --constraint=${ARCHITECTURE}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --export=ALL

# Load Spack environment for OpenMPI
source ~/.bashrc
romeo_load_x64cpu_env 2>/dev/null || true
spack load openmpi@4.1.7%gcc@11.4.1 2>/dev/null || true

# No OpenMP for v5
export OMP_NUM_THREADS=1

# Run benchmark
echo "=== G${order} v5 Pure MPI - ${nodes} nodes, ${mpi_procs} procs ==="
cd ~/golomb
srun ./build/golomb_v5 ${order} --csv results/romeo/${job_name}.csv
SLURM_EOF

    chmod +x "$job_file"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# =============================================================================
# Main Generation Logic
# =============================================================================

echo "=== Generating Comprehensive Benchmark Jobs ==="
echo "Output directory: ${JOBS_DIR}"
echo ""

# -----------------------------------------------------------------------------
# v1: Sequential Baseline (G8-G11)
# -----------------------------------------------------------------------------
echo "--- v1: Sequential Baseline ---"
for order in 8 9 10 11; do
    generate_seq_job 1 $order 1 1 4
    echo "  seq_G${order}_v1_c1_t1.slurm"
done
echo ""

# -----------------------------------------------------------------------------
# v2: OpenMP Scaling
# -----------------------------------------------------------------------------
echo "--- v2: OpenMP Scaling ---"

# 8 cores, various threads (G8-G12)
for order in 8 9 10 11 12; do
    for threads in 1 2 4 8; do
        generate_seq_job 2 $order $threads 8 32
        echo "  seq_G${order}_v2_c8_t${threads}.slurm"
    done
done

# 32 cores, various threads (G8-G12)
for order in 8 9 10 11 12; do
    for threads in 1 4 8 16 32; do
        generate_seq_job 2 $order $threads 32 128
        echo "  seq_G${order}_v2_c32_t${threads}.slurm"
    done
done

# 64 cores, various threads (G8-G13)
for order in 8 9 10 11 12 13; do
    for threads in 1 8 16 32 64; do
        generate_seq_job 2 $order $threads 64 256
        echo "  seq_G${order}_v2_c64_t${threads}.slurm"
    done
done

# 128 cores, various threads (G10-G14)
for order in 10 11 12 13 14; do
    for threads in 1 16 32 64 128; do
        generate_seq_job 2 $order $threads 128 512
        echo "  seq_G${order}_v2_c128_t${threads}.slurm"
    done
done
echo ""

# -----------------------------------------------------------------------------
# v3: Hybrid MPI+OpenMP (Master/Worker)
# -----------------------------------------------------------------------------
echo "--- v3: Hybrid MPI+OpenMP ---"

# 1 node, 4 procs x 8 threads (G8-G12)
for order in 8 9 10 11 12; do
    generate_mpi_job 3 $order 4 8 1
    echo "  mpi_G${order}_v3_n1_p4_t8.slurm"
done

# 1 node, 8 procs x 4 threads (G8-G12)
for order in 8 9 10 11 12; do
    generate_mpi_job 3 $order 8 4 1
    echo "  mpi_G${order}_v3_n1_p8_t4.slurm"
done

# 2 nodes, 8 procs x 16 threads (G8-G13)
for order in 8 9 10 11 12 13; do
    generate_mpi_job 3 $order 8 16 2
    echo "  mpi_G${order}_v3_n2_p8_t16.slurm"
done

# 4 nodes, 16 procs x 16 threads (G10-G13)
for order in 10 11 12 13; do
    generate_mpi_job 3 $order 16 16 4
    echo "  mpi_G${order}_v3_n4_p16_t16.slurm"
done

# 8 nodes, 32 procs x 16 threads (G11-G14)
for order in 11 12 13 14; do
    generate_mpi_job 3 $order 32 16 8
    echo "  mpi_G${order}_v3_n8_p32_t16.slurm"
done
echo ""

# -----------------------------------------------------------------------------
# v4: Hypercube MPI+OpenMP
# -----------------------------------------------------------------------------
echo "--- v4: Hypercube MPI+OpenMP ---"

# 1 node, 4 procs x 8 threads (G8-G12)
for order in 8 9 10 11 12; do
    generate_mpi_job 4 $order 4 8 1
    echo "  mpi_G${order}_v4_n1_p4_t8.slurm"
done

# 1 node, 8 procs x 4 threads (G8-G12)
for order in 8 9 10 11 12; do
    generate_mpi_job 4 $order 8 4 1
    echo "  mpi_G${order}_v4_n1_p8_t4.slurm"
done

# 2 nodes, 16 procs x 8 threads (G8-G13)
for order in 8 9 10 11 12 13; do
    generate_mpi_job 4 $order 16 8 2
    echo "  mpi_G${order}_v4_n2_p16_t8.slurm"
done

# 4 nodes, 32 procs x 8 threads (G10-G13)
for order in 10 11 12 13; do
    generate_mpi_job 4 $order 32 8 4
    echo "  mpi_G${order}_v4_n4_p32_t8.slurm"
done

# 8 nodes, 64 procs x 8 threads (G11-G14)
for order in 11 12 13 14; do
    generate_mpi_job 4 $order 64 8 8
    echo "  mpi_G${order}_v4_n8_p64_t8.slurm"
done
echo ""

# -----------------------------------------------------------------------------
# v5: Pure MPI (no OpenMP)
# -----------------------------------------------------------------------------
echo "--- v5: Pure MPI ---"

# 1 node, 8 procs (G8-G11)
for order in 8 9 10 11; do
    generate_v5_job $order 8 1
    echo "  mpi_G${order}_v5_n1_p8.slurm"
done

# 1 node, 16 procs (G8-G11)
for order in 8 9 10 11; do
    generate_v5_job $order 16 1
    echo "  mpi_G${order}_v5_n1_p16.slurm"
done

# 1 node, 32 procs (G8-G12)
for order in 8 9 10 11 12; do
    generate_v5_job $order 32 1
    echo "  mpi_G${order}_v5_n1_p32.slurm"
done

# 2 nodes, 64 procs (G8-G12)
for order in 8 9 10 11 12; do
    generate_v5_job $order 64 2
    echo "  mpi_G${order}_v5_n2_p64.slurm"
done

# 4 nodes, 128 procs (G9-G13)
for order in 9 10 11 12 13; do
    generate_v5_job $order 128 4
    echo "  mpi_G${order}_v5_n4_p128.slurm"
done

# 8 nodes, 256 procs (G10-G13)
for order in 10 11 12 13; do
    generate_v5_job $order 256 8
    echo "  mpi_G${order}_v5_n8_p256.slurm"
done
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=== Generation Complete ==="
echo "Total jobs generated: ${JOB_COUNT}"
echo ""
echo "Job breakdown:"
v1_count=$(ls -1 ${JOBS_DIR}/seq_*_v1_*.slurm 2>/dev/null | wc -l)
v2_count=$(ls -1 ${JOBS_DIR}/seq_*_v2_*.slurm 2>/dev/null | wc -l)
v3_count=$(ls -1 ${JOBS_DIR}/mpi_*_v3_*.slurm 2>/dev/null | wc -l)
v4_count=$(ls -1 ${JOBS_DIR}/mpi_*_v4_*.slurm 2>/dev/null | wc -l)
v5_count=$(ls -1 ${JOBS_DIR}/mpi_*_v5_*.slurm 2>/dev/null | wc -l)
echo "  v1 (Sequential):     $v1_count jobs"
echo "  v2 (OpenMP):         $v2_count jobs"
echo "  v3 (Hybrid MPI+OMP): $v3_count jobs"
echo "  v4 (Hypercube):      $v4_count jobs"
echo "  v5 (Pure MPI):       $v5_count jobs"
echo ""
echo "Jobs are in: ${JOBS_DIR}/"
echo ""
echo "To submit all jobs on Romeo:"
echo "  1. make romeo-deploy"
echo "  2. ssh romeo"
echo "  3. cd golomb && for f in jobs/*.slurm; do sbatch \$f; done"
