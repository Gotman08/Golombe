#!/bin/bash
#SBATCH --job-name=golomb_compare
#SBATCH --account=r250127
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --constraint=x64cpu
#SBATCH --output=results/comparison_%j.log
#SBATCH --error=results/comparison_%j.err

# Comparaison de toutes les versions du solveur Golomb
# Sequential: v1-v6
# Parallel: mpi_v1-v5

# Charger l'environnement Romeo
source /etc/profile.d/spack-env.sh
env_x64cpu
module load openmpi/gnu/4.1.7

cd $HOME/golomb
mkdir -p results/comparison

echo "=========================================="
echo "    COMPARAISON TOUTES VERSIONS GOLOMB   "
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo ""

OUTPUT_FILE="results/comparison/benchmark_$(date +%Y%m%d_%H%M%S).csv"
echo "version,order,processes,threads,time_ms,nodes_explored,optimal,length" > $OUTPUT_FILE

# Function to run and log
run_benchmark() {
    local version=$1
    local order=$2
    local procs=$3
    local threads=$4
    local timeout_sec=$5

    echo "Testing $version on G$order with $procs procs, $threads threads..."

    if [[ $version == mpi_* ]]; then
        # MPI version
        result=$(timeout ${timeout_sec}s mpirun -np $procs ./build/golomb_$version $order 2>&1)
    elif [[ $version == golomb_v6 ]]; then
        # OpenMP version
        result=$(OMP_NUM_THREADS=$threads timeout ${timeout_sec}s ./build/$version $order 2>&1)
    else
        # Sequential version
        result=$(timeout ${timeout_sec}s ./build/$version $order 2>&1)
    fi

    # Parse results
    time_ms=$(echo "$result" | grep -oP 'Time: \K[\d.]+(?= ms)')
    nodes=$(echo "$result" | grep -oP 'Nodes explored: \K\d+')
    optimal=$(echo "$result" | grep -q "OPTIMAL" && echo "yes" || echo "no")
    length=$(echo "$result" | grep -oP 'Length: \K\d+')

    if [[ -n "$time_ms" ]]; then
        echo "$version,$order,$procs,$threads,$time_ms,$nodes,$optimal,$length" >> $OUTPUT_FILE
        echo "  -> $time_ms ms, length=$length, optimal=$optimal"
    else
        echo "$version,$order,$procs,$threads,TIMEOUT,,,," >> $OUTPUT_FILE
        echo "  -> TIMEOUT or ERROR"
    fi
}

echo ""
echo "=== 1. VERSIONS SEQUENTIELLES (G7-G10) ==="
echo ""

# Sequential versions avec timeout court
for order in 7 8 9 10; do
    echo "--- G$order ---"

    # v1 et v2 seulement pour petits ordres
    if [[ $order -le 7 ]]; then
        run_benchmark "golomb_v1" $order 1 1 60
        run_benchmark "golomb_v2" $order 1 1 60
    fi

    # v3, v4, v5 pour tous
    run_benchmark "golomb_v3" $order 1 1 120
    run_benchmark "golomb_v4" $order 1 1 120
    run_benchmark "golomb_v5" $order 1 1 120

    # v6 avec différents threads
    for threads in 1 2 4 8; do
        run_benchmark "golomb_v6" $order 1 $threads 120
    done
    echo ""
done

echo ""
echo "=== 2. VERSIONS MPI (G9-G11) ==="
echo ""

for order in 9 10 11; do
    echo "--- G$order ---"

    for procs in 2 4 8 16; do
        # Skip if not enough tasks
        if [[ $procs -gt $SLURM_NTASKS ]]; then
            continue
        fi

        run_benchmark "mpi_v1" $order $procs 1 300
        run_benchmark "mpi_v2" $order $procs 1 300
        run_benchmark "mpi_v3" $order $procs 1 300
        run_benchmark "mpi_v5" $order $procs 1 300
    done
    echo ""
done

echo ""
echo "=== 3. VERSION HYBRIDE MPI+OpenMP (G10-G11) ==="
echo ""

for order in 10 11; do
    echo "--- G$order ---"

    # 2 ranks x 8 threads = 16 cores
    OMP_NUM_THREADS=8 run_benchmark "mpi_v4" $order 2 8 600

    # 4 ranks x 4 threads = 16 cores
    OMP_NUM_THREADS=4 run_benchmark "mpi_v4" $order 4 4 600

    # 8 ranks x 2 threads = 16 cores
    OMP_NUM_THREADS=2 run_benchmark "mpi_v4" $order 8 2 600
    echo ""
done

echo ""
echo "=== 4. G12 COMPARAISON (best configs) ==="
echo ""

echo "--- G12 ---"
run_benchmark "mpi_v3" 12 16 1 1800
run_benchmark "mpi_v5" 12 16 1 1800
OMP_NUM_THREADS=4 run_benchmark "mpi_v4" 12 4 4 1800

echo ""
echo "=========================================="
echo "    RESULTATS SAUVEGARDÉS DANS:          "
echo "    $OUTPUT_FILE                          "
echo "=========================================="
echo ""
cat $OUTPUT_FILE

echo ""
echo "Job completed at $(date)"
