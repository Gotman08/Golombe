#!/bin/bash
# Script to generate CSV data for parallel MPI versions

RESULTS_DIR="results/parallel"
BUILD_DIR="build"

mkdir -p "$RESULTS_DIR"

# Get sequential reference times from v5
get_seq_time() {
    local order=$1
    grep "^5,$order," results/sequential/v5_results.csv | cut -d',' -f3
}

# Output file
OUTPUT="$RESULTS_DIR/benchmark_results.csv"
echo "version,order,procs,time_ms,speedup,efficiency,solution" > "$OUTPUT"

# Run MPI benchmark
run_mpi() {
    local version=$1
    local order=$2
    local procs=$3
    local timeout_sec=$4

    echo "  Running MPI v$version for G$order with $procs procs..."

    result=$(timeout $timeout_sec mpirun --oversubscribe -np $procs ./$BUILD_DIR/golomb_mpi_v$version $order 2>&1)

    if [ $? -eq 124 ]; then
        echo "    TIMEOUT"
        return 1
    fi

    # Try different output formats
    time_ms=$(echo "$result" | grep -oP '(Total time|Time):\s*\K[\d.]+' | head -1)
    if [ -z "$time_ms" ]; then
        time_ms="0"
    fi

    solution=$(echo "$result" | grep -oP 'Solution:\s*\K\[.*?\]' || echo "[]")

    # Get sequential time for speedup calculation
    seq_time=$(get_seq_time $order)

    if [ -n "$seq_time" ] && [ "$time_ms" != "0" ] && [ $(echo "$time_ms > 0" | bc) -eq 1 ]; then
        speedup=$(echo "scale=2; $seq_time / $time_ms" | bc)
        efficiency=$(echo "scale=2; $speedup / $procs" | bc)
    else
        speedup="0"
        efficiency="0"
    fi

    echo "$version,$order,$procs,$time_ms,$speedup,$efficiency,\"$solution\"" >> "$OUTPUT"
    echo "    Time: ${time_ms}ms, Speedup: ${speedup}x, Efficiency: ${efficiency}"
}

echo "=== Generating MPI v1 results (basic master/worker) ==="
for order in 8 9; do
    for procs in 2 4 8; do
        run_mpi 1 $order $procs 300
    done
done

echo ""
echo "=== Generating MPI v2 results (hypercube) ==="
for order in 8 9; do
    for procs in 2 4 8; do
        run_mpi 2 $order $procs 300
    done
done

echo ""
echo "=== Generating MPI v3 results (optimized) ==="
for order in 8 9 10; do
    for procs in 2 4 8; do
        run_mpi 3 $order $procs 600
    done
done

echo ""
echo "=== Parallel CSV generated ==="
cat "$OUTPUT"
