#!/bin/bash
# Script to generate CSV data for all versions

RESULTS_DIR="results/sequential"
BUILD_DIR="build"

mkdir -p "$RESULTS_DIR"

# Function to run a version and extract results
run_version() {
    local version=$1
    local order=$2
    local timeout_sec=$3

    result=$(timeout $timeout_sec ./$BUILD_DIR/golomb_v$version $order 2>&1)

    if [ $? -eq 124 ]; then
        echo "TIMEOUT"
        return 1
    fi

    time_ms=$(echo "$result" | grep -oP 'Time:\s*\K[\d.]+' || echo "0")
    nodes=$(echo "$result" | grep -oP 'Nodes explored:\s*\K[\d,]+' | tr -d ',' || echo "0")
    nodes_pruned=$(echo "$result" | grep -oP 'Nodes pruned:\s*\K[\d,]+' | tr -d ',' || echo "0")
    solution=$(echo "$result" | grep -oP 'Solution:\s*\K\[.*?\]' || echo "[]")
    length=$(echo "$result" | grep -oP 'Length:\s*\K\d+' || echo "0")

    echo "$version,$order,$time_ms,$nodes,$nodes_pruned,\"$solution\",$length"
}

echo "=== Generating v1 results (brute force) ==="
echo "version,order,time_ms,nodes_explored,nodes_pruned,solution,length" > $RESULTS_DIR/v1_results.csv
for order in 4 5 6 7; do
    echo "  Running v1 for G$order..."
    run_version 1 $order 300 >> $RESULTS_DIR/v1_results.csv
done

echo "=== Generating v2 results (backtracking) ==="
echo "version,order,time_ms,nodes_explored,nodes_pruned,solution,length" > $RESULTS_DIR/v2_results.csv
for order in 4 5 6 7 8; do
    echo "  Running v2 for G$order..."
    run_version 2 $order 300 >> $RESULTS_DIR/v2_results.csv
done

echo "=== Generating v3 results (branch & bound) ==="
echo "version,order,time_ms,nodes_explored,nodes_pruned,solution,length" > $RESULTS_DIR/v3_results.csv
for order in 4 5 6 7 8 9; do
    echo "  Running v3 for G$order..."
    run_version 3 $order 300 >> $RESULTS_DIR/v3_results.csv
done

echo "=== Generating v4 results (optimized) ==="
echo "version,order,time_ms,nodes_explored,nodes_pruned,solution,length" > $RESULTS_DIR/v4_results.csv
for order in 4 5 6 7 8 9 10 11; do
    echo "  Running v4 for G$order..."
    run_version 4 $order 600 >> $RESULTS_DIR/v4_results.csv
done

echo "=== Generating v5 results (final) ==="
echo "version,order,time_ms,nodes_explored,nodes_pruned,solution,length" > $RESULTS_DIR/v5_results.csv
for order in 4 5 6 7 8 9 10 11; do
    echo "  Running v5 for G$order..."
    run_version 5 $order 600 >> $RESULTS_DIR/v5_results.csv
done

echo ""
echo "=== All CSV files generated ==="
ls -la $RESULTS_DIR/
