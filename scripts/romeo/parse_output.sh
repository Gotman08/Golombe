#!/bin/bash
# =============================================================================
# Parse output from Golomb solver versions without native --csv support
# =============================================================================
# This script extracts timing and solution data from stdout of v1-v4 and MPI v1-v2
# which don't have built-in CSV export functionality.
# =============================================================================

# Parse sequential solver output (v1-v4)
# Arguments: output_file version order
# Output: CSV line (version,order,time_ms,nodes_explored,nodes_pruned,solution,length)
parse_sequential() {
    local output_file="$1"
    local version="$2"
    local order="$3"

    # Extract fields using grep with Perl regex
    local time=$(grep -oP 'Time:\s*\K[\d.]+' "$output_file" 2>/dev/null | head -1)
    local nodes=$(grep -oP 'Nodes explored:\s*\K[\d,]+' "$output_file" 2>/dev/null | tr -d ',' | head -1)
    local pruned=$(grep -oP 'Nodes pruned:\s*\K[\d,]+' "$output_file" 2>/dev/null | tr -d ',' | head -1)
    local solution=$(grep -oP 'Solution:\s*\K\[.*?\]' "$output_file" 2>/dev/null | head -1)
    local length=$(grep -oP 'Length:\s*\K\d+' "$output_file" 2>/dev/null | head -1)

    # Alternative patterns for different output formats
    if [[ -z "$time" ]]; then
        time=$(grep -oP 'time:\s*\K[\d.]+' "$output_file" 2>/dev/null | head -1)
    fi
    if [[ -z "$time" ]]; then
        time=$(grep -oP '\K[\d.]+\s*ms' "$output_file" 2>/dev/null | grep -oP '[\d.]+' | head -1)
    fi

    # Default values if not found
    time=${time:-0}
    nodes=${nodes:-0}
    pruned=${pruned:-0}
    length=${length:-0}

    # Output CSV line
    echo "${version},${order},${time},${nodes},${pruned},\"${solution}\",${length}"
}

# Parse MPI solver output (v1-v2)
# Arguments: output_file version order procs
# Output: CSV line (version,order,procs,time_ms,speedup,efficiency,nodes,solution,length)
parse_mpi() {
    local output_file="$1"
    local version="$2"
    local order="$3"
    local procs="$4"

    # Extract fields - MPI versions have slightly different output format
    local time=$(grep -oP 'Total time:\s*\K[\d.]+' "$output_file" 2>/dev/null | head -1)
    if [[ -z "$time" ]]; then
        time=$(grep -oP 'Time:\s*\K[\d.]+' "$output_file" 2>/dev/null | head -1)
    fi

    local nodes=$(grep -oP 'Total nodes:\s*\K[\d,]+' "$output_file" 2>/dev/null | tr -d ',' | head -1)
    if [[ -z "$nodes" ]]; then
        nodes=$(grep -oP 'Nodes explored:\s*\K[\d,]+' "$output_file" 2>/dev/null | tr -d ',' | head -1)
    fi

    local solution=$(grep -oP 'Solution:\s*\K\[.*?\]' "$output_file" 2>/dev/null | head -1)
    local length=$(grep -oP 'Length:\s*\K\d+' "$output_file" 2>/dev/null | head -1)

    # Default values
    time=${time:-0}
    nodes=${nodes:-0}
    length=${length:-0}

    # Speedup and efficiency are 0 (will be calculated later with sequential baseline)
    echo "${version},${order},${procs},${time},0,0,${nodes},\"${solution}\",${length}"
}

# Parse v6 hardware-optimized output (if not using --csv)
# Arguments: output_file version order threads
# Output: CSV line (version,order,threads,time_ms,nodes_explored,nodes_pruned,solution,length)
parse_v6() {
    local output_file="$1"
    local version="$2"
    local order="$3"
    local threads="$4"

    local time=$(grep -oP 'Time:\s*\K[\d.]+' "$output_file" 2>/dev/null | head -1)
    local nodes=$(grep -oP 'Nodes explored:\s*\K[\d,]+' "$output_file" 2>/dev/null | tr -d ',' | head -1)
    local pruned=$(grep -oP 'Nodes pruned:\s*\K[\d,]+' "$output_file" 2>/dev/null | tr -d ',' | head -1)
    local solution=$(grep -oP 'Solution:\s*\K\[.*?\]' "$output_file" 2>/dev/null | head -1)
    local length=$(grep -oP 'Length:\s*\K\d+' "$output_file" 2>/dev/null | head -1)

    # Default values
    time=${time:-0}
    nodes=${nodes:-0}
    pruned=${pruned:-0}
    length=${length:-0}

    echo "${version},${order},${threads},${time},${nodes},${pruned},\"${solution}\",${length}"
}

# Write CSV header for sequential results
write_seq_header() {
    echo "version,order,time_ms,nodes_explored,nodes_pruned,solution,length"
}

# Write CSV header for v6 results (includes threads)
write_v6_header() {
    echo "version,order,threads,time_ms,nodes_explored,nodes_pruned,solution,length"
}

# Write CSV header for MPI results
write_mpi_header() {
    echo "version,order,procs,time_ms,speedup,efficiency,nodes,solution,length"
}

# Check if file contains error messages
check_for_errors() {
    local output_file="$1"

    if grep -qi "error\|failed\|abort\|segfault\|timeout" "$output_file" 2>/dev/null; then
        return 1
    fi
    return 0
}

# Main function for standalone usage
main() {
    local mode="$1"
    local output_file="$2"
    shift 2

    case "$mode" in
        sequential)
            parse_sequential "$output_file" "$@"
            ;;
        mpi)
            parse_mpi "$output_file" "$@"
            ;;
        v6)
            parse_v6 "$output_file" "$@"
            ;;
        *)
            echo "Usage: $0 {sequential|mpi|v6} output_file [args...]" >&2
            echo "  sequential: output_file version order" >&2
            echo "  mpi: output_file version order procs" >&2
            echo "  v6: output_file version order threads" >&2
            exit 1
            ;;
    esac
}

# Run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
