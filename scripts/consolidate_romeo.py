#!/usr/bin/env python3
"""
Consolidate Romeo HPC benchmark results for visualization.

Organizes CSVs into sequential/, parallel/, openmp/ directories
and calculates proper speedup using correct T_seq baselines.
"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results" / "romeo"

# T_seq baselines (v1 single-threaded times in ms)
# G12 and G13 use v6 t1 as proxy (v6 is ~6% slower than v1)
T_SEQ = {
    10: 264.01,    # v1 measured
    11: 5093.03,   # v1 measured
    12: 47581.73,  # v6 t1 (v1 missing)
    13: 987642.49, # v6 t1 (v1 missing)
    14: None,      # Not available
}


def consolidate_sequential():
    """Consolidate sequential benchmark results."""
    seq_dir = RESULTS_DIR / "sequential"
    seq_dir.mkdir(exist_ok=True)

    rows = []

    # Process all seq_*.csv files
    for csv_file in RESULTS_DIR.glob("seq_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                rows.append(df.iloc[0].to_dict())
        except Exception as e:
            print(f"  Warning: {csv_file.name}: {e}")

    if not rows:
        print("  No sequential data found")
        return None

    result = pd.DataFrame(rows)

    # Normalize column names
    if 'threads' in result.columns:
        result = result.rename(columns={'threads': 'omp_threads'})

    # Sort by version, order, threads
    sort_cols = ['version', 'order']
    if 'omp_threads' in result.columns:
        sort_cols.append('omp_threads')
    result = result.sort_values(sort_cols)

    # Save consolidated file
    output_file = seq_dir / "all.csv"
    result.to_csv(output_file, index=False)
    print(f"  Saved {len(result)} rows to {output_file}")

    return result


def consolidate_parallel():
    """Consolidate MPI parallel benchmark results."""
    par_dir = RESULTS_DIR / "parallel"
    par_dir.mkdir(exist_ok=True)

    rows = []

    # Process all mpi_*.csv files (excluding trace files)
    for csv_file in RESULTS_DIR.glob("mpi_*.csv"):
        if "trace" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()

                # Extract version and procs from filename
                import re
                match_version = re.search(r'_v(\d+)_', csv_file.name)
                match_procs = re.search(r'_p(\d+)\.csv', csv_file.name)

                if match_version:
                    row['version'] = int(match_version.group(1))
                if match_procs:
                    row['procs'] = int(match_procs.group(1))

                # For v4 hybrid: use mpi_procs as procs if not set
                if 'mpi_procs' in row and pd.notna(row.get('mpi_procs')):
                    if 'procs' not in row or pd.isna(row.get('procs')):
                        row['procs'] = int(row['mpi_procs'])

                rows.append(row)
        except Exception as e:
            print(f"  Warning: {csv_file.name}: {e}")

    if not rows:
        print("  No parallel data found")
        return None

    result = pd.DataFrame(rows)

    # Ensure procs column exists and is filled
    if 'mpi_procs' in result.columns:
        result['procs'] = result.apply(
            lambda r: int(r['mpi_procs']) if pd.notna(r.get('mpi_procs')) else r.get('procs'),
            axis=1
        )

    # Calculate speedup using correct T_seq
    for idx, row in result.iterrows():
        order = int(row['order'])
        t_seq = T_SEQ.get(order)
        time_ms = row['time_ms']

        if t_seq is not None and pd.notna(time_ms) and time_ms > 0:
            speedup = t_seq / time_ms
            result.loc[idx, 'speedup'] = speedup

            # Calculate efficiency based on actual parallelism
            if 'total_workers' in row and pd.notna(row.get('total_workers')):
                workers = int(row['total_workers'])
            elif pd.notna(row.get('procs')):
                workers = int(row['procs'])
            else:
                workers = 1

            result.loc[idx, 'efficiency'] = speedup / workers * 100

    # Sort by version, order, procs
    sort_cols = ['version', 'order']
    if 'procs' in result.columns:
        sort_cols.append('procs')
    result = result.sort_values(sort_cols)

    # Save consolidated file
    output_file = par_dir / "all.csv"
    result.to_csv(output_file, index=False)
    print(f"  Saved {len(result)} rows to {output_file}")

    return result


def consolidate_openmp():
    """Consolidate OpenMP scaling results."""
    omp_dir = RESULTS_DIR / "openmp"
    omp_dir.mkdir(exist_ok=True)

    rows = []

    # Process v6 sequential files with multiple threads
    for csv_file in RESULTS_DIR.glob("seq_*_v6_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                row = df.iloc[0].to_dict()

                # Extract threads from filename
                import re
                match = re.search(r'_t(\d+)\.csv', csv_file.name)
                if match:
                    row['threads'] = int(match.group(1))

                rows.append(row)
        except Exception as e:
            print(f"  Warning: {csv_file.name}: {e}")

    if not rows:
        print("  No OpenMP data found")
        return None

    result = pd.DataFrame(rows)

    # Calculate speedup using T_seq (single-thread v6)
    for order in result['order'].unique():
        order = int(order)
        mask = result['order'] == order

        # Get single-thread baseline for this order
        t1_mask = mask & (result['threads'] == 1)
        if t1_mask.any():
            t_seq = result.loc[t1_mask, 'time_ms'].values[0]
            result.loc[mask, 'speedup'] = t_seq / result.loc[mask, 'time_ms']
            result.loc[mask, 'efficiency'] = result.loc[mask, 'speedup'] / result.loc[mask, 'threads'] * 100

    # Sort
    result = result.sort_values(['order', 'threads'])

    # Save
    output_file = omp_dir / "all.csv"
    result.to_csv(output_file, index=False)
    print(f"  Saved {len(result)} rows to {output_file}")

    return result


def print_summary(seq_df, par_df, omp_df):
    """Print summary of consolidated data."""
    print("\n" + "="*60)
    print(" DATA SUMMARY")
    print("="*60)

    if seq_df is not None:
        print(f"\nSequential: {len(seq_df)} entries")
        print(f"  Versions: {sorted(seq_df['version'].unique())}")
        print(f"  Orders: {sorted(seq_df['order'].unique())}")

    if par_df is not None:
        print(f"\nParallel (MPI): {len(par_df)} entries")
        print(f"  Versions: {sorted(par_df['version'].unique())}")
        print(f"  Orders: {sorted(par_df['order'].unique())}")
        if 'procs' in par_df.columns:
            print(f"  Procs: {sorted(par_df['procs'].unique())}")

        # Print speedup summary for useful orders
        print("\n  Speedup Summary (G11-G13):")
        for version in sorted(par_df['version'].unique()):
            for order in [11, 12, 13]:
                mask = (par_df['version'] == version) & (par_df['order'] == order)
                if mask.any():
                    subset = par_df[mask].sort_values('procs' if 'procs' in par_df.columns else 'time_ms')
                    max_speedup = subset['speedup'].max()
                    best_procs = subset.loc[subset['speedup'].idxmax(), 'procs'] if 'procs' in subset.columns else 'N/A'
                    print(f"    v{version} G{order}: max speedup = {max_speedup:.2f}x at p={best_procs}")

    if omp_df is not None:
        print(f"\nOpenMP: {len(omp_df)} entries")
        print(f"  Orders: {sorted(omp_df['order'].unique())}")
        print(f"  Threads: {sorted(omp_df['threads'].unique())}")

        # Print OpenMP speedup
        print("\n  OpenMP Speedup (v6):")
        for order in sorted(omp_df['order'].unique()):
            mask = omp_df['order'] == order
            if mask.any() and 'speedup' in omp_df.columns:
                subset = omp_df[mask].sort_values('threads')
                max_speedup = subset['speedup'].max()
                best_threads = subset.loc[subset['speedup'].idxmax(), 'threads']
                print(f"    G{int(order)}: max speedup = {max_speedup:.2f}x at t={int(best_threads)}")

    print("\n" + "="*60)
    print(" T_seq BASELINES USED")
    print("="*60)
    for order, t_seq in sorted(T_SEQ.items()):
        if t_seq:
            source = "v1" if order <= 11 else "v6 t1 (proxy)"
            print(f"  G{order}: {t_seq:.2f} ms ({source})")
    print()


def main():
    print("="*60)
    print(" Consolidating Romeo HPC Benchmark Results")
    print("="*60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    os.chdir(RESULTS_DIR)

    print("\n[1/3] Sequential results...")
    seq_df = consolidate_sequential()

    print("\n[2/3] Parallel (MPI) results...")
    par_df = consolidate_parallel()

    print("\n[3/3] OpenMP results...")
    omp_df = consolidate_openmp()

    print_summary(seq_df, par_df, omp_df)

    print("Done! Run visualization with:")
    print(f"  python3 tools/visualization/generate_all.py --data results/romeo --output results/romeo/plots")


if __name__ == "__main__":
    main()
