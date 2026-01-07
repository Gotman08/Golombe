#!/usr/bin/env python3
"""
Generate Visualization Plots from Romeo HPC Benchmark Results

This script loads all CSV files from results/romeo/ and generates
comprehensive performance analysis plots.

Usage:
    python generate_romeo_plots.py [--output DIR]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import re
import argparse
from typing import Tuple, Dict, List

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'v1': '#1f77b4',  # Blue
    'v2': '#ff7f0e',  # Orange
    'v3': '#2ca02c',  # Green
    'v4': '#d62728',  # Red
    'v5': '#9467bd',  # Purple
}
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']


def load_romeo_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all Romeo CSV files and return consolidated DataFrames."""
    seq_dfs = []
    mpi_dfs = []

    for csv_file in data_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if df.empty or len(df.columns) < 2:
                continue

            # Extract metadata from filename
            filename = csv_file.stem

            # Parse filename pattern: seq_G{order}_v{version}_c{cores}_t{threads}
            # or mpi_G{order}_v{version}_n{nodes}_p{procs}_t{threads}
            if filename.startswith('seq_'):
                # Sequential job
                match = re.match(r'seq_G(\d+)_v(\d+)_c(\d+)_t(\d+)', filename)
                if match:
                    order, version, cores, threads = map(int, match.groups())
                    df['order'] = order
                    df['version'] = version
                    df['cores'] = cores
                    df['config_threads'] = threads
                seq_dfs.append(df)
            elif filename.startswith('mpi_'):
                # MPI job - check for different patterns
                # Pattern 1: mpi_G{order}_v{version}_n{nodes}_p{procs}_t{threads} (hybrid)
                match = re.match(r'mpi_G(\d+)_v(\d+)_n(\d+)_p(\d+)_t(\d+)', filename)
                if match:
                    order, version, nodes, procs, threads = map(int, match.groups())
                    if 'order' not in df.columns:
                        df['order'] = order
                    if 'version' not in df.columns:
                        df['version'] = version
                    df['nodes'] = nodes
                    if 'procs' not in df.columns and 'mpi_procs' in df.columns:
                        df['procs'] = df['mpi_procs']
                    elif 'procs' not in df.columns:
                        df['procs'] = procs
                    mpi_dfs.append(df)
                    continue

                # Pattern 2: mpi_G{order}_v5_n{nodes}_p{procs} (pure MPI, no threads)
                match = re.match(r'mpi_G(\d+)_v5_n(\d+)_p(\d+)', filename)
                if match:
                    order, nodes, procs = map(int, match.groups())
                    if 'order' not in df.columns:
                        df['order'] = order
                    if 'version' not in df.columns:
                        df['version'] = 5
                    df['nodes'] = nodes
                    if 'procs' not in df.columns and 'mpi_procs' in df.columns:
                        df['procs'] = df['mpi_procs']
                    elif 'procs' not in df.columns:
                        df['procs'] = procs
                    mpi_dfs.append(df)
                    continue

                # Pattern 3: mpi_G{order}_v{version}_p{procs} (older format)
                match = re.match(r'mpi_G(\d+)_v(\d+)_p(\d+)', filename)
                if match:
                    order, version, procs = map(int, match.groups())
                    df['order'] = order
                    df['version'] = version
                    df['procs'] = procs
                    mpi_dfs.append(df)

        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue

    seq_df = pd.concat(seq_dfs, ignore_index=True) if seq_dfs else pd.DataFrame()
    mpi_df = pd.concat(mpi_dfs, ignore_index=True) if mpi_dfs else pd.DataFrame()

    return seq_df, mpi_df


def plot_time_comparison(seq_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot execution time comparison between versions."""
    if seq_df.empty:
        print("  Skipping time_comparison (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get best time per version and order
    best_times = seq_df.groupby(['version', 'order'])['time_ms'].min().reset_index()

    for version in sorted(best_times['version'].unique()):
        data = best_times[best_times['version'] == version].sort_values('order')
        color = COLORS.get(f'v{version}', '#333333')
        ax.semilogy(data['order'], data['time_ms'],
                    marker='o', label=f'v{version}', color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Golomb Order', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Execution Time vs Golomb Order (Romeo HPC)', fontsize=14)
    ax.legend(title='Version')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_time_comparison.png")


def plot_omp_speedup(seq_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot OpenMP speedup for v2."""
    v2_data = seq_df[seq_df['version'] == 2].copy()
    if v2_data.empty or 'config_threads' not in v2_data.columns:
        print("  Skipping omp_speedup (no v2 data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate speedup relative to single thread
    for order in sorted(v2_data['order'].unique()):
        if order < 10:  # Skip small orders
            continue

        order_data = v2_data[v2_data['order'] == order]

        # Get baseline (1 thread)
        baseline = order_data[order_data['config_threads'] == 1]['time_ms'].min()
        if pd.isna(baseline) or baseline == 0:
            continue

        # Group by threads and get best time
        grouped = order_data.groupby('config_threads')['time_ms'].min().reset_index()
        grouped['speedup'] = baseline / grouped['time_ms']
        grouped = grouped.sort_values('config_threads')

        ax.plot(grouped['config_threads'], grouped['speedup'],
                marker='o', label=f'G{order}', linewidth=2, markersize=6)

    # Ideal speedup line
    threads = sorted(v2_data['config_threads'].unique())
    ax.plot(threads, threads, '--', color='gray', alpha=0.7, label='Ideal')

    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('OpenMP Speedup (v2) - Romeo HPC', fontsize=14)
    ax.legend(title='Order')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_omp_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_omp_speedup.png")


def plot_omp_efficiency(seq_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot OpenMP efficiency for v2."""
    v2_data = seq_df[seq_df['version'] == 2].copy()
    if v2_data.empty or 'config_threads' not in v2_data.columns:
        print("  Skipping omp_efficiency (no v2 data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for order in sorted(v2_data['order'].unique()):
        if order < 10:
            continue

        order_data = v2_data[v2_data['order'] == order]

        # Get baseline (1 thread)
        baseline = order_data[order_data['config_threads'] == 1]['time_ms'].min()
        if pd.isna(baseline) or baseline == 0:
            continue

        grouped = order_data.groupby('config_threads')['time_ms'].min().reset_index()
        grouped['speedup'] = baseline / grouped['time_ms']
        grouped['efficiency'] = (grouped['speedup'] / grouped['config_threads']) * 100
        grouped = grouped.sort_values('config_threads')

        ax.plot(grouped['config_threads'], grouped['efficiency'],
                marker='o', label=f'G{order}', linewidth=2, markersize=6)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Ideal (100%)')
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.5, label='50% threshold')

    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('OpenMP Parallel Efficiency (v2) - Romeo HPC', fontsize=14)
    ax.legend(title='Order', loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 120)

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_omp_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_omp_efficiency.png")


def plot_strong_scaling(seq_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot strong scaling (time vs threads) for multiple orders."""
    v2_data = seq_df[seq_df['version'] == 2].copy()
    if v2_data.empty or 'config_threads' not in v2_data.columns:
        print("  Skipping strong_scaling (no v2 data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for order in sorted(v2_data['order'].unique()):
        if order < 10:
            continue

        order_data = v2_data[v2_data['order'] == order]
        grouped = order_data.groupby('config_threads')['time_ms'].min().reset_index()
        grouped = grouped.sort_values('config_threads')

        if len(grouped) < 2:
            continue

        ax.loglog(grouped['config_threads'], grouped['time_ms'],
                  marker='o', label=f'G{order}', linewidth=2, markersize=6)

    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Strong Scaling: Time vs Threads (v2) - Romeo HPC', fontsize=14)
    ax.legend(title='Order')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_strong_scaling_omp.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_strong_scaling_omp.png")


def plot_mpi_scaling(mpi_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot MPI scaling for v3 and v4."""
    if mpi_df.empty or 'procs' not in mpi_df.columns:
        print("  Skipping mpi_scaling (no MPI data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # v3 scaling
    ax1 = axes[0]
    v3_data = mpi_df[mpi_df['version'] == 3]
    if not v3_data.empty:
        for order in sorted(v3_data['order'].unique()):
            if order < 10:
                continue
            order_data = v3_data[v3_data['order'] == order]
            grouped = order_data.groupby('procs')['time_ms'].min().reset_index()
            grouped = grouped[grouped['time_ms'] > 0].sort_values('procs')
            if len(grouped) >= 2:
                ax1.loglog(grouped['procs'], grouped['time_ms'],
                          marker='o', label=f'G{order}', linewidth=2)

    ax1.set_xlabel('MPI Processes', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('v3 (Master/Worker) MPI Scaling', fontsize=14)
    ax1.legend(title='Order')
    ax1.grid(True, alpha=0.3, which='both')

    # v4 scaling
    ax2 = axes[1]
    v4_data = mpi_df[mpi_df['version'] == 4]
    if not v4_data.empty:
        for order in sorted(v4_data['order'].unique()):
            if order < 10:
                continue
            order_data = v4_data[v4_data['order'] == order]
            grouped = order_data.groupby('procs')['time_ms'].min().reset_index()
            grouped = grouped[grouped['time_ms'] > 0].sort_values('procs')
            if len(grouped) >= 2:
                ax2.loglog(grouped['procs'], grouped['time_ms'],
                          marker='s', label=f'G{order}', linewidth=2)

    ax2.set_xlabel('MPI Processes', fontsize=12)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('v4 (Hypercube) MPI Scaling', fontsize=14)
    ax2.legend(title='Order')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_mpi_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_mpi_scaling.png")


def plot_mpi_comparison(mpi_df: pd.DataFrame, output_dir: Path) -> None:
    """Compare v3 vs v4 MPI performance."""
    if mpi_df.empty or 'procs' not in mpi_df.columns:
        print("  Skipping mpi_comparison (no MPI data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Find common orders
    orders = sorted(set(mpi_df[mpi_df['version'] == 3]['order'].unique()) &
                   set(mpi_df[mpi_df['version'] == 4]['order'].unique()))

    x = np.arange(len(orders))
    width = 0.35

    v3_times = []
    v4_times = []

    for order in orders:
        v3_time = mpi_df[(mpi_df['version'] == 3) & (mpi_df['order'] == order)]['time_ms'].min()
        v4_time = mpi_df[(mpi_df['version'] == 4) & (mpi_df['order'] == order)]['time_ms'].min()
        v3_times.append(v3_time if v3_time > 0 else np.nan)
        v4_times.append(v4_time if v4_time > 0 else np.nan)

    bars1 = ax.bar(x - width/2, v3_times, width, label='v3 (Master/Worker)', color=COLORS['v3'])
    bars2 = ax.bar(x + width/2, v4_times, width, label='v4 (Hypercube)', color=COLORS['v4'])

    ax.set_xlabel('Golomb Order', fontsize=12)
    ax.set_ylabel('Best Time (ms)', fontsize=12)
    ax.set_title('MPI Version Comparison - Romeo HPC', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{o}' for o in orders])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_mpi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_mpi_comparison.png")


def plot_heatmap(seq_df: pd.DataFrame, mpi_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate heatmap of best times per version and order."""
    # Combine data
    all_data = []

    if not seq_df.empty:
        seq_best = seq_df.groupby(['version', 'order'])['time_ms'].min().reset_index()
        all_data.append(seq_best)

    if not mpi_df.empty and 'time_ms' in mpi_df.columns:
        mpi_best = mpi_df[mpi_df['time_ms'] > 0].groupby(['version', 'order'])['time_ms'].min().reset_index()
        all_data.append(mpi_best)

    if not all_data:
        print("  Skipping heatmap (no data)")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.groupby(['version', 'order'])['time_ms'].min().reset_index()

    # Pivot for heatmap
    pivot = combined.pivot(index='version', columns='order', values='time_ms')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Log transform for better visualization
    log_pivot = np.log10(pivot.replace(0, np.nan))

    im = ax.imshow(log_pivot, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f'G{c}' for c in pivot.columns])
    ax.set_yticklabels([f'v{v}' for v in pivot.index])

    ax.set_xlabel('Golomb Order', fontsize=12)
    ax.set_ylabel('Version', fontsize=12)
    ax.set_title('Execution Time Heatmap (log10 ms) - Romeo HPC', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(time_ms)', fontsize=10)

    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val) and val > 0:
                text = f'{val:.0f}' if val >= 10 else f'{val:.1f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                       color='white' if log_pivot.iloc[i, j] > 2 else 'black')

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_heatmap.png")


def plot_best_times(seq_df: pd.DataFrame, mpi_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of best execution times per order across all versions."""
    all_data = []

    if not seq_df.empty:
        all_data.append(seq_df[['version', 'order', 'time_ms']])
    if not mpi_df.empty and 'time_ms' in mpi_df.columns:
        mpi_valid = mpi_df[mpi_df['time_ms'] > 0][['version', 'order', 'time_ms']]
        all_data.append(mpi_valid)

    if not all_data:
        print("  Skipping best_times (no data)")
        return

    combined = pd.concat(all_data, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    orders = sorted(combined['order'].unique())
    versions = sorted(combined['version'].unique())

    x = np.arange(len(orders))
    width = 0.15

    for i, version in enumerate(versions):
        times = []
        for order in orders:
            t = combined[(combined['version'] == version) & (combined['order'] == order)]['time_ms'].min()
            times.append(t if not pd.isna(t) and t > 0 else np.nan)

        color = COLORS.get(f'v{version}', '#333333')
        offset = (i - len(versions)/2 + 0.5) * width
        ax.bar(x + offset, times, width, label=f'v{version}', color=color)

    ax.set_xlabel('Golomb Order', fontsize=12)
    ax.set_ylabel('Best Time (ms)', fontsize=12)
    ax.set_title('Best Execution Time by Version and Order - Romeo HPC', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{o}' for o in orders])
    ax.legend(title='Version')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_best_times.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_best_times.png")


def plot_cores_vs_threads(seq_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze impact of cores vs threads configuration."""
    v2_data = seq_df[(seq_df['version'] == 2) & (seq_df['order'] >= 10)].copy()
    if v2_data.empty or 'cores' not in v2_data.columns:
        print("  Skipping cores_vs_threads (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by order and cores
    for order in sorted(v2_data['order'].unique()):
        order_data = v2_data[v2_data['order'] == order]
        grouped = order_data.groupby('cores')['time_ms'].min().reset_index()
        grouped = grouped.sort_values('cores')

        if len(grouped) >= 2:
            ax.plot(grouped['cores'], grouped['time_ms'],
                   marker='o', label=f'G{order}', linewidth=2, markersize=6)

    ax.set_xlabel('Number of Cores Allocated', fontsize=12)
    ax.set_ylabel('Best Time (ms)', fontsize=12)
    ax.set_title('Impact of Core Allocation (v2) - Romeo HPC', fontsize=14)
    ax.legend(title='Order')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'romeo_cores_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> romeo_cores_impact.png")


def main():
    parser = argparse.ArgumentParser(description='Generate Romeo HPC benchmark plots')
    parser.add_argument('--data', '-d', type=Path,
                       default=Path('results/romeo'),
                       help='Directory containing Romeo CSV files')
    parser.add_argument('--output', '-o', type=Path,
                       default=Path('results/plots'),
                       help='Output directory for plots')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.parent.parent
    data_dir = script_dir / args.data if not args.data.is_absolute() else args.data
    output_dir = script_dir / args.output if not args.output.is_absolute() else args.output

    print(f"=== Romeo HPC Visualization Generator ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading CSV files...")
    seq_df, mpi_df = load_romeo_data(data_dir)

    print(f"  Sequential data: {len(seq_df)} rows")
    print(f"  MPI data: {len(mpi_df)} rows")
    print()

    # Generate plots
    print("Generating plots...")

    plot_time_comparison(seq_df, output_dir)
    plot_omp_speedup(seq_df, output_dir)
    plot_omp_efficiency(seq_df, output_dir)
    plot_strong_scaling(seq_df, output_dir)
    plot_mpi_scaling(mpi_df, output_dir)
    plot_mpi_comparison(mpi_df, output_dir)
    plot_heatmap(seq_df, mpi_df, output_dir)
    plot_best_times(seq_df, mpi_df, output_dir)
    plot_cores_vs_threads(seq_df, output_dir)

    print()
    print(f"=== Done! Plots saved to {output_dir} ===")


if __name__ == '__main__':
    main()
