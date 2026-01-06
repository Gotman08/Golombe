#!/usr/bin/env python3
"""
Golomb Ruler Solver - Romeo HPC Results Visualization

Generates plots from Romeo cluster benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
ROMEO_DIR = PROJECT_DIR / 'results' / 'romeo'
PLOTS_DIR = ROMEO_DIR / 'plots'

# Create plots directory
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_romeo_results():
    """Load all Romeo CSV results."""
    seq_results = []
    mpi_results = []

    for csv_file in ROMEO_DIR.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'procs' in df.columns:
                mpi_results.append(df)
            else:
                seq_results.append(df)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")

    seq_df = pd.concat(seq_results, ignore_index=True) if seq_results else pd.DataFrame()
    mpi_df = pd.concat(mpi_results, ignore_index=True) if mpi_results else pd.DataFrame()

    return seq_df, mpi_df


def plot_sequential_vs_mpi(seq_df, mpi_df, save=True):
    """Compare sequential vs MPI execution times for same order."""
    if seq_df.empty or mpi_df.empty:
        print("Insufficient data for seq vs MPI comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Find common orders
    common_orders = set(seq_df['order'].unique()) & set(mpi_df['order'].unique())

    if not common_orders:
        print("No common orders found between sequential and MPI results")
        return

    x = np.arange(len(common_orders))
    width = 0.35

    seq_times = []
    mpi_times = []
    orders = sorted(common_orders)

    for order in orders:
        seq_time = seq_df[seq_df['order'] == order]['time_ms'].values
        mpi_time = mpi_df[mpi_df['order'] == order]['time_ms'].values
        seq_times.append(seq_time[0] if len(seq_time) > 0 else 0)
        mpi_times.append(mpi_time[0] if len(mpi_time) > 0 else 0)

    bars1 = ax.bar(x - width/2, seq_times, width, label='Sequentiel', color='#377eb8', edgecolor='black')
    bars2 = ax.bar(x + width/2, mpi_times, width, label='MPI (8 procs)', color='#4daf4a', edgecolor='black')

    ax.set_xlabel('Ordre de Golomb', fontsize=12)
    ax.set_ylabel('Temps d\'execution (ms)', fontsize=12)
    ax.set_title('Comparaison Sequentiel vs MPI - Romeo HPC', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{o}' for o in orders])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, seq_times):
        if val > 0:
            ax.annotate(f'{val:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar, val in zip(bars2, mpi_times):
        if val > 0:
            ax.annotate(f'{val:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    if save:
        filepath = PLOTS_DIR / 'romeo_seq_vs_mpi.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_speedup(seq_df, mpi_df, save=True):
    """Plot MPI speedup compared to sequential."""
    if seq_df.empty or mpi_df.empty:
        print("Insufficient data for speedup plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    common_orders = set(seq_df['order'].unique()) & set(mpi_df['order'].unique())

    if not common_orders:
        return

    speedups = []
    orders = sorted(common_orders)
    procs_used = []

    for order in orders:
        seq_time = seq_df[seq_df['order'] == order]['time_ms'].values[0]
        mpi_row = mpi_df[mpi_df['order'] == order].iloc[0]
        mpi_time = mpi_row['time_ms']
        procs = mpi_row['procs']

        speedup = seq_time / mpi_time if mpi_time > 0 else 0
        speedups.append(speedup)
        procs_used.append(procs)

    colors = ['#4daf4a' if s > 1 else '#e41a1c' for s in speedups]
    bars = ax.bar(range(len(orders)), speedups, color=colors, edgecolor='black', linewidth=1.5)

    # Add ideal speedup line
    for i, procs in enumerate(procs_used):
        ax.axhline(y=procs, xmin=(i/len(orders))-0.05, xmax=(i/len(orders))+0.15,
                   color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5, linewidth=1)

    # Add value labels
    for i, (bar, speedup, procs) in enumerate(zip(bars, speedups, procs_used)):
        efficiency = (speedup / procs) * 100
        ax.annotate(f'{speedup:.2f}x\n({efficiency:.0f}% eff.)',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Ordre de Golomb', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup MPI vs Sequentiel - Romeo HPC', fontsize=14)
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([f'G{o}\n({p} procs)' for o, p in zip(orders, procs_used)])
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4daf4a', edgecolor='black', label='Speedup mesure'),
        Patch(facecolor='none', edgecolor='red', linestyle='--', label='Speedup ideal'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    if save:
        filepath = PLOTS_DIR / 'romeo_speedup.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_golomb_ruler(solution, order, save=True):
    """Visualize a Golomb ruler as a graphical representation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2]})

    # Parse solution if string
    if isinstance(solution, str):
        solution = eval(solution)

    length = solution[-1]

    # Top: Linear ruler representation
    ax1.set_xlim(-1, length + 1)
    ax1.set_ylim(-0.5, 1.5)

    # Draw ruler baseline
    ax1.axhline(y=0.5, color='black', linewidth=2, xmin=0.02, xmax=0.98)

    # Draw marks
    colors = plt.cm.tab10(np.linspace(0, 1, len(solution)))
    for i, (mark, color) in enumerate(zip(solution, colors)):
        ax1.plot(mark, 0.5, 'o', markersize=15, color=color, markeredgecolor='black', markeredgewidth=2)
        ax1.annotate(str(mark), xy=(mark, 0.5), xytext=(0, 20), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')
        # Draw tick
        ax1.plot([mark, mark], [0.3, 0.7], color='black', linewidth=2)

    ax1.set_title(f'Regle de Golomb G{order} - Longueur Optimale: {length}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.axis('off')

    # Bottom: Difference visualization (matrix style)
    n = len(solution)
    differences = []
    diff_labels = []

    for i in range(n):
        for j in range(i+1, n):
            diff = solution[j] - solution[i]
            differences.append(diff)
            diff_labels.append(f'{solution[i]}-{solution[j]}')

    # Create difference matrix for visualization
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff_matrix[i, j] = solution[j] - solution[i]

    # Plot as triangular heatmap
    mask = np.tril(np.ones_like(diff_matrix, dtype=bool))
    masked_matrix = np.ma.array(diff_matrix, mask=mask)

    im = ax2.imshow(masked_matrix, cmap='YlOrRd', aspect='auto')

    # Add text annotations
    for i in range(n):
        for j in range(i+1, n):
            text = ax2.text(j, i, int(diff_matrix[i, j]),
                           ha='center', va='center', fontsize=10, fontweight='bold')

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([f'm{i}={solution[i]}' for i in range(n)], fontsize=9)
    ax2.set_yticklabels([f'm{i}={solution[i]}' for i in range(n)], fontsize=9)
    ax2.set_title(f'Matrice des differences (toutes uniques: {len(set(differences))} differences)', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Difference', fontsize=10)

    plt.tight_layout()
    if save:
        filepath = PLOTS_DIR / f'romeo_golomb_G{order}.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_summary_table(seq_df, mpi_df, save=True):
    """Create a summary table of all Romeo results."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Prepare data
    rows = []

    # Sequential results
    if not seq_df.empty:
        for _, row in seq_df.iterrows():
            rows.append([
                f"Sequentiel v{int(row['version'])}",
                f"G{int(row['order'])}",
                f"{row['time_ms']:.2f} ms",
                '-',
                f"{int(row.get('nodes_explored', 0)):,}",
                str(row.get('solution', '-'))[:40] + '...' if len(str(row.get('solution', ''))) > 40 else str(row.get('solution', '-')),
                str(int(row.get('length', 0)))
            ])

    # MPI results
    if not mpi_df.empty:
        for _, row in mpi_df.iterrows():
            speedup = row.get('speedup', 0)
            if speedup == 0 and not seq_df.empty:
                seq_time = seq_df[seq_df['order'] == row['order']]['time_ms']
                if len(seq_time) > 0:
                    speedup = seq_time.values[0] / row['time_ms']

            rows.append([
                f"MPI v{int(row['version'])}",
                f"G{int(row['order'])}",
                f"{row['time_ms']:.2f} ms",
                f"{int(row['procs'])} procs",
                f"{int(row.get('nodes', 0)):,}",
                str(row.get('solution', '-'))[:40] + '...' if len(str(row.get('solution', ''))) > 40 else str(row.get('solution', '-')),
                str(int(row.get('length', 0)))
            ])

    if not rows:
        print("No data for summary table")
        return

    headers = ['Version', 'Ordre', 'Temps', 'Processus', 'Noeuds', 'Solution', 'Longueur']

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)

    plt.title('Resume des resultats - Romeo HPC Cluster', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    if save:
        filepath = PLOTS_DIR / 'romeo_summary_table.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_nodes_comparison(seq_df, mpi_df, save=True):
    """Compare nodes explored between sequential and MPI."""
    if seq_df.empty or mpi_df.empty:
        print("Insufficient data for nodes comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    common_orders = set(seq_df['order'].unique()) & set(mpi_df['order'].unique())

    if not common_orders:
        return

    orders = sorted(common_orders)
    x = np.arange(len(orders))
    width = 0.35

    seq_nodes = []
    mpi_nodes = []

    for order in orders:
        seq_n = seq_df[seq_df['order'] == order]['nodes_explored'].values
        mpi_n = mpi_df[mpi_df['order'] == order]['nodes'].values
        seq_nodes.append(seq_n[0] if len(seq_n) > 0 else 0)
        mpi_nodes.append(mpi_n[0] if len(mpi_n) > 0 else 0)

    bars1 = ax.bar(x - width/2, seq_nodes, width, label='Sequentiel', color='#377eb8', edgecolor='black')
    bars2 = ax.bar(x + width/2, mpi_nodes, width, label='MPI', color='#4daf4a', edgecolor='black')

    ax.set_xlabel('Ordre de Golomb', fontsize=12)
    ax.set_ylabel('Noeuds explores', fontsize=12)
    ax.set_title('Comparaison du nombre de noeuds explores', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{o}' for o in orders])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis with millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else str(int(x))))

    plt.tight_layout()
    if save:
        filepath = PLOTS_DIR / 'romeo_nodes_comparison.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_architecture_info(save=True):
    """Create an info graphic about Romeo architecture used."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    info_text = """
    Romeo HPC Cluster - Universite de Reims
    =======================================

    Architecture: x64cpu (AMD EPYC Zen4)
    Compilateur: GCC 11.4.1
    MPI: Open MPI 4.1.7
    Gestionnaire: SLURM + Spack 1.0.1

    Configuration des tests:
    ------------------------
    - Sequentiel: 1 coeur, 8GB RAM
    - MPI: 8 processus, 1 noeud
    - Partition: instant (jobs < 1h)

    Optimisations actives:
    ---------------------
    - Branch & Bound avec elagage
    - Bitset pour lookup O(1)
    - Brisure de symetrie
    - Heuristique greedy pour borne superieure
    """

    ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#2E86AB', linewidth=2))

    plt.title('Configuration Romeo HPC', fontsize=16, fontweight='bold')

    plt.tight_layout()
    if save:
        filepath = PLOTS_DIR / 'romeo_architecture.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def main():
    print("=" * 60)
    print("Golomb Ruler Solver - Romeo Results Visualization")
    print("=" * 60)
    print(f"\nResults directory: {ROMEO_DIR}")
    print(f"Plots directory: {PLOTS_DIR}")

    # Load data
    seq_df, mpi_df = load_romeo_results()

    print(f"\nSequential results: {len(seq_df)} rows")
    print(f"MPI results: {len(mpi_df)} rows")

    if seq_df.empty and mpi_df.empty:
        print("\nNo results found! Make sure CSV files are in results/romeo/")
        return

    print("\n--- Generating plots ---")

    # Generate all plots
    plot_summary_table(seq_df, mpi_df)
    plot_sequential_vs_mpi(seq_df, mpi_df)
    plot_speedup(seq_df, mpi_df)
    plot_nodes_comparison(seq_df, mpi_df)
    plot_architecture_info()

    # Generate Golomb ruler visualization for each solution
    all_results = pd.concat([seq_df, mpi_df], ignore_index=True) if not seq_df.empty or not mpi_df.empty else pd.DataFrame()

    if not all_results.empty and 'solution' in all_results.columns:
        for order in all_results['order'].unique():
            row = all_results[all_results['order'] == order].iloc[0]
            if 'solution' in row and pd.notna(row['solution']):
                print(f"Generating Golomb ruler visualization for G{int(order)}...")
                plot_golomb_ruler(row['solution'], int(order))

    print("\n" + "=" * 60)
    print(f"All plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
