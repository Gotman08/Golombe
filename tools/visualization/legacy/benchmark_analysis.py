#!/usr/bin/env python3
"""
Golomb Ruler Solver - Comprehensive Benchmark Analysis

Generates professional visualizations showing optimization evolution
and platform comparison (PC Local vs Romeo HPC).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from datetime import datetime

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_DIR / 'results' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# BENCHMARK DATA - PC Local (extracted from benchmark_raw.txt)
# =============================================================================

# Sequential versions - Time in ms
SEQ_DATA = {
    'v1': {  # Brute Force
        7: {'time': 5290.12, 'nodes': 15890700, 'pruned': 0},
        8: {'time': 372040.85, 'nodes': 969443904, 'pruned': 0},
    },
    'v2': {  # Backtracking
        7: {'time': 0.38, 'nodes': 7178, 'pruned': 0},
        8: {'time': 3.31, 'nodes': 61865, 'pruned': 0},
        9: {'time': 34.94, 'nodes': 512355, 'pruned': 0},
        10: {'time': 284.81, 'nodes': 4217620, 'pruned': 0},
    },
    'v3': {  # Branch & Bound
        7: {'time': 0.42, 'nodes': 4341, 'pruned': 4709, 'ratio': 52.0},
        8: {'time': 2.47, 'nodes': 37209, 'pruned': 52922, 'ratio': 58.7},
        9: {'time': 22.07, 'nodes': 303757, 'pruned': 546926, 'ratio': 64.3},
        10: {'time': 195.03, 'nodes': 2466476, 'pruned': 5398765, 'ratio': 68.6},
    },
    'v4': {  # Optimized (Bitset + Symmetry)
        7: {'time': 0.26, 'nodes': 4341, 'pruned': 4709, 'ratio': 52.0},
        8: {'time': 2.14, 'nodes': 37209, 'pruned': 52922, 'ratio': 58.7},
        9: {'time': 27.84, 'nodes': 303757, 'pruned': 546926, 'ratio': 64.3},
        10: {'time': 205.22, 'nodes': 2466476, 'pruned': 5398765, 'ratio': 68.6},
    },
    'v5': {  # Final Sequential
        7: {'time': 0.21, 'nodes': 4341, 'pruned': 4709, 'ratio': 52.0},
        8: {'time': 2.38, 'nodes': 37209, 'pruned': 52922, 'ratio': 58.7},
        9: {'time': 20.52, 'nodes': 303757, 'pruned': 546926, 'ratio': 64.3},
        10: {'time': 200.66, 'nodes': 2466476, 'pruned': 5398765, 'ratio': 68.6},
    },
}

# V6 OpenMP data - Time in ms
V6_DATA = {
    1: {7: 0.83, 8: 3.06, 9: 23.20, 10: 225.97},
    2: {7: 0.56, 8: 2.01, 9: 14.71, 10: 102.61},
    4: {7: 0.68, 8: 1.14, 9: 6.55, 10: 69.29},
    8: {7: 16.27, 8: 1.61, 9: 6.05, 10: 43.05},
}

# MPI data - Time in ms for G10
MPI_DATA = {
    'v1': {2: 253.55, 4: 90.40, 8: 160.28},
    'v2': {2: 264.02, 4: 101.95, 8: 89.41},
    'v3': {2: 270.94, 4: 108.54, 8: 178.94},
}

# Romeo HPC data
ROMEO_DATA = {
    'seq': {
        5: {'time': 1.31, 'nodes': 7315},
        10: {'time': 152.36, 'nodes': 2466476},
    },
    'mpi': {
        10: {8: 52.26},
    }
}

# Version descriptions
VERSION_INFO = {
    'v1': ('Brute Force', '#e41a1c', 'Enumeration exhaustive'),
    'v2': ('Backtracking', '#377eb8', 'Retour arriere + terminaison precoce'),
    'v3': ('Branch & Bound', '#4daf4a', 'Elagage + borne superieure greedy'),
    'v4': ('Optimized', '#984ea3', 'Bitset O(1) + brisure symetrie'),
    'v5': ('Final', '#ff7f00', 'Version production'),
    'v6': ('Hardware', '#a65628', 'OpenMP + AVX2 SIMD'),
}


def plot_version_evolution():
    """Plot the evolution of optimization techniques."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    orders = [7, 8, 9, 10]
    versions = ['v1', 'v2', 'v3', 'v4', 'v5']

    # Top left: Time comparison (bar chart)
    ax1 = axes[0, 0]
    x = np.arange(len(orders))
    width = 0.15

    for i, v in enumerate(versions):
        times = []
        for o in orders:
            if o in SEQ_DATA[v]:
                times.append(SEQ_DATA[v][o]['time'])
            else:
                times.append(np.nan)

        color = VERSION_INFO[v][1]
        bars = ax1.bar(x + i * width, times, width, label=VERSION_INFO[v][0],
                      color=color, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Ordre de Golomb')
    ax1.set_ylabel('Temps (ms) - Echelle log')
    ax1.set_title('Evolution des temps d\'execution par version')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels([f'G{o}' for o in orders])
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # Top right: Speedup vs v1
    ax2 = axes[0, 1]

    # Only for orders where v1 exists
    orders_v1 = [7, 8]
    x = np.arange(len(orders_v1))
    width = 0.2

    for i, v in enumerate(['v2', 'v3', 'v4', 'v5']):
        speedups = []
        for o in orders_v1:
            v1_time = SEQ_DATA['v1'][o]['time']
            v_time = SEQ_DATA[v][o]['time']
            speedups.append(v1_time / v_time)

        color = VERSION_INFO[v][1]
        bars = ax2.bar(x + i * width, speedups, width, label=VERSION_INFO[v][0],
                      color=color, edgecolor='black', linewidth=0.5)

        # Add value labels
        for j, (bar, s) in enumerate(zip(bars, speedups)):
            ax2.annotate(f'{s:.0f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center',
                        fontsize=8, fontweight='bold')

    ax2.set_xlabel('Ordre de Golomb')
    ax2.set_ylabel('Speedup vs v1 (Brute Force)')
    ax2.set_title('Acceleration par rapport a la version brute force')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f'G{o}' for o in orders_v1])
    ax2.set_yscale('log')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    # Bottom left: Nodes explored
    ax3 = axes[1, 0]

    for v in ['v1', 'v2', 'v3']:
        times = []
        nodes = []
        for o in sorted(SEQ_DATA[v].keys()):
            times.append(o)
            nodes.append(SEQ_DATA[v][o]['nodes'])

        color = VERSION_INFO[v][1]
        ax3.semilogy(times, nodes, 'o-', color=color, linewidth=2, markersize=10,
                    label=VERSION_INFO[v][0])

    ax3.set_xlabel('Ordre de Golomb')
    ax3.set_ylabel('Noeuds explores (echelle log)')
    ax3.set_title('Reduction du nombre de noeuds explores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom right: Pruning ratio
    ax4 = axes[1, 1]

    orders_bb = [7, 8, 9, 10]
    ratios = [SEQ_DATA['v3'][o]['ratio'] for o in orders_bb]

    bars = ax4.bar(range(len(orders_bb)), ratios, color='#4daf4a', edgecolor='black')

    for i, (bar, r) in enumerate(zip(bars, ratios)):
        ax4.annotate(f'{r:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold')

    ax4.set_xlabel('Ordre de Golomb')
    ax4.set_ylabel('Ratio d\'elagage (%)')
    ax4.set_title('Efficacite du Branch & Bound (v3)')
    ax4.set_xticks(range(len(orders_bb)))
    ax4.set_xticklabels([f'G{o}' for o in orders_bb])
    ax4.set_ylim(0, 100)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = OUTPUT_DIR / '01_version_evolution.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_optimization_impact():
    """Show the cumulative impact of each optimization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use G8 for demonstration (has v1 data)
    order = 8

    versions = ['v1', 'v2', 'v3', 'v4', 'v5']
    times = [SEQ_DATA[v][order]['time'] for v in versions]

    techniques = [
        'v1: Brute Force\n(Baseline)',
        'v2: Backtracking\n(+Early termination)',
        'v3: Branch & Bound\n(+Greedy pruning)',
        'v4: Optimized\n(+Bitset O(1))',
        'v5: Final\n(+Production ready)'
    ]

    colors = [VERSION_INFO[v][1] for v in versions]

    # Bar chart
    bars = ax.bar(range(len(versions)), times, color=colors, edgecolor='black', linewidth=2)

    # Add value labels and speedups
    v1_time = times[0]
    for i, (bar, t) in enumerate(zip(bars, times)):
        speedup = v1_time / t

        # Time label
        if t >= 1000:
            label = f'{t/1000:.1f}s'
        else:
            label = f'{t:.2f}ms'

        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center',
                   fontsize=11, fontweight='bold')

        # Speedup label (except v1)
        if i > 0:
            ax.annotate(f'{speedup:.0f}x faster',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()/2),
                       ha='center', fontsize=10, color='white', fontweight='bold')

    ax.set_xlabel('Version et technique d\'optimisation', fontsize=12)
    ax.set_ylabel('Temps d\'execution (ms) - Echelle log', fontsize=12)
    ax.set_title(f'Impact cumulatif des optimisations sur G{order}\n'
                f'Speedup total: {v1_time/times[-1]:,.0f}x', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(techniques, fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add arrows showing improvement
    for i in range(len(versions) - 1):
        improvement = times[i] / times[i+1]
        if improvement > 1:
            ax.annotate('', xy=(i+1, times[i+1]), xytext=(i, times[i]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))

    plt.tight_layout()
    filepath = OUTPUT_DIR / '02_optimization_impact.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_openmp_scaling():
    """Plot OpenMP scaling for v6."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    threads = [1, 2, 4, 8]
    orders = [8, 9, 10]
    colors = ['#377eb8', '#4daf4a', '#e41a1c']

    # Left: Time vs threads
    ax1 = axes[0]

    for i, order in enumerate(orders):
        times = [V6_DATA[t][order] for t in threads]
        ax1.plot(threads, times, 'o-', color=colors[i], linewidth=2, markersize=10,
                label=f'G{order}')

    ax1.set_xlabel('Nombre de threads OpenMP')
    ax1.set_ylabel('Temps (ms)')
    ax1.set_title('Temps d\'execution v6 vs nombre de threads')
    ax1.set_xticks(threads)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Speedup and efficiency
    ax2 = axes[1]

    for i, order in enumerate(orders):
        base_time = V6_DATA[1][order]
        speedups = [base_time / V6_DATA[t][order] for t in threads]
        ax2.plot(threads, speedups, 'o-', color=colors[i], linewidth=2, markersize=10,
                label=f'G{order}')

    # Ideal speedup
    ax2.plot(threads, threads, 'k--', linewidth=2, label='Ideal')

    ax2.set_xlabel('Nombre de threads OpenMP')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup OpenMP (v6)')
    ax2.set_xticks(threads)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 9)
    ax2.set_ylim(0, 9)

    plt.tight_layout()
    filepath = OUTPUT_DIR / '03_openmp_scaling.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_mpi_comparison():
    """Compare MPI versions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    procs = [2, 4, 8]
    mpi_versions = ['v1', 'v2', 'v3']
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    labels = ['MPI v1 (Basic)', 'MPI v2 (Hypercube)', 'MPI v3 (Optimized)']

    # Left: Time comparison
    ax1 = axes[0]

    x = np.arange(len(procs))
    width = 0.25

    for i, (v, label, color) in enumerate(zip(mpi_versions, labels, colors)):
        times = [MPI_DATA[v][p] for p in procs]
        bars = ax1.bar(x + i * width, times, width, label=label, color=color, edgecolor='black')

        for bar, t in zip(bars, times):
            ax1.annotate(f'{t:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    # Add sequential baseline
    seq_time = SEQ_DATA['v5'][10]['time']
    ax1.axhline(y=seq_time, color='gray', linestyle='--', linewidth=2,
               label=f'Sequentiel v5 ({seq_time:.0f}ms)')

    ax1.set_xlabel('Nombre de processus MPI')
    ax1.set_ylabel('Temps (ms)')
    ax1.set_title('Comparaison des versions MPI (G10)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([str(p) for p in procs])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Speedup comparison
    ax2 = axes[1]

    for i, (v, label, color) in enumerate(zip(mpi_versions, labels, colors)):
        speedups = [seq_time / MPI_DATA[v][p] for p in procs]
        ax2.plot(procs, speedups, 'o-', color=color, linewidth=2, markersize=10, label=label)

    # Ideal speedup
    ax2.plot(procs, procs, 'k--', linewidth=2, label='Ideal')

    ax2.set_xlabel('Nombre de processus MPI')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup MPI vs Sequentiel (G10)')
    ax2.set_xticks(procs)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_DIR / '04_mpi_comparison.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_pc_vs_romeo():
    """Compare PC Local vs Romeo HPC."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Sequential G10 comparison
    ax1 = axes[0]

    pc_time = SEQ_DATA['v5'][10]['time']
    romeo_time = ROMEO_DATA['seq'][10]['time']

    bars = ax1.bar(['PC Local\n(WSL2)', 'Romeo HPC\n(AMD EPYC)'],
                   [pc_time, romeo_time],
                   color=['#377eb8', '#e41a1c'], edgecolor='black', linewidth=2)

    for bar, t in zip(bars, [pc_time, romeo_time]):
        ax1.annotate(f'{t:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center',
                    fontsize=12, fontweight='bold')

    speedup = pc_time / romeo_time
    ax1.set_title(f'Sequentiel v5 - G10\nRomeo {speedup:.1f}x plus rapide', fontsize=14)
    ax1.set_ylabel('Temps (ms)')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: MPI G10 comparison
    ax2 = axes[1]

    pc_mpi = MPI_DATA['v3'][8]
    romeo_mpi = ROMEO_DATA['mpi'][10][8]

    bars = ax2.bar(['PC Local\n(8 procs)', 'Romeo HPC\n(8 procs)'],
                   [pc_mpi, romeo_mpi],
                   color=['#377eb8', '#e41a1c'], edgecolor='black', linewidth=2)

    for bar, t in zip(bars, [pc_mpi, romeo_mpi]):
        ax2.annotate(f'{t:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center',
                    fontsize=12, fontweight='bold')

    speedup = pc_mpi / romeo_mpi
    ax2.set_title(f'MPI v3 - G10 (8 processus)\nRomeo {speedup:.1f}x plus rapide', fontsize=14)
    ax2.set_ylabel('Temps (ms)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = OUTPUT_DIR / '05_pc_vs_romeo.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_summary_table():
    """Create comprehensive summary table."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Prepare data
    rows = []

    # Sequential versions for G8
    v1_time = SEQ_DATA['v1'][8]['time']
    for v in ['v1', 'v2', 'v3', 'v4', 'v5']:
        if 8 in SEQ_DATA[v]:
            d = SEQ_DATA[v][8]
            time = d['time']
            speedup = v1_time / time
            nodes = d['nodes']
            rows.append([
                VERSION_INFO[v][0],
                f'{time:.2f} ms' if time < 1000 else f'{time/1000:.1f} s',
                f'{speedup:,.0f}x',
                f'{nodes:,}',
                VERSION_INFO[v][2]
            ])

    # V6 best (4 threads)
    v6_time = V6_DATA[4][8]
    rows.append([
        'v6 (4 threads)',
        f'{v6_time:.2f} ms',
        f'{v1_time/v6_time:,.0f}x',
        '~37,000',
        'OpenMP parallel + AVX2 SIMD'
    ])

    # MPI best
    mpi_time = min(MPI_DATA['v2'][p] for p in [2, 4, 8])
    mpi_procs = [p for p in [2, 4, 8] if MPI_DATA['v2'][p] == mpi_time][0]
    seq_time = SEQ_DATA['v5'][10]['time']
    rows.append([
        f'MPI v2 ({mpi_procs}p)',
        f'{mpi_time:.2f} ms',
        f'{seq_time/mpi_time:.2f}x vs seq',
        '~2.5M',
        'Hypercube topology'
    ])

    headers = ['Version', 'Temps (G8)', 'Speedup', 'Noeuds', 'Technique']

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.2)

    # Style
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Highlight best sequential
    for i in range(1, len(rows) + 1):
        if 'v5' in rows[i-1][0] or 'v6' in rows[i-1][0]:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#ccffcc')

    plt.title('Resume des performances - Toutes versions\n'
             f'Speedup total v1->v6: {v1_time/v6_time:,.0f}x sur G8',
             fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    filepath = OUTPUT_DIR / '06_summary_table.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_scalability():
    """Plot scalability analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Time vs Order for v5
    ax1 = axes[0]

    orders = [7, 8, 9, 10]
    times = [SEQ_DATA['v5'][o]['time'] for o in orders]

    ax1.semilogy(orders, times, 'o-', color='#ff7f00', linewidth=2, markersize=12)

    # Fit exponential
    coeffs = np.polyfit(orders, np.log(times), 1)
    x_fit = np.linspace(7, 10, 100)
    y_fit = np.exp(coeffs[1]) * np.exp(coeffs[0] * x_fit)
    ax1.semilogy(x_fit, y_fit, '--', color='gray', linewidth=1.5,
                label=f'Tendance: O(e^{{{coeffs[0]:.2f}n}})')

    # Add value labels
    for o, t in zip(orders, times):
        label = f'{t:.1f}ms' if t < 1000 else f'{t/1000:.1f}s'
        ax1.annotate(label, xy=(o, t), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    ax1.set_xlabel('Ordre de Golomb (n)')
    ax1.set_ylabel('Temps (ms) - Echelle log')
    ax1.set_title('Scalabilite v5 - Temps vs Ordre')
    ax1.set_xticks(orders)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Growth factor
    ax2 = axes[1]

    growth = [times[i+1] / times[i] for i in range(len(times)-1)]

    bars = ax2.bar(range(len(growth)), growth, color='#984ea3', edgecolor='black')

    for i, (bar, g) in enumerate(zip(bars, growth)):
        ax2.annotate(f'{g:.1f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=12, fontweight='bold')

    ax2.axhline(y=np.mean(growth), color='red', linestyle='--', linewidth=2,
               label=f'Moyenne: {np.mean(growth):.1f}x')

    ax2.set_xlabel('Transition d\'ordre')
    ax2.set_ylabel('Facteur de croissance')
    ax2.set_title('Facteur de croissance entre ordres consecutifs')
    ax2.set_xticks(range(len(growth)))
    ax2.set_xticklabels([f'G{orders[i]}->G{orders[i+1]}' for i in range(len(growth))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = OUTPUT_DIR / '07_scalability.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_final_comparison():
    """Create final comprehensive comparison."""
    fig = plt.figure(figsize=(18, 12))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: V1 vs V5 speedup per order
    ax1 = fig.add_subplot(gs[0, 0])
    orders = [7, 8]
    speedups = [SEQ_DATA['v1'][o]['time'] / SEQ_DATA['v5'][o]['time'] for o in orders]

    bars = ax1.bar(range(len(orders)), speedups, color=['#4daf4a', '#377eb8'], edgecolor='black')
    for i, (bar, s) in enumerate(zip(bars, speedups)):
        ax1.annotate(f'{s:,.0f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center',
                    fontsize=14, fontweight='bold')

    ax1.set_title('Speedup v1->v5', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(orders)))
    ax1.set_xticklabels([f'G{o}' for o in orders])
    ax1.set_ylabel('Speedup')
    ax1.set_yscale('log')

    # Plot 2: Best times per platform
    ax2 = fig.add_subplot(gs[0, 1])

    pc_seq = SEQ_DATA['v5'][10]['time']
    pc_omp = V6_DATA[4][10]
    pc_mpi = min(MPI_DATA['v2'].values())
    romeo_seq = ROMEO_DATA['seq'][10]['time']
    romeo_mpi = ROMEO_DATA['mpi'][10][8]

    x = np.arange(3)
    width = 0.35

    pc_vals = [pc_seq, pc_omp, pc_mpi]
    romeo_vals = [romeo_seq, romeo_seq, romeo_mpi]  # Use seq for v6 comparison

    ax2.bar(x - width/2, pc_vals, width, label='PC Local', color='#377eb8', edgecolor='black')
    ax2.bar(x + width/2, romeo_vals, width, label='Romeo HPC', color='#e41a1c', edgecolor='black')

    ax2.set_title('Meilleurs temps G10 par configuration', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Seq v5', 'OpenMP v6', 'MPI best'])
    ax2.set_ylabel('Temps (ms)')
    ax2.legend()

    # Plot 3: Nodes reduction
    ax3 = fig.add_subplot(gs[0, 2])

    v1_nodes = SEQ_DATA['v1'][8]['nodes']
    v3_nodes = SEQ_DATA['v3'][8]['nodes']
    reduction = (1 - v3_nodes/v1_nodes) * 100

    ax3.pie([v3_nodes, v1_nodes - v3_nodes],
            labels=[f'Explores\n({v3_nodes:,})', f'Elagages\n({v1_nodes-v3_nodes:,})'],
            colors=['#4daf4a', '#e41a1c'],
            autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Reduction des noeuds (G8)\n{reduction:.1f}% economise', fontsize=12, fontweight='bold')

    # Plot 4: Timeline of optimizations
    ax4 = fig.add_subplot(gs[1, :])

    versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6\n(4T)', 'MPI\nv2']
    times_g8 = [
        SEQ_DATA['v1'][8]['time'],
        SEQ_DATA['v2'][8]['time'],
        SEQ_DATA['v3'][8]['time'],
        SEQ_DATA['v4'][8]['time'],
        SEQ_DATA['v5'][8]['time'],
        V6_DATA[4][8],
        min(MPI_DATA['v2'].values()) * SEQ_DATA['v5'][8]['time'] / SEQ_DATA['v5'][10]['time']  # Estimate
    ]

    colors = [VERSION_INFO.get(f'v{i+1}', ('', '#999999'))[1] for i in range(5)]
    colors.extend(['#a65628', '#ff00ff'])

    bars = ax4.bar(range(len(versions)), times_g8, color=colors, edgecolor='black', linewidth=2)

    for i, (bar, t) in enumerate(zip(bars, times_g8)):
        label = f'{t:.1f}ms' if t < 1000 else f'{t/1000:.0f}s'
        ax4.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')

    ax4.set_title('Evolution complete des performances (G8)\n'
                 f'Speedup total: {times_g8[0]/min(times_g8[1:]):,.0f}x',
                 fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(versions)))
    ax4.set_xticklabels(versions)
    ax4.set_ylabel('Temps (ms) - Echelle log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add technique annotations
    techniques = [
        'Baseline',
        '+Backtrack',
        '+B&B',
        '+Bitset',
        '+Polish',
        '+OpenMP',
        '+Distrib'
    ]
    for i, tech in enumerate(techniques):
        ax4.annotate(tech, xy=(i, 0), xytext=(0, -25), textcoords='offset points',
                    ha='center', fontsize=9, style='italic')

    plt.suptitle('Analyse Complete des Optimisations - Golomb Ruler Solver',
                fontsize=18, fontweight='bold', y=0.98)

    filepath = OUTPUT_DIR / '08_final_comparison.png'
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def main():
    print("=" * 70)
    print("Golomb Ruler Solver - Comprehensive Benchmark Analysis")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n--- Generating analysis plots ---\n")

    plot_version_evolution()
    plot_optimization_impact()
    plot_openmp_scaling()
    plot_mpi_comparison()
    plot_pc_vs_romeo()
    plot_summary_table()
    plot_scalability()
    plot_final_comparison()

    print("\n" + "=" * 70)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
