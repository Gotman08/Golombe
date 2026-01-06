#!/usr/bin/env python3
"""
Golomb Ruler Solver - Platform Comparison Visualization

Compares performance between local PC and Romeo HPC cluster.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform
import subprocess
from pathlib import Path
from datetime import datetime

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / 'results'
LOCAL_DIR = RESULTS_DIR / 'local'
ROMEO_DIR = RESULTS_DIR / 'romeo'
OUTPUT_DIR = RESULTS_DIR / 'comparison'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Platform info
PLATFORMS = {
    'local': {
        'name': 'PC Local (WSL2)',
        'color': '#377eb8',
        'marker': 'o'
    },
    'romeo': {
        'name': 'Romeo HPC (AMD EPYC)',
        'color': '#e41a1c',
        'marker': 's'
    }
}


def get_local_cpu_info():
    """Get local CPU information."""
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        info = {}
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                info[key.strip()] = val.strip()
        return {
            'model': info.get('Model name', 'Unknown'),
            'cores': info.get('CPU(s)', 'Unknown'),
            'arch': info.get('Architecture', 'Unknown')
        }
    except:
        return {'model': 'Unknown', 'cores': 'Unknown', 'arch': 'Unknown'}


def load_data():
    """Load all benchmark data from both platforms."""
    data = {'local': {'seq': None, 'mpi': None}, 'romeo': {'seq': None, 'mpi': None}}

    # Local data
    local_seq = LOCAL_DIR / 'seq_v5_benchmark.csv'
    local_mpi = LOCAL_DIR / 'mpi_v3_benchmark.csv'

    if local_seq.exists():
        data['local']['seq'] = pd.read_csv(local_seq)
    if local_mpi.exists():
        data['local']['mpi'] = pd.read_csv(local_mpi)

    # Romeo data
    romeo_seq_files = list(ROMEO_DIR.glob('seq_*.csv'))
    romeo_mpi_files = list(ROMEO_DIR.glob('mpi_*.csv'))

    if romeo_seq_files:
        dfs = [pd.read_csv(f) for f in romeo_seq_files]
        data['romeo']['seq'] = pd.concat(dfs, ignore_index=True)
    if romeo_mpi_files:
        dfs = [pd.read_csv(f) for f in romeo_mpi_files]
        data['romeo']['mpi'] = pd.concat(dfs, ignore_index=True)

    return data


def plot_sequential_comparison(data, save=True):
    """Compare sequential performance between platforms."""
    local_seq = data['local']['seq']
    romeo_seq = data['romeo']['seq']

    if local_seq is None or romeo_seq is None:
        print("Insufficient sequential data for comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Find common orders
    common_orders = set(local_seq['order'].unique()) & set(romeo_seq['order'].unique())
    if not common_orders:
        print("No common orders found")
        return

    orders = sorted(common_orders)

    # Left plot: Time comparison (bar chart)
    ax1 = axes[0]
    x = np.arange(len(orders))
    width = 0.35

    local_times = [local_seq[local_seq['order'] == o]['time_ms'].values[0] for o in orders]
    romeo_times = [romeo_seq[romeo_seq['order'] == o]['time_ms'].values[0] for o in orders]

    bars1 = ax1.bar(x - width/2, local_times, width, label=PLATFORMS['local']['name'],
                    color=PLATFORMS['local']['color'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, romeo_times, width, label=PLATFORMS['romeo']['name'],
                    color=PLATFORMS['romeo']['color'], edgecolor='black')

    ax1.set_xlabel('Ordre de Golomb')
    ax1.set_ylabel('Temps (ms)')
    ax1.set_title('Temps d\'execution - Sequentiel v5')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'G{o}' for o in orders])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=8)

    # Right plot: Speedup (Romeo faster than PC)
    ax2 = axes[1]
    speedups = [l/r if r > 0 else 1 for l, r in zip(local_times, romeo_times)]

    colors = ['#4daf4a' if s > 1 else '#ff7f00' for s in speedups]
    bars = ax2.bar(range(len(orders)), speedups, color=colors, edgecolor='black')

    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Egalite')
    ax2.set_xlabel('Ordre de Golomb')
    ax2.set_ylabel('Speedup (PC Local / Romeo)')
    ax2.set_title('Romeo vs PC Local - Facteur d\'acceleration')
    ax2.set_xticks(range(len(orders)))
    ax2.set_xticklabels([f'G{o}' for o in orders])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotations
    for i, (bar, s) in enumerate(zip(bars, speedups)):
        label = f'{s:.2f}x'
        if s > 1:
            label += '\nRomeo+'
        else:
            label += '\nPC+'
        ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save:
        filepath = OUTPUT_DIR / 'sequential_comparison.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_mpi_comparison(data, save=True):
    """Compare MPI performance between platforms."""
    local_mpi = data['local']['mpi']
    romeo_mpi = data['romeo']['mpi']
    local_seq = data['local']['seq']

    if local_mpi is None:
        print("No local MPI data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get G10 sequential time as baseline
    if local_seq is not None:
        seq_time = local_seq[local_seq['order'] == 10]['time_ms'].values
        seq_time = seq_time[0] if len(seq_time) > 0 else 187.44
    else:
        seq_time = 187.44

    # Left plot: Time vs processes
    ax1 = axes[0]

    # Local MPI data
    local_procs = local_mpi['procs'].values
    local_times = local_mpi['time_ms'].values

    ax1.plot(local_procs, local_times, 'o-', color=PLATFORMS['local']['color'],
             linewidth=2, markersize=10, label=PLATFORMS['local']['name'])

    # Romeo MPI data (if available)
    if romeo_mpi is not None and len(romeo_mpi) > 0:
        romeo_procs = romeo_mpi['procs'].values
        romeo_times = romeo_mpi['time_ms'].values
        ax1.plot(romeo_procs, romeo_times, 's-', color=PLATFORMS['romeo']['color'],
                 linewidth=2, markersize=10, label=PLATFORMS['romeo']['name'])

    ax1.axhline(y=seq_time, color='gray', linestyle='--', linewidth=1.5,
                label=f'Sequentiel ({seq_time:.1f}ms)')

    ax1.set_xlabel('Nombre de processus MPI')
    ax1.set_ylabel('Temps (ms)')
    ax1.set_title('Temps d\'execution MPI v3 - G10')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(local_procs)

    # Add value labels
    for p, t in zip(local_procs, local_times):
        ax1.annotate(f'{t:.1f}ms', xy=(p, t), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    # Right plot: Speedup and efficiency
    ax2 = axes[1]

    speedups = [seq_time / t for t in local_times]
    efficiencies = [s / p * 100 for s, p in zip(speedups, local_procs)]

    # Create twin axis for efficiency
    ax2b = ax2.twinx()

    # Speedup bars
    x = np.arange(len(local_procs))
    bars = ax2.bar(x - 0.2, speedups, 0.4, color=PLATFORMS['local']['color'],
                   edgecolor='black', label='Speedup')

    # Efficiency bars
    bars2 = ax2.bar(x + 0.2, [e/100 * max(speedups) for e in efficiencies], 0.4,
                    color='#984ea3', edgecolor='black', alpha=0.7, label='Efficacite')

    # Ideal speedup line
    ax2.plot(x, local_procs, 'k--', linewidth=2, label='Speedup ideal')

    ax2.set_xlabel('Nombre de processus')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup et Efficacite MPI - G10')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(p) for p in local_procs])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotations
    for i, (s, e) in enumerate(zip(speedups, efficiencies)):
        ax2.annotate(f'{s:.2f}x\n({e:.0f}%)', xy=(i, s),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save:
        filepath = OUTPUT_DIR / 'mpi_comparison.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_scalability(data, save=True):
    """Plot scalability analysis."""
    local_seq = data['local']['seq']

    if local_seq is None:
        print("No sequential data for scalability plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    orders = local_seq['order'].values
    times = local_seq['time_ms'].values
    nodes = local_seq['nodes_explored'].values

    # Left plot: Time vs Order (log scale)
    ax1 = axes[0]
    ax1.semilogy(orders, times, 'o-', color=PLATFORMS['local']['color'],
                 linewidth=2, markersize=10, label='Temps mesure')

    # Fit exponential trend
    if len(orders) > 2:
        from scipy.optimize import curve_fit
        try:
            def exp_func(x, a, b):
                return a * np.exp(b * x)
            popt, _ = curve_fit(exp_func, orders, times, p0=[0.001, 1], maxfev=5000)
            x_fit = np.linspace(min(orders), max(orders), 100)
            y_fit = exp_func(x_fit, *popt)
            ax1.semilogy(x_fit, y_fit, '--', color='gray', linewidth=1.5,
                        label=f'Tendance: {popt[0]:.2e} * e^({popt[1]:.2f}*n)')
        except:
            pass

    ax1.set_xlabel('Ordre de Golomb (n)')
    ax1.set_ylabel('Temps (ms, echelle log)')
    ax1.set_title('Scalabilite - Temps vs Ordre')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(orders)

    # Right plot: Nodes explored vs Order
    ax2 = axes[1]
    ax2.semilogy(orders, nodes, 's-', color='#4daf4a',
                 linewidth=2, markersize=10)

    ax2.set_xlabel('Ordre de Golomb (n)')
    ax2.set_ylabel('Noeuds explores (echelle log)')
    ax2.set_title('Complexite - Noeuds explores vs Ordre')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(orders)

    # Add annotations
    for o, n in zip(orders, nodes):
        label = f'{n/1e6:.1f}M' if n >= 1e6 else f'{n/1e3:.0f}K' if n >= 1e3 else str(int(n))
        ax2.annotate(label, xy=(o, n), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    plt.tight_layout()
    if save:
        filepath = OUTPUT_DIR / 'scalability.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_summary_table(data, save=True):
    """Create comprehensive summary table."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Prepare data
    rows = []

    local_seq = data['local']['seq']
    romeo_seq = data['romeo']['seq']
    local_mpi = data['local']['mpi']
    romeo_mpi = data['romeo']['mpi']

    # Sequential comparison
    if local_seq is not None and romeo_seq is not None:
        common_orders = set(local_seq['order'].unique()) & set(romeo_seq['order'].unique())
        for order in sorted(common_orders):
            local_row = local_seq[local_seq['order'] == order].iloc[0]
            romeo_row = romeo_seq[romeo_seq['order'] == order].iloc[0]

            local_time = local_row['time_ms']
            romeo_time = romeo_row['time_ms']
            speedup = local_time / romeo_time if romeo_time > 0 else 0

            rows.append([
                f'G{order}',
                'Sequentiel v5',
                f'{local_time:.2f} ms',
                f'{romeo_time:.2f} ms',
                f'{speedup:.2f}x' if speedup > 1 else f'{1/speedup:.2f}x PC+',
                'Romeo' if speedup > 1 else 'PC Local',
                f"{int(local_row['nodes_explored']):,}"
            ])

    # MPI comparison
    if local_mpi is not None:
        for _, row in local_mpi.iterrows():
            procs = int(row['procs'])
            local_time = row['time_ms']

            # Find Romeo equivalent
            romeo_time = None
            if romeo_mpi is not None:
                romeo_match = romeo_mpi[romeo_mpi['procs'] == procs]
                if len(romeo_match) > 0:
                    romeo_time = romeo_match.iloc[0]['time_ms']

            if romeo_time:
                speedup = local_time / romeo_time
                winner = 'Romeo' if speedup > 1 else 'PC Local'
                speedup_str = f'{speedup:.2f}x' if speedup > 1 else f'{1/speedup:.2f}x PC+'
            else:
                speedup_str = '-'
                winner = '-'
                romeo_time = 0

            rows.append([
                f'G{int(row["order"])}',
                f'MPI v3 ({procs} procs)',
                f'{local_time:.2f} ms',
                f'{romeo_time:.2f} ms' if romeo_time else '-',
                speedup_str,
                winner,
                f"{int(row['nodes']):,}"
            ])

    if not rows:
        print("No data for summary table")
        return

    headers = ['Ordre', 'Version', 'PC Local', 'Romeo HPC', 'Speedup', 'Gagnant', 'Noeuds']

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', weight='bold')

    for i in range(1, len(rows) + 1):
        # Highlight winner
        winner = rows[i-1][5]
        if winner == 'Romeo':
            table[(i, 5)].set_facecolor('#ffcccc')
        elif winner == 'PC Local':
            table[(i, 5)].set_facecolor('#ccffcc')

        # Alternate row colors
        base_color = '#f8f8f8' if i % 2 == 0 else 'white'
        for j in [0, 1, 2, 3, 4, 6]:
            table[(i, j)].set_facecolor(base_color)

    plt.title('Comparaison des performances - PC Local vs Romeo HPC\n', fontsize=16, fontweight='bold')

    plt.tight_layout()
    if save:
        filepath = OUTPUT_DIR / 'summary_table.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_efficiency_analysis(data, save=True):
    """Detailed efficiency analysis for MPI."""
    local_mpi = data['local']['mpi']
    local_seq = data['local']['seq']

    if local_mpi is None or local_seq is None:
        print("Insufficient data for efficiency analysis")
        return

    # Get sequential baseline
    seq_time = local_seq[local_seq['order'] == 10]['time_ms'].values
    seq_time = seq_time[0] if len(seq_time) > 0 else 187.44

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    procs = local_mpi['procs'].values
    times = local_mpi['time_ms'].values
    speedups = [seq_time / t for t in times]
    efficiencies = [s / p * 100 for s, p in zip(speedups, procs)]

    # Top left: Speedup comparison with Amdahl's law
    ax1 = axes[0, 0]
    ax1.plot(procs, speedups, 'o-', color=PLATFORMS['local']['color'],
             linewidth=2, markersize=12, label='Mesure')
    ax1.plot(procs, procs, 'k--', linewidth=2, label='Ideal (lineaire)')

    # Amdahl's law for different parallel fractions
    p_range = np.linspace(1, max(procs), 50)
    for f in [0.95, 0.90, 0.80]:
        amdahl = 1 / ((1 - f) + f / p_range)
        ax1.plot(p_range, amdahl, '--', alpha=0.5, label=f'Amdahl (f={f:.0%})')

    ax1.set_xlabel('Nombre de processus')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Loi d\'Amdahl')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(procs) + 1)
    ax1.set_ylim(0, max(procs) + 1)

    # Top right: Efficiency
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(procs)), efficiencies,
                   color=['#4daf4a' if e >= 50 else '#ff7f00' for e in efficiencies],
                   edgecolor='black')

    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=2)
    ax2.axhline(y=50, color='red', linestyle=':', linewidth=1.5, label='Seuil 50%')

    ax2.set_xlabel('Nombre de processus')
    ax2.set_ylabel('Efficacite (%)')
    ax2.set_title('Efficacite parallele')
    ax2.set_xticks(range(len(procs)))
    ax2.set_xticklabels([str(p) for p in procs])
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (bar, e) in enumerate(zip(bars, efficiencies)):
        ax2.annotate(f'{e:.1f}%', xy=(i, e), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

    # Bottom left: Time breakdown
    ax3 = axes[1, 0]

    # Theoretical perfect scaling
    perfect_times = [seq_time / p for p in procs]
    overhead = [t - pt for t, pt in zip(times, perfect_times)]

    x = np.arange(len(procs))
    width = 0.6

    ax3.bar(x, perfect_times, width, label='Temps ideal', color='#4daf4a', edgecolor='black')
    ax3.bar(x, overhead, width, bottom=perfect_times, label='Overhead', color='#e41a1c', edgecolor='black')

    ax3.set_xlabel('Nombre de processus')
    ax3.set_ylabel('Temps (ms)')
    ax3.set_title('Decomposition du temps d\'execution')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(p) for p in procs])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Bottom right: Summary metrics
    ax4 = axes[1, 1]
    ax4.axis('off')

    cpu_info = get_local_cpu_info()

    summary_text = f"""
    === Resume de l'analyse MPI ===

    Plateforme: PC Local (WSL2)
    CPU: {cpu_info['model']}
    Coeurs: {cpu_info['cores']}

    Temps sequentiel (G10): {seq_time:.2f} ms

    === Resultats MPI ===
    """

    for p, t, s, e in zip(procs, times, speedups, efficiencies):
        summary_text += f"""
    {p} processus:
      - Temps: {t:.2f} ms
      - Speedup: {s:.2f}x
      - Efficacite: {e:.1f}%
        """

    # Estimate parallel fraction from best speedup
    best_idx = np.argmax(speedups)
    best_speedup = speedups[best_idx]
    best_procs = procs[best_idx]
    # From Amdahl: S = 1 / ((1-f) + f/p) => f = (1 - 1/S) / (1 - 1/p)
    if best_procs > 1 and best_speedup > 1:
        f_est = (1 - 1/best_speedup) / (1 - 1/best_procs)
        f_est = min(f_est, 1.0)
        summary_text += f"""
    === Estimation ===
    Fraction parallelisable: ~{f_est*100:.1f}%
    Speedup max theorique: {1/(1-f_est):.1f}x
        """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#2E86AB'))

    plt.tight_layout()
    if save:
        filepath = OUTPUT_DIR / 'efficiency_analysis.png'
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_architecture_comparison(save=True):
    """Compare architecture specifications."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    cpu_info = get_local_cpu_info()

    text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║                    COMPARAISON DES ARCHITECTURES                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                   ║
    ║  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐      ║
    ║  │         PC LOCAL (WSL2)         │    │         ROMEO HPC               │      ║
    ║  ├─────────────────────────────────┤    ├─────────────────────────────────┤      ║
    ║  │ CPU: {cpu_info['model'][:28]:<28} │    │ CPU: AMD EPYC (Zen4)           │      ║
    ║  │ Arch: {cpu_info['arch']:<27} │    │ Arch: x86_64                    │      ║
    ║  │ Coeurs: {cpu_info['cores']:<25} │    │ Coeurs: 128+ par noeud         │      ║
    ║  │ RAM: ~16 GB                     │    │ RAM: 256+ GB par noeud         │      ║
    ║  │ OS: WSL2 Linux                  │    │ OS: RHEL 9                      │      ║
    ║  │ MPI: Open MPI 4.1.6             │    │ MPI: Open MPI 4.1.7             │      ║
    ║  │ GCC: Local version              │    │ GCC: 11.4.1                     │      ║
    ║  └─────────────────────────────────┘    └─────────────────────────────────┘      ║
    ║                                                                                   ║
    ║  ┌───────────────────────────────────────────────────────────────────────────┐   ║
    ║  │                        CARACTERISTIQUES ROMEO                              │   ║
    ║  ├───────────────────────────────────────────────────────────────────────────┤   ║
    ║  │ • Architecture heterogene: x64cpu (AMD EPYC) + armgpu (ARM + NVIDIA H100) │   ║
    ║  │ • Gestionnaire de jobs: SLURM                                              │   ║
    ║  │ • Gestionnaire de packages: Spack 1.0.1                                    │   ║
    ║  │ • Stockage scratch haute performance: /scratch_p                           │   ║
    ║  │ • Reseau: InfiniBand haute vitesse                                         │   ║
    ║  │ • Partitions: instant (<1h), short (<8h), long (<72h)                      │   ║
    ║  └───────────────────────────────────────────────────────────────────────────┘   ║
    ║                                                                                   ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#16213e', linewidth=2),
            color='#00ff88')

    plt.title('Comparaison des Architectures', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    if save:
        filepath = OUTPUT_DIR / 'architecture_comparison.png'
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    plt.close()


def main():
    print("=" * 70)
    print("Golomb Ruler Solver - Platform Comparison")
    print("=" * 70)
    print(f"\nLocal results: {LOCAL_DIR}")
    print(f"Romeo results: {ROMEO_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    data = load_data()

    print(f"\n--- Data loaded ---")
    print(f"Local sequential: {len(data['local']['seq']) if data['local']['seq'] is not None else 0} rows")
    print(f"Local MPI: {len(data['local']['mpi']) if data['local']['mpi'] is not None else 0} rows")
    print(f"Romeo sequential: {len(data['romeo']['seq']) if data['romeo']['seq'] is not None else 0} rows")
    print(f"Romeo MPI: {len(data['romeo']['mpi']) if data['romeo']['mpi'] is not None else 0} rows")

    print("\n--- Generating plots ---")

    plot_architecture_comparison()
    plot_summary_table(data)
    plot_sequential_comparison(data)
    plot_mpi_comparison(data)
    plot_scalability(data)
    plot_efficiency_analysis(data)

    print("\n" + "=" * 70)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
