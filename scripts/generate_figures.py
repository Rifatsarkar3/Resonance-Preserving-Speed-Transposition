"""Generate publication figures from training results and ablations."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_dataset_performance(stats_path: str = "outputs/statistical_analysis.json",
                             output_path: str = "outputs/figures/dataset_performance.png"):
    """
    Plot dataset-level performance (accuracy ± std).
    """
    stats = load_json(stats_path)
    summary = stats['summary']
    by_dataset = stats['by_dataset']

    datasets = summary['datasets']
    means = [by_dataset[ds]['accuracy']['mean'] * 100 for ds in datasets]
    stds = [by_dataset[ds]['accuracy']['std'] * 100 for ds in datasets]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(datasets))

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['#2ecc71', '#e74c3c', '#3498db'], edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('WaPIGT Performance by Dataset (5-Seed Protocol)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 2, f'{mean:.1f}+/-{std:.1f}%', ha='center', fontsize=10, fontweight='bold')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Dataset performance figure saved to {output_path}")
    plt.close()


def plot_ablation_contributions(ablation_path: str = "outputs/metrics/ablation_results.json",
                                output_path: str = "outputs/figures/ablation_contributions.png"):
    """
    Plot ablation study contributions (component impact).
    """
    try:
        ablations = load_json(ablation_path)
    except FileNotFoundError:
        logger.warning(f"Ablation results not found at {ablation_path}")
        return

    tasks = []
    lwpt_impact = []
    piffg_impact = []
    scr_impact = []

    for task_name, task_ablations in sorted(ablations.items()):
        full_acc = task_ablations.get('full', {}).get('test', {}).get('accuracy')
        if full_acc is None:
            continue

        lwpt_acc = task_ablations.get('-LWPT', {}).get('test', {}).get('accuracy', full_acc)
        piffg_acc = task_ablations.get('-PIFFG', {}).get('test', {}).get('accuracy', full_acc)
        scr_acc = task_ablations.get('-SCR', {}).get('test', {}).get('accuracy', full_acc)

        tasks.append(task_name.split('_')[1])
        lwpt_impact.append((full_acc - lwpt_acc) * 100)
        piffg_impact.append((full_acc - piffg_acc) * 100)
        scr_impact.append((full_acc - scr_acc) * 100)

    if not tasks:
        logger.warning("No ablation results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(tasks))
    width = 0.25

    ax.bar(x_pos - width, lwpt_impact, width, label='Learnable Wavelets (LWPT)',
           color='#3498db', edgecolor='black', linewidth=1)
    ax.bar(x_pos, piffg_impact, width, label='Physics Graph (PIFFG)',
           color='#2ecc71', edgecolor='black', linewidth=1)
    ax.bar(x_pos + width, scr_impact, width, label='Consistency Reg. (SCR)',
           color='#e74c3c', edgecolor='black', linewidth=1)

    ax.set_ylabel('Accuracy Impact (pp)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_title('Component Contribution to Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tasks)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Ablation contributions figure saved to {output_path}")
    plt.close()


def plot_task_distribution(stats_path: str = "outputs/statistical_analysis.json",
                           output_path: str = "outputs/figures/task_distribution.png"):
    """
    Plot distribution of task accuracies within each dataset.
    """
    stats = load_json(stats_path)
    by_task = stats['by_task']

    pu_tasks = [t for t in by_task.keys() if t.startswith('PU')]
    cwru_tasks = [t for t in by_task.keys() if t.startswith('CWRU')]
    jnu_tasks = [t for t in by_task.keys() if t.startswith('JNU')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, tasks, dataset_name, color in zip(
        axes,
        [pu_tasks, cwru_tasks, jnu_tasks],
        ['PU', 'CWRU', 'JNU'],
        ['#2ecc71', '#e74c3c', '#3498db']
    ):
        accs = [by_task[t].get('accuracy', {}).get('mean', 0) * 100 for t in tasks]
        task_ids = [t.split('_')[1] for t in tasks]

        ax.bar(range(len(tasks)), accs, color=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_xlabel('Task', fontsize=11)
        ax.set_title(f'{dataset_name} Dataset ({len(tasks)} tasks)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(task_ids, rotation=45)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Task Accuracy Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Task distribution figure saved to {output_path}")
    plt.close()


def generate_all_figures(stats_path: str = "outputs/statistical_analysis.json",
                         ablation_path: str = "outputs/metrics/ablation_results.json"):
    """Generate all publication figures."""
    logger.info("Generating publication figures...")

    plot_dataset_performance(stats_path)
    plot_ablation_contributions(ablation_path)
    plot_task_distribution(stats_path)

    logger.info("\nFigures generated successfully!")
    logger.info("Outputs:")
    logger.info("  - outputs/figures/dataset_performance.png")
    logger.info("  - outputs/figures/ablation_contributions.png")
    logger.info("  - outputs/figures/task_distribution.png")


if __name__ == "__main__":
    generate_all_figures()
