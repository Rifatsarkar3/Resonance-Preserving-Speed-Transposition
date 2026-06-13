"""Generate MSSP paper figures -> paper/figures/*.pdf

Figures:
  1. fig_jnu_comparison.pdf  - grouped bars: all models x JNU tasks (+overall)
  2. fig_ablation.pdf        - ablation progression on JNU T03
  3. fig_speed_gap.pdf       - raw signal + envelope spectrum at 600 vs 1000 rpm
  4. fig_latency.pdf         - inference latency vs JNU overall accuracy

Re-run after any results update; reads outputs/full_comparison_results.json.
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import Config

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D", "WaPIGT-MS"]
MODEL_LABEL = {"WaPIGT-MS": "WaPIGT (ours)", "ViT1D": "ViT-1D"}
COLORS = {"WDCNN": "#95a5a6", "TICNN": "#3498db", "MSCNN": "#9b59b6",
          "PhysFormer": "#e67e22", "ViT1D": "#1abc9c", "WaPIGT-MS": "#c0392b"}

# measured on RTX 5070, 200 CUDA-synchronized runs (batch=1)
LATENCY_MS = {"WDCNN": 2.60, "TICNN": 9.43, "MSCNN": 5.04,
              "PhysFormer": 3.92, "ViT1D": 2.49, "WaPIGT-MS": 34.76}

# ablation_quick.py final numbers (JNU T03, 3 seeds)
ABLATION = [
    ("Base\nTransformer", 59.2, 4.5),
    ("+MST", 89.0, 5.5),
    ("+PIFFG", 95.8, 1.4),
    ("+SCR", 95.5, 1.1),
    ("+Triplet\n(full)", 93.7, 1.0),
]


def load_jnu_results(path="outputs/full_comparison_results.json",
                     override_wapigt=None):
    """Returns {model: {task: (mean, std)}} for JNU tasks.
    override_wapigt: optional {task: [accs]} replacing WaPIGT-MS entries."""
    d = json.load(open(path))
    accs = {}
    for r in d["runs"]:
        if r.get("status") == "success" and r["task"].startswith("JNU"):
            accs.setdefault(r["model"], {}).setdefault(r["task"], []).append(r["test_acc"])
    if override_wapigt:
        accs["WaPIGT-MS"] = override_wapigt
    out = {}
    for m, tasks in accs.items():
        out[m] = {t: (float(np.mean(a)) * 100, float(np.std(a)) * 100)
                  for t, a in tasks.items()}
    return out


def load_rpst_results(wapigt_path="outputs/tsm_aug_results.json",
                      baselines_path="outputs/tsm_baselines_results.json"):
    """Returns {model: {task: (mean, std)}} for the +RPST regime."""
    accs = {}
    for r in json.load(open(wapigt_path))["runs"]:
        if r.get("status") == "success":
            accs.setdefault("WaPIGT-MS", {}).setdefault(r["task"], []).append(r["test_acc"])
    for r in json.load(open(baselines_path))["runs"]:
        if r.get("status") == "success":
            m = r["model"].replace("+TSM", "")
            accs.setdefault(m, {}).setdefault(r["task"], []).append(r["test_acc"])
    return {m: {t: (float(np.mean(a)) * 100, float(np.std(a)) * 100)
                for t, a in tasks.items()} for m, tasks in accs.items()}


def _comparison_panel(ax, results, title, show_legend):
    tasks = ["JNU_T01", "JNU_T02", "JNU_T03"]
    task_labels = ["T01", "T02", "T03", "Overall"]
    n_m = len(MODEL_ORDER)
    width = 0.8 / n_m
    x = np.arange(4)
    for j, m in enumerate(MODEL_ORDER):
        if m not in results:
            continue
        means = [results[m][t][0] for t in tasks]
        stds = [results[m][t][1] for t in tasks]
        means.append(float(np.mean(means)))
        stds.append(0.0)
        pos = x + (j - (n_m - 1) / 2) * width
        ax.bar(pos, means, width * 0.92, yerr=stds, capsize=2,
               color=COLORS[m], edgecolor="black", linewidth=0.4,
               label=MODEL_LABEL.get(m, m), error_kw=dict(lw=0.7))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylim(50, 103)
    if show_legend:
        ax.legend(ncol=3, loc="lower left", framealpha=0.9, fontsize=7)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)


def fig_jnu_comparison(results, rpst_results=None):
    if rpst_results:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
        _comparison_panel(axes[0], results, "(a) Standard protocol (no augmentation)", True)
        _comparison_panel(axes[1], rpst_results, "(b) +RPST (all models)", False)
    else:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        _comparison_panel(ax, results, "", True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_jnu_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_jnu_comparison.pdf")


def fig_ablation():
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    labels = [a[0] for a in ABLATION]
    means = [a[1] for a in ABLATION]
    stds = [a[2] for a in ABLATION]
    colors = ["#95a5a6", "#3498db", "#2980b9", "#1f618d", "#c0392b"]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=3,
           color=colors, edgecolor="black", linewidth=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1.2, f"{m:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("JNU T03 accuracy (%)")
    ax.set_ylim(40, 105)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_ablation.pdf")


def fig_speed_gap():
    config = Config.from_yaml("config.yaml")
    root = Path(config.data.jnu_raw_root) / "JNU"
    fs = 50000.0
    # inner-race fault at the two extreme speeds
    files = {}
    for f in sorted(root.glob("*.csv")):
        p = f.stem.split("_")[0].lower()
        if p.startswith("ib600"):
            files["600 rpm"] = f
        elif p.startswith("ib1000"):
            files["1000 rpm"] = f
    if len(files) < 2:
        print("speed-gap: JNU ib files not found, skipping")
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6))
    for col, (name, f) in enumerate(files.items()):
        sig = pd.read_csv(f).iloc[:, 0].values.astype(np.float32)[:12000]
        t = np.arange(len(sig)) / fs * 1000
        axes[0, col].plot(t, sig, lw=0.3, color="#2c3e50")
        axes[0, col].set_title(f"Inner-race fault, {name}")
        axes[0, col].set_xlabel("Time (ms)")
        axes[0, col].set_ylabel("Amplitude")
        env = np.abs(hilbert(sig - sig.mean()))
        spec = np.abs(np.fft.rfft(env - env.mean()))
        freqs = np.fft.rfftfreq(len(env), 1 / fs)
        mask = freqs <= 400
        axes[1, col].plot(freqs[mask], spec[mask] / spec[mask].max(),
                          lw=0.6, color="#c0392b")
        # BPFI order for JNU ER-16K (N=8, d=7.5, D=38.5): 4*(1+d/D) = 4.78
        f_shaft = 10.0 if "600" in name else 16.67
        bpfi = 4.78 * f_shaft
        axes[1, col].axvline(bpfi, color="#3498db", ls="--", lw=0.8,
                             label=f"BPFI = {bpfi:.0f} Hz")
        axes[1, col].set_xlabel("Frequency (Hz)")
        axes[1, col].set_ylabel("Norm. envelope spectrum")
        axes[1, col].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_speed_gap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_speed_gap.pdf")


def fig_latency(results):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in MODEL_ORDER:
        if m not in results:
            continue
        overall = np.mean([results[m][t][0] for t in
                           ["JNU_T01", "JNU_T02", "JNU_T03"]])
        ax.scatter(LATENCY_MS[m], overall, s=60, color=COLORS[m],
                   edgecolor="black", linewidth=0.5, zorder=3)
        dx, dy = 1.02, 0
        ax.annotate(MODEL_LABEL.get(m, m), (LATENCY_MS[m], overall),
                    xytext=(LATENCY_MS[m] * dx + 0.5, overall + dy),
                    fontsize=8)
    ax.set_xlabel("Inference latency (ms, batch=1, RTX 5070)")
    ax.set_ylabel("JNU mean accuracy (%)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_latency.pdf")


if __name__ == "__main__":
    res = load_jnu_results()
    try:
        rpst = load_rpst_results()
    except FileNotFoundError:
        rpst = None
    fig_jnu_comparison(res, rpst)
    fig_ablation()
    fig_speed_gap()
    fig_latency(res)
    print("Done ->", OUT)
