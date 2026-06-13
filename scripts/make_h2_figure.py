"""Generate fig_extrapolation.pdf from the H2 speed-extrapolation results.
Grouped bars: per model, the 4 arms (none / RPST-extrap / resampling-extrap / RPST-interp)
on the unseen 1000 rpm test, with std error bars. Visualises: RPST extrapolates + stabilises,
resampling drops below baseline."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})
OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["WaPIGT", "TICNN", "ViT1D"]
LABEL = {"ViT1D": "ViT-1D"}
ARMS = ["none", "rpst_extrap", "sra_extrap", "rpst_interp"]
ARM_LABEL = {"none": "no aug", "rpst_extrap": "RPST extrap.\n{700,800,900}",
             "sra_extrap": "resampling\n{700,800,900}", "rpst_interp": "RPST interp.\n{800,1000}"}
ARM_COLOR = {"none": "#95a5a6", "rpst_extrap": "#c0392b",
             "sra_extrap": "#2980b9", "rpst_interp": "#e67e22"}

d = json.load(open("outputs/h2_extrapolation_results.json"))
accs = {}
for r in d["runs"]:
    if r.get("status") == "success":
        accs.setdefault((r["model"], r["arm"]), []).append(r["acc"])

def stat(model, arm):
    v = np.array(accs.get((model, arm), [np.nan])) * 100
    return v.mean(), v.std()

fig, ax = plt.subplots(figsize=(7.2, 3.2))
x = np.arange(len(MODELS)); w = 0.2
for j, arm in enumerate(ARMS):
    means = [stat(m, arm)[0] for m in MODELS]
    stds = [stat(m, arm)[1] for m in MODELS]
    off = (j - 1.5) * w
    ax.bar(x + off, means, w, yerr=stds, capsize=2.5, label=ARM_LABEL[arm],
           color=ARM_COLOR[arm], edgecolor="black", linewidth=0.4,
           error_kw={"elinewidth": 0.6, "capthick": 0.6})

ax.set_ylabel("accuracy on unseen 1000 rpm (%)")
ax.set_xticks(x)
ax.set_xticklabels([LABEL.get(m, m) for m in MODELS])
ax.set_ylim(70, 104)
ax.legend(loc="lower center", ncol=4, framealpha=0.9, fontsize=7.0,
          bbox_to_anchor=(0.5, 1.005), columnspacing=1.0, handlelength=1.2)
ax.grid(axis="y", alpha=0.3, lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_extrapolation.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_extrapolation.png", bbox_inches="tight")
plt.close(fig)
for m in MODELS:
    print(m, {a: f"{stat(m,a)[0]:.1f}±{stat(m,a)[1]:.1f}" for a in ARMS})
print("-> fig_extrapolation.pdf")
