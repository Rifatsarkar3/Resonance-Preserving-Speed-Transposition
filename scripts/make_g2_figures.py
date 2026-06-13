"""Generate fig_rpst_gains.pdf from the definitive 5-seed g2 results."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})
OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D", "WaPIGT"]
LABEL = {"ViT1D": "ViT-1D", "WaPIGT": "WaPIGT"}
TASKS = ["JNU_T01", "JNU_T02", "JNU_T03"]
SEEDS = [42, 1337, 2025, 999, 7]

d = json.load(open("outputs/g2_definitive_results.json"))
acc = {}
for r in d["runs"]:
    if r.get("status") == "success":
        acc[(r["model"], r["task"], r["regime"], r["seed"])] = r["acc"]

def mean_over(model, regime):
    vals = [acc[(model, t, regime, s)] for t in TASKS for s in SEEDS
            if (model, t, regime, s) in acc]
    return np.mean(vals) * 100

noaug = [mean_over(m, "noaug") for m in MODELS]
rpst = [mean_over(m, "RPST") for m in MODELS]

fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = np.arange(len(MODELS)); w = 0.38
ax.bar(x - w/2, noaug, w, label="no augmentation", color="#95a5a6", edgecolor="black", linewidth=0.4)
ax.bar(x + w/2, rpst, w, label="+ RPST", color="#c0392b", edgecolor="black", linewidth=0.4)
for i, (n, r) in enumerate(zip(noaug, rpst)):
    ax.annotate(f"+{r-n:.1f}", (i, r + 0.4), ha="center", fontsize=7, color="#c0392b")
ax.set_ylabel("JNU mean accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels([LABEL.get(m, m) for m in MODELS])
ax.set_ylim(82, 102)
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3, lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_rpst_gains.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_rpst_gains.png", bbox_inches="tight")
plt.close(fig)
print("fig_rpst_gains.pdf  noaug=", [round(v,1) for v in noaug], " rpst=", [round(v,1) for v in rpst])
