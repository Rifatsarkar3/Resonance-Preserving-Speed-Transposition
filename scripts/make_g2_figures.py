"""fig_rpst_gains: JNU mean accuracy, no augmentation vs +RPST, per architecture
(5-seed definitive g2 results). House style via figstyle."""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import figstyle as fs
import matplotlib.pyplot as plt

fs.apply()
MODELS = ["WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D", "WaPIGT"]
LABEL = {"ViT1D": "ViT-1D", "WaPIGT": "WaPIGT\n(full)"}
TASKS = ["JNU_T01", "JNU_T02", "JNU_T03"]
SEEDS = [42, 1337, 2025, 999, 7]

d = json.load(open("outputs/g2_definitive_results.json"))
acc = {(r["model"], r["task"], r["regime"], r["seed"]): r["acc"]
       for r in d["runs"] if r.get("status") == "success"}

def stats(model, regime):
    v = np.array([acc[(model, t, regime, s)] for t in TASKS for s in SEEDS
                  if (model, t, regime, s) in acc]) * 100
    return v.mean(), v.std()

noaug = [stats(m, "noaug") for m in MODELS]
rpst = [stats(m, "RPST") for m in MODELS]

fig, ax = plt.subplots(figsize=(7.0, 3.1))
x = np.arange(len(MODELS)); w = 0.38
ax.bar(x - w/2, [m for m, _ in noaug], w, yerr=[s for _, s in noaug], capsize=2.5,
       label="no augmentation", color=fs.PALETTE["baseline"], edgecolor="black",
       error_kw={"elinewidth": 0.7, "capthick": 0.7})
ax.bar(x + w/2, [m for m, _ in rpst], w, yerr=[s for _, s in rpst], capsize=2.5,
       label="with RPST", color=fs.PALETTE["rpst"], edgecolor="black",
       error_kw={"elinewidth": 0.7, "capthick": 0.7})
for i, ((n, _), (r, rs)) in enumerate(zip(noaug, rpst)):
    ax.annotate(f"+{r-n:.1f}", (i + w/2, min(r + rs + 0.6, 103.0)), ha="center",
                fontsize=7, color=fs.PALETTE["rpst"], fontweight="bold")

fs.clean(ax)
ax.set_ylabel("JNU mean accuracy (%)")
ax.set_xticks(x); ax.set_xticklabels([LABEL.get(m, m) for m in MODELS])
ax.set_ylim(80, 104)
ax.legend(loc="lower right", ncol=2)
fig.tight_layout()
fs.save(fig, "fig_rpst_gains")
print("fig_rpst_gains:", [(LABEL.get(m, m), round(n, 1), round(r, 1)) for m, (n, _), (r, _) in zip(MODELS, noaug, rpst)])
