"""fig_extrapolation: speed extrapolation to the never-synthesised 1000 rpm
(h2 results). Grouped bars per model x arm, std error bars. House style."""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import figstyle as fs
import matplotlib.pyplot as plt

fs.apply()
MODELS = ["WaPIGT", "TICNN", "ViT1D"]
LABEL = {"ViT1D": "ViT-1D"}
ARMS = ["none", "rpst_extrap", "sra_extrap", "rpst_interp"]
ARM_LABEL = {"none": "no aug.", "rpst_extrap": "RPST extrap.\n$\\{700,800,900\\}$\n(test unseen)",
             "sra_extrap": "resampling\n$\\{700,800,900\\}$\n(test unseen)",
             "rpst_interp": "RPST, incl. test\n$\\{800,1000\\}$\n(test synthesised)"}
ARM_COLOR = {"none": fs.PALETTE["baseline"], "rpst_extrap": fs.PALETTE["rpst"],
             "sra_extrap": fs.PALETTE["resample"], "rpst_interp": fs.PALETTE["interp"]}

d = json.load(open("outputs/h2_extrapolation_results.json"))
accs = {}
for r in d["runs"]:
    if r.get("status") == "success":
        accs.setdefault((r["model"], r["arm"]), []).append(r["acc"])

def stat(m, a):
    v = np.array(accs.get((m, a), [np.nan])) * 100
    return v.mean(), v.std()

fig, ax = plt.subplots(figsize=(7.2, 3.3))
x = np.arange(len(MODELS)); w = 0.2
for j, arm in enumerate(ARMS):
    means = [stat(m, arm)[0] for m in MODELS]
    stds = [stat(m, arm)[1] for m in MODELS]
    ax.bar(x + (j - 1.5) * w, means, w, yerr=stds, capsize=2.5, label=ARM_LABEL[arm],
           color=ARM_COLOR[arm], edgecolor="black",
           error_kw={"elinewidth": 0.6, "capthick": 0.6})

fs.clean(ax)
ax.set_ylabel("accuracy on unseen 1000 rpm (%)")
ax.set_xticks(x); ax.set_xticklabels([LABEL.get(m, m) for m in MODELS])
ax.set_ylim(70, 104)
ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.01), columnspacing=1.0,
          handlelength=1.2, fontsize=7.0)
fig.tight_layout()
fs.save(fig, "fig_extrapolation")
for m in MODELS:
    print(m, {a: f"{stat(m, a)[0]:.1f}" for a in ARMS})
