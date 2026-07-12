"""Replot fig_piffg_graph from the saved measured edge weights
(outputs/piffg_edge_weights.json) -- no model run needed. House style, with white
label backgrounds so the highlighted (strongest) edge weight stays readable."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import figstyle as fs
import matplotlib.pyplot as plt

fs.apply()
NODE = ["BPFO", "BPFI", "FTF", "BSF", "$f_s$", "$2f_s$", "$3f_s$"]
IDX = {"BPFO": 0, "BPFI": 1, "FTF": 2, "BSF": 3, "f_s": 4, "2f_s": 5, "3f_s": 6}
POS = {0: (-1.0, 0.6), 1: (-1.0, -0.6), 2: (0.0, 1.0), 3: (0.0, -1.0),
       4: (1.0, 0.0), 5: (1.9, 0.6), 6: (1.9, -0.6)}

ew = json.load(open("outputs/piffg_edge_weights.json"))
# keys are "<name>-<name>"; node names contain no '-'
edges = {}
for k, w in ew.items():
    a, b = k.split("-")
    edges[tuple(sorted((IDX[a], IDX[b])))] = w
wmax = max(edges.values())

fig, ax = plt.subplots(figsize=(4.6, 3.2))
for (a, b), w in edges.items():
    xa, ya = POS[a]; xb, yb = POS[b]
    strong = w == wmax
    col = fs.PALETTE["rpst"] if strong else "#586672"
    ax.plot([xa, xb], [ya, yb], color=col, lw=0.6 + 4.0 * w / wmax,
            alpha=0.55 + 0.45 * w / wmax, zorder=1)
    # offset the label off the edge line for readability
    dx, dy = (0.18, 0.0) if abs(xa - xb) < 0.1 else (0.0, 0.10)
    ax.text((xa + xb) / 2 + dx, (ya + yb) / 2 + dy, f"{w:.2f}", fontsize=7,
            ha="center", va="center", color=col,
            fontweight="bold" if strong else "normal", zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
for i, (x, y) in POS.items():
    ax.scatter([x], [y], s=950, color="#eef1f3", edgecolor="#2c3e50", linewidth=1.2, zorder=2)
    ax.text(x, y, NODE[i], ha="center", va="center", fontsize=8, zorder=4)
ax.set_xlim(-1.6, 2.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
fig.tight_layout()
fs.save(fig, "fig_piffg_graph", width_in=4.6)
print("fig_piffg_graph replotted; edges:", {f"{NODE[a]}-{NODE[b]}".replace('$',''): round(w,3) for (a,b),w in edges.items()})
