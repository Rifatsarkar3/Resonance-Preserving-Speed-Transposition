"""fig_architecture: WaPIGT pipeline schematic in the manuscript house style.
Boxes colour-coded: data-flow (grey), physics-informed inputs (blue), training
objectives (red). Vector PDF + high-DPI PNG via figstyle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

fs.apply()

DATA = "#eef1f3"      # data-flow blocks
PHYS = "#d6e6f2"      # physics-informed inputs (light blue)
LOSS = "#f7ddd0"      # training objectives (light vermillion)
EDGE = "#2c3e50"
boxes = {}


def box(ax, name, cx, cy, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 fc=fc, ec=EDGE, lw=1.1, zorder=2))
    if title:
        ax.text(cx, cy + h*0.27, title, ha="center", va="center",
                fontsize=8.6, fontweight="bold", zorder=3)
        ax.text(cx, cy - h*0.13, body, ha="center", va="center", fontsize=7.2,
                zorder=3, linespacing=1.25)
    else:
        ax.text(cx, cy, body, ha="center", va="center", fontsize=8.0,
                zorder=3, linespacing=1.25)
    boxes[name] = (cx, cy, w, h)


def edge(ax, a, b, label=None, astart="r", aend="l", lw=1.3, color=EDGE):
    def anchor(box, side):
        cx, cy, w, h = box
        return {"l": (cx - w/2, cy), "r": (cx + w/2, cy),
                "t": (cx, cy + h/2), "b": (cx, cy - h/2)}[side]
    p1, p2 = anchor(boxes[a], astart), anchor(boxes[b], aend)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                 lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=1))
    if label:
        ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.18, label, ha="center",
                va="bottom", fontsize=6.3, color="#555", zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(0, 18); ax.set_ylim(0, 7); ax.axis("off")

    box(ax, "sig", 1.7, 5.3, 2.6, 1.2, None, "Raw signal\n$\\mathbf{x}\\in\\mathbb{R}^{L}$", DATA)
    box(ax, "rpst", 5.0, 5.3, 2.9, 1.4, "RPST", "phase-vocoder\nspeed transposition\n(train only)", PHYS)
    box(ax, "mst", 8.4, 5.3, 2.9, 1.4, "MST", "instance norm $+$\ninception\n$\\rightarrow$ 256 tokens", DATA)
    box(ax, "enc", 11.8, 5.3, 2.9, 1.4, "Transformer", "4 layers, 8 heads\n$[\\mathrm{CLS}]+$pos.", DATA)
    box(ax, "head", 15.4, 5.3, 2.2, 1.2, None, "Classifier", DATA)

    box(ax, "geom", 6.6, 2.5, 2.7, 1.1, None, "Geometry $\\boldsymbol{\\theta}$,\nshaft speed $f_s$", DATA)
    box(ax, "piffg", 10.0, 2.5, 2.9, 1.4, "PIFFG", "2-layer GAT,\n7 fault-freq. nodes", PHYS)

    box(ax, "scr", 12.7, 0.7, 3.2, 1.05, "SCR", "KL(attn $\\|$ fault bins)", LOSS)
    box(ax, "loss", 16.0, 1.9, 3.4, 1.7, None,
        "$\\mathcal{L}_{\\mathrm{CE}}$\n$+\\,\\lambda_{\\mathrm{SCR}}\\mathcal{L}_{\\mathrm{SCR}}$\n$+\\,\\lambda_{\\mathrm{trip}}\\mathcal{L}_{\\mathrm{trip}}$", LOSS)

    edge(ax, "sig", "rpst"); edge(ax, "rpst", "mst")
    edge(ax, "mst", "enc"); edge(ax, "enc", "head")
    edge(ax, "geom", "piffg")
    edge(ax, "piffg", "enc", label="$\\mathbf{g}$ bias / layer", astart="t", aend="b",
         color=fs.PALETTE["resample"])
    edge(ax, "enc", "scr", label="layer-2 attn", astart="b", aend="t",
         color=fs.PALETTE["rpst"])
    edge(ax, "scr", "loss", astart="r", aend="l", color=fs.PALETTE["rpst"])
    edge(ax, "head", "loss", astart="b", aend="t", color=EDGE)

    ax.legend(handles=[Patch(fc=DATA, ec=EDGE, label="data flow"),
                       Patch(fc=PHYS, ec=EDGE, label="physics-informed input"),
                       Patch(fc=LOSS, ec=EDGE, label="training objective")],
              loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=3, fontsize=7,
              handlelength=1.1, columnspacing=1.1)
    fig.tight_layout()
    fs.save(fig, "fig_architecture", width_in=6.5)
    print("fig_architecture done")


if __name__ == "__main__":
    main()
