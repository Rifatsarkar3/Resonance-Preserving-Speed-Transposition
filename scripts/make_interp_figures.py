"""Generate physical-interpretation figures from a real trained WaPIGT-final model.

Trains WaPIGT-final (MST+PIFFG+SCR, no triplet) on JNU T03 + RPST, seed 42,
logging SCR loss per epoch, then produces:
  paper/figures/fig_freq_attn.pdf      - CLS attention overlaid on FFT (test sample)
  paper/figures/fig_piffg_graph.pdf    - PIFFG graph with learned GAT edge weights
  paper/figures/fig_tsne.pdf           - t-SNE of test CLS embeddings
  paper/figures/fig_scr_convergence.pdf- SCR loss + val accuracy vs epoch
Also saves the measured PIFFG edge weights to outputs/piffg_edge_weights.json.
"""
import sys, json, numpy as np
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss
from src.utils.fault_frequencies import compute_fault_frequencies
from scripts.test_tsm_aug import jnu_loaders, reorg_bp, evaluate, JNU_FS

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TASK = "JNU_T03"
N_EPOCHS = 120
PATIENCE = 20
CLASS_NAMES = ["Normal", "Inner race", "Outer race", "Ball"]
NODE_NAMES = ["BPFO", "BPFI", "FTF", "BSF", "$f_s$", "$2f_s$", "$3f_s$"]


def main():
    print(f"Device: {device}")
    train_l, val_l, test_l = jnu_loaders(TASK)
    set_all_seeds(SEED)
    model = WaPIGT(
        n_classes=4, hidden_dim=config.model.hidden_dim,
        n_encoder_layers=config.model.n_encoder_layers,
        n_heads=config.model.n_heads, mlp_dim=config.model.mlp_dim,
        dropout=config.model.dropout, n_gat_heads=config.model.n_gat_heads,
        gat_dropout=config.model.gat_dropout,
    ).to(device)
    scr = SpectrumConsistencyRegularizer(sigma=2.0)
    loss_fn = WaPIGTLoss(
        n_classes=4, scr_module=scr, scr_lambda=config.model.scr_lambda,
        scr_warmup_epochs=config.model.scr_warmup_epochs, n_epochs=N_EPOCHS,
        triplet_lambda=0.0, triplet_margin=0.5, triplet_warmup_epochs=9999,
    )
    opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)

    scr_curve, val_curve = [], []
    best_val, best_state, no_improve = 0.0, None, 0
    fixed_val_batch = next(iter(val_l))

    for epoch in range(N_EPOCHS):
        loss_fn.set_epoch(epoch)
        model.train()
        for batch in train_l:
            sigs = batch["signal"].to(device)
            labs = batch["label"].to(device)
            bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
            opt.zero_grad()
            logits, attn, embeddings = model(sigs, bp, fs_sampling=JNU_FS)
            ffb = batch.get("fault_freq_bins")
            loss = loss_fn(logits, labs, attn,
                           ffb.to(device) if ffb is not None else None,
                           sigs.shape[-1], JNU_FS, embeddings=embeddings)
            if torch.isnan(loss):
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # epoch-level SCR loss on a fixed val batch
        model.eval()
        with torch.no_grad():
            sigs = fixed_val_batch["signal"].to(device)
            bp = reorg_bp(fixed_val_batch["bearing_params"], sigs.shape[0])
            _, attn, _ = model(sigs, bp, fs_sampling=JNU_FS)
            ffb = fixed_val_batch.get("fault_freq_bins")
            try:
                s = scr(attn, ffb.to(device), sigs.shape[-1], JNU_FS)
                scr_curve.append(float(s))
            except Exception:
                scr_curve.append(float("nan"))
        v = evaluate(model, val_l)
        val_curve.append(v)
        if v >= best_val:
            if v > best_val:
                no_improve = 0
            else:
                no_improve += 1
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
        print(f"epoch {epoch}: val={v:.3f} scr={scr_curve[-1]:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_l)
    print(f"final test acc: {test_acc:.4f}")
    model.eval()

    # ---- fig_scr_convergence ------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(4.2, 2.6))
    ep = np.arange(1, len(scr_curve) + 1)
    ax1.plot(ep, scr_curve, color="#c0392b", lw=1.2, label="$\\mathcal{L}_{SCR}$")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SCR loss (KL)", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(ep, np.array(val_curve) * 100, color="#2980b9", lw=1.2, ls="--",
             label="val acc")
    ax2.set_ylabel("Validation accuracy (%)", color="#2980b9")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scr_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_scr_convergence.pdf")

    # ---- collect test attention + embeddings --------------------------------
    all_emb, all_lab, sample_attn, sample_sig, sample_lab = [], [], None, None, None
    with torch.no_grad():
        for batch in test_l:
            sigs = batch["signal"].to(device)
            labs = batch["label"]
            bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
            logits, attn, emb = model(sigs, bp, fs_sampling=JNU_FS)
            all_emb.append(emb.cpu().numpy())
            all_lab.append(labs.numpy())
            if sample_attn is None:
                # pick an outer-race sample (label 2)
                idx = (labs == 2).nonzero().flatten()
                if len(idx) > 0:
                    i = idx[0].item()
                    sample_attn = attn[i].mean(0)[0, 1:].cpu().numpy()  # CLS->tokens
                    sample_sig = sigs[i].squeeze().cpu().numpy()
                    sample_lab = 2
    all_emb = np.concatenate(all_emb)
    all_lab = np.concatenate(all_lab)

    # ---- fig_freq_attn -------------------------------------------------------
    n_tok = len(sample_attn)
    f_nyq = JNU_FS / 2
    tok_freq = (np.arange(n_tok) + 0.5) * f_nyq / n_tok
    spec = np.abs(np.fft.rfft(sample_sig - sample_sig.mean()))
    spec_f = np.fft.rfftfreq(len(sample_sig), 1 / JNU_FS)
    # test speed for T03 is 600 rpm -> f_shaft = 10 Hz
    freqs = compute_fault_frequencies(N_balls=8, d_mm=7.5, D_mm=38.5,
                                      alpha_deg=0.0, f_shaft_hz=10.0)
    fig, ax1 = plt.subplots(figsize=(7.0, 2.8))
    fmax = 2000
    m = spec_f <= fmax
    ax1.plot(spec_f[m], spec[m] / spec[m].max(), color="#7f8c8d", lw=0.5,
             label="FFT magnitude (norm.)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Normalised FFT magnitude")
    ax2 = ax1.twinx()
    mt = tok_freq <= fmax
    ax2.plot(tok_freq[mt], sample_attn[mt] / sample_attn[mt].max(), color="#c0392b",
             lw=1.2, label="CLS attention (layer 2)")
    ax2.set_ylabel("Normalised attention", color="#c0392b")
    for name, color in [("BPFO", "#2980b9"), ("BPFI", "#27ae60")]:
        for k in (1, 2):
            f = freqs[name] * k
            if f <= fmax:
                ax1.axvline(f, color=color, ls="--", lw=0.8, alpha=0.8)
                ax1.text(f, 1.02, f"{k}$\\times${name}" if k > 1 else name,
                         rotation=90, fontsize=6, ha="right", va="bottom", color=color)
    fig.tight_layout()
    fig.savefig(OUT / "fig_freq_attn.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_freq_attn.pdf")

    # ---- fig_tsne ------------------------------------------------------------
    from sklearn.manifold import TSNE
    z = TSNE(n_components=2, random_state=0, perplexity=20).fit_transform(all_emb)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    colors = ["#2980b9", "#27ae60", "#e67e22", "#c0392b"]
    for c in range(4):
        mk = all_lab == c
        ax.scatter(z[mk, 0], z[mk, 1], s=8, color=colors[c], label=CLASS_NAMES[c],
                   alpha=0.8, edgecolor="none")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(OUT / "fig_tsne.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_tsne.pdf")

    # ---- fig_piffg_graph: GAT layer-1 edge attention -------------------------
    piffg = model.piffg if hasattr(model, "piffg") else None
    if piffg is None:
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "PhysicsInformedFaultFrequencyGraph":
                piffg = mod
                break
    bp_one = reorg_bp(fixed_val_batch["bearing_params"], 1)
    g = piffg._build_graph(bp_one, JNU_FS).to(device)
    with torch.no_grad():
        _, (edge_index, alpha) = piffg.gat1(g.x, g.edge_index,
                                            return_attention_weights=True)
    edge_index = edge_index.cpu().numpy()
    alpha = alpha.mean(1).cpu().numpy()  # mean over heads
    edges = {}
    for k in range(edge_index.shape[1]):
        s, d = int(edge_index[0, k]), int(edge_index[1, k])
        if s == d:
            continue
        key = tuple(sorted((s, d)))
        edges.setdefault(key, []).append(float(alpha[k]))
    edge_w = {k: float(np.mean(v)) for k, v in edges.items()}
    with open("outputs/piffg_edge_weights.json", "w") as f:
        json.dump({f"{NODE_NAMES[a]}-{NODE_NAMES[b]}".replace("$", ""): w
                   for (a, b), w in edge_w.items()}, f, indent=2)
    print("edge weights:", {f"{a}-{b}": round(w, 3) for (a, b), w in edge_w.items()})

    pos = {0: (-1.0, 0.6), 1: (-1.0, -0.6), 2: (0.0, 1.0), 3: (0.0, -1.0),
           4: (1.0, 0.0), 5: (1.9, 0.6), 6: (1.9, -0.6)}
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    wmax = max(edge_w.values())
    for (a, b), w in edge_w.items():
        xa, ya = pos[a]; xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], color="#34495e", lw=0.5 + 3.5 * w / wmax,
                alpha=0.45 + 0.55 * w / wmax, zorder=1)
        ax.text((xa + xb) / 2, (ya + yb) / 2 + 0.07, f"{w:.2f}", fontsize=7,
                ha="center", color="#2c3e50", zorder=3)
    for i, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, color="#ecf0f1", edgecolor="#2c3e50",
                   linewidth=1.2, zorder=2)
        ax.text(x, y, NODE_NAMES[i], ha="center", va="center", fontsize=8, zorder=4)
    ax.set_xlim(-1.6, 2.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_piffg_graph.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_piffg_graph.pdf")
    print("Done. test acc:", test_acc)


if __name__ == "__main__":
    main()
