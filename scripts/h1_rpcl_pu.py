"""H1: Resonance-Preserving Consistency Learning (RPCL) on PU cross-speed.

NEW METHOD. Beyond plain RPST (which adds transposed copies as independent labelled
samples), RPCL enforces that the resonance-preserving transpositions of the SAME
window map to the same embedding and prediction -- an explicit speed-invariance
constraint derived from the resonance-preservation principle.

Each training window -> V views (identity at source speed + RPST to target speed),
each with its own correctly-scaled shaft frequency for PIFFG/SCR. Loss:
  L = CE(all views) + lambda(t) * [ ||emb_i - emb_mean||^2  +  KL(p_i || p_mean) ]
with stop-grad on the group means and a linear warmup on lambda.

Compares WaPIGT under {RPST, RPCL} on PU_S1/S2 (headroom ~70%), 3 seeds.
-> outputs/h1_rpcl_pu_results.json
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss
from scripts.metrics_util import evaluate_full
from scripts.g1_pu_cross_speed import (
    load_vibration, tsm_stretch, PUCrossSpeedDataset, pu_loaders, reorg_bp,
    PU_FS, PU_SIG_LEN, PU_BEARING, COND_FS, COND_RPM, CLASS_BEARINGS,
    TRAIN_REPS, N_TRAIN_PER_FILE, TASKS,
)

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/h1_rpcl_pu_results.json"
LOGITS_DIR = Path("outputs/logits"); LOGITS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1337, 2025]


class RPCLGroupDataset(Dataset):
    """Each item = V views (identity + RPST transposition) of one base window,
    with per-view shaft frequency. Used only for training."""

    def __init__(self, condition, target_condition, reps, n_per_file):
        self.items = []  # list of (views ndarray (V,L), view_fs list, label)
        pu_root = Path(config.data.pu_raw_root) / "PU"
        f_src = COND_FS[condition]
        f_tgt = COND_FS[target_condition]
        rate = COND_RPM[target_condition] / COND_RPM[condition]
        for label, bearings in CLASS_BEARINGS.items():
            for bid in bearings:
                for rep in reps:
                    path = pu_root / bid / f"{condition}_{bid}_{rep}.mat"
                    if not path.exists():
                        continue
                    sig = load_vibration(path)
                    sig_t = tsm_stretch(sig, rate)
                    ms = max(1, len(sig) - PU_SIG_LEN)
                    ms_t = max(1, len(sig_t) - PU_SIG_LEN)
                    for i in range(n_per_file):
                        s0 = (i * PU_SIG_LEN) % ms
                        w0 = sig[s0:s0 + PU_SIG_LEN]
                        st = (i * PU_SIG_LEN) % ms_t
                        wt = sig_t[st:st + PU_SIG_LEN]
                        w0 = np.pad(w0, (0, PU_SIG_LEN - len(w0))) if len(w0) < PU_SIG_LEN else w0
                        wt = np.pad(wt, (0, PU_SIG_LEN - len(wt))) if len(wt) < PU_SIG_LEN else wt
                        views = np.stack([w0, wt]).astype(np.float32)  # (2, L)
                        self.items.append((views, [f_src, f_tgt], label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        views, view_fs, label = self.items[idx]
        return {
            "views": torch.from_numpy(views).unsqueeze(1),   # (V, 1, L)
            "view_fs": torch.tensor(view_fs, dtype=torch.float32),  # (V,)
            "label": torch.tensor(label, dtype=torch.long),
        }


def rpcl_loaders(task):
    t = TASKS[task]
    train = RPCLGroupDataset(t["train"], t["test"], TRAIN_REPS, N_TRAIN_PER_FILE)
    _, val_l, test_l = pu_loaders(task, augment=False)
    kw = dict(batch_size=16, num_workers=0, pin_memory=False)  # 16 groups x V=2 = 32
    return DataLoader(train, shuffle=True, **kw), val_l, test_l


def build_wapigt():
    return WaPIGT(
        n_classes=4, hidden_dim=config.model.hidden_dim,
        n_encoder_layers=config.model.n_encoder_layers,
        n_heads=config.model.n_heads, mlp_dim=config.model.mlp_dim,
        dropout=config.model.dropout, n_gat_heads=config.model.n_gat_heads,
        gat_dropout=config.model.gat_dropout).to(device)


def bp_for_fs(fs_list):
    """List of bearing-param dicts (fixed geometry, per-view shaft freq)."""
    return [dict(PU_BEARING, f_s=float(fs)) for fs in fs_list]


@torch.no_grad()
def quick_eval(model, loader):
    model.eval()
    c = t = 0
    for batch in loader:
        sigs = batch["signal"].to(device)
        labs = batch["label"].to(device)
        bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
        logits, *_ = model(sigs, bp, fs_sampling=PU_FS)
        c += (logits.argmax(1) == labs).sum().item()
        t += labs.size(0)
    return c / max(t, 1)


def train_rpcl(loaders, seed, lam_emb=0.5, lam_pred=0.5, warmup=15,
               n_epochs=120, patience=20):
    train_l, val_l, test_l = loaders
    set_all_seeds(seed)
    model = build_wapigt()
    scr = SpectrumConsistencyRegularizer(sigma=2.0)
    loss_fn = WaPIGTLoss(
        n_classes=4, scr_module=scr, scr_lambda=config.model.scr_lambda,
        scr_warmup_epochs=config.model.scr_warmup_epochs, n_epochs=n_epochs,
        triplet_lambda=0.0, triplet_margin=0.5, triplet_warmup_epochs=9999)
    opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)
    best_val, best_state, no_improve = 0.0, None, 0
    for epoch in range(n_epochs):
        loss_fn.set_epoch(epoch)
        lam = min(1.0, epoch / max(1, warmup))
        model.train()
        for batch in train_l:
            views = batch["views"].to(device)        # (B, V, 1, L)
            vfs = batch["view_fs"].to(device)         # (B, V)
            labs = batch["label"].to(device)          # (B,)
            B, V = views.shape[0], views.shape[1]
            x = views.reshape(B * V, 1, PU_SIG_LEN)
            fs_flat = vfs.reshape(B * V).tolist()
            bp = bp_for_fs(fs_flat)
            opt.zero_grad()
            logits, attn, emb = model(x, bp, fs_sampling=PU_FS)  # (B*V,C),(B*V,..),(B*V,d)
            labs_rep = labs.repeat_interleave(V)
            # classification + SCR on all views (SCR uses per-view fault bins from f_s)
            loss_cls = loss_fn(logits, labs_rep, attn, None,
                               PU_SIG_LEN, PU_FS, embeddings=emb)
            # consistency across views of each group
            C = logits.shape[-1]
            p = F.softmax(logits, dim=-1).reshape(B, V, C)
            p_mean = p.mean(1, keepdim=True).detach()
            l_pred = (p * (torch.log(p + 1e-8) - torch.log(p_mean + 1e-8))).sum(-1).mean()
            e = F.normalize(emb, dim=-1).reshape(B, V, -1)
            e_mean = e.mean(1, keepdim=True).detach()
            l_emb = ((e - e_mean) ** 2).sum(-1).mean()
            loss = loss_cls + lam * (lam_pred * l_pred + lam_emb * l_emb)
            if torch.isnan(loss):
                opt.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = quick_eval(model, val_l)
        if v >= best_val:                       # tie-keeping (augmented regime)
            if v > best_val:
                no_improve = 0
            else:
                no_improve += 1
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return best_val, evaluate_full(model, test_l, device, is_wapigt=True,
                                   fs=PU_FS, reorg_bp=reorg_bp)


def train_rpst(loaders, seed, n_epochs=120, patience=20):
    """Plain RPST baseline through the identical pipeline (no consistency)."""
    train_l, val_l, test_l = loaders
    set_all_seeds(seed)
    model = build_wapigt()
    scr = SpectrumConsistencyRegularizer(sigma=2.0)
    loss_fn = WaPIGTLoss(
        n_classes=4, scr_module=scr, scr_lambda=config.model.scr_lambda,
        scr_warmup_epochs=config.model.scr_warmup_epochs, n_epochs=n_epochs,
        triplet_lambda=0.0, triplet_margin=0.5, triplet_warmup_epochs=9999)
    opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)
    best_val, best_state, no_improve = 0.0, None, 0
    for epoch in range(n_epochs):
        loss_fn.set_epoch(epoch)
        model.train()
        for batch in train_l:
            views = batch["views"].to(device)
            vfs = batch["view_fs"].to(device)
            labs = batch["label"].to(device)
            B, V = views.shape[0], views.shape[1]
            x = views.reshape(B * V, 1, PU_SIG_LEN)
            bp = bp_for_fs(vfs.reshape(B * V).tolist())
            opt.zero_grad()
            logits, attn, emb = model(x, bp, fs_sampling=PU_FS)
            loss = loss_fn(logits, labs.repeat_interleave(V), attn, None,
                           PU_SIG_LEN, PU_FS, embeddings=emb)
            if torch.isnan(loss):
                opt.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = quick_eval(model, val_l)
        if v >= best_val:
            if v > best_val:
                no_improve = 0
            else:
                no_improve += 1
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return best_val, evaluate_full(model, test_l, device, is_wapigt=True,
                                   fs=PU_FS, reorg_bp=reorg_bp)


def main():
    print(f"Device: {device}")
    results = {"start_time": datetime.now().isoformat(), "runs": []}
    for task in ["PU_S1", "PU_S2"]:
        loaders = rpcl_loaders(task)
        print(f"{task}: train groups={len(loaders[0].dataset)} "
              f"val={len(loaders[1].dataset)} test={len(loaders[2].dataset)}", flush=True)
        for method in ["RPST", "RPCL"]:
            accs = []
            for seed in SEEDS:
                try:
                    if method == "RPCL":
                        val, (metrics, logits, labels) = train_rpcl(loaders, seed)
                    else:
                        val, (metrics, logits, labels) = train_rpst(loaders, seed)
                    np.savez(LOGITS_DIR / f"h1_WaPIGT_{method}_{task}_s{seed}.npz",
                             logits=logits, labels=labels)
                    print(f"WaPIGT[{method}] {task} seed={seed}: val={val:.4f} "
                          f"acc={metrics['acc']:.4f} mf1={metrics['mf1']:.4f} "
                          f"auroc={metrics['auroc']:.4f} kappa={metrics['kappa']:.4f}",
                          flush=True)
                    results["runs"].append({"model": "WaPIGT", "method": method,
                                            "task": task, "seed": seed,
                                            "status": "success", "val_acc": val, **metrics})
                    accs.append(metrics["acc"])
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"WaPIGT[{method}] {task} seed={seed} FAILED: {e}", flush=True)
                    results["runs"].append({"model": "WaPIGT", "method": method,
                                            "task": task, "seed": seed,
                                            "status": "failed", "error": str(e)})
                with open(OUTPUT, "w") as f:
                    json.dump(results, f, indent=2)
            if accs:
                print(f">>> WaPIGT[{method}] {task}: "
                      f"{np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n", flush=True)
    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
