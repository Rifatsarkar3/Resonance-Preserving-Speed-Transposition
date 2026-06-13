"""H3: identity-anchored RPCL. The synthetic (transposed) views are pulled toward the
REAL (identity) view's embedding/prediction -- real signal as teacher -- instead of a
moving mutual mean (which raised variance in H1). More principled and typically more
stable. Compares WaPIGT {RPCL-anchored} on PU_S1/S2 (3 seeds) vs H1's RPST/RPCL-mean.
-> outputs/h3_rpcl_anchored_results.json
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss
from scripts.metrics_util import evaluate_full
from scripts.h1_rpcl_pu import rpcl_loaders, build_wapigt, bp_for_fs, quick_eval
from scripts.g1_pu_cross_speed import reorg_bp, PU_FS, PU_SIG_LEN

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/h3_rpcl_anchored_results.json"
LOGITS_DIR = Path("outputs/logits"); LOGITS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1337, 2025]


def train_anchored(loaders, seed, lam_emb=0.5, lam_pred=0.5, warmup=15,
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
            views = batch["views"].to(device); vfs = batch["view_fs"].to(device)
            labs = batch["label"].to(device)
            B, V = views.shape[0], views.shape[1]
            x = views.reshape(B * V, 1, PU_SIG_LEN)
            bp = bp_for_fs(vfs.reshape(B * V).tolist())
            opt.zero_grad()
            logits, attn, emb = model(x, bp, fs_sampling=PU_FS)
            loss_cls = loss_fn(logits, labs.repeat_interleave(V), attn, None,
                               PU_SIG_LEN, PU_FS, embeddings=emb)
            C = logits.shape[-1]
            p = F.softmax(logits, dim=-1).reshape(B, V, C)
            e = F.normalize(emb, dim=-1).reshape(B, V, -1)
            # anchor = identity view (index 0 = real signal), detached teacher
            p_anchor = p[:, 0:1, :].detach()
            e_anchor = e[:, 0:1, :].detach()
            p_syn = p[:, 1:, :]
            e_syn = e[:, 1:, :]
            l_pred = (p_syn * (torch.log(p_syn + 1e-8) - torch.log(p_anchor + 1e-8))).sum(-1).mean()
            l_emb = ((e_syn - e_anchor) ** 2).sum(-1).mean()
            loss = loss_cls + lam * (lam_pred * l_pred + lam_emb * l_emb)
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
        accs = []
        for seed in SEEDS:
            try:
                val, (m, lg, lb) = train_anchored(loaders, seed)
                np.savez(LOGITS_DIR / f"h3_WaPIGT_RPCLanchor_{task}_s{seed}.npz", logits=lg, labels=lb)
                print(f"WaPIGT[RPCL-anchor] {task} seed={seed}: val={val:.4f} "
                      f"acc={m['acc']:.4f} mf1={m['mf1']:.4f} auroc={m['auroc']:.4f} kappa={m['kappa']:.4f}",
                      flush=True)
                results["runs"].append({"model": "WaPIGT", "method": "RPCL-anchor",
                                        "task": task, "seed": seed, "status": "success",
                                        "val_acc": val, **m})
                accs.append(m["acc"])
            except Exception as e:
                import traceback; traceback.print_exc()
                results["runs"].append({"model": "WaPIGT", "method": "RPCL-anchor",
                                        "task": task, "seed": seed, "status": "failed", "error": str(e)})
            with open(OUTPUT, "w") as f:
                json.dump(results, f, indent=2)
        if accs:
            print(f">>> WaPIGT[RPCL-anchor] {task}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n", flush=True)
    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
