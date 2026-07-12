"""G2+G3: definitive 5-seed JNU benchmark with logit logging and primary metrics.

6 models x 3 tasks x 2 regimes (noaug / RPST) x 5 seeds (42,1337,2025,999,7).
WaPIGT = canonical full config (MST+PIFFG+SCR+triplet).
Checkpoint rule: strict (>) for noaug, tie-keeping (>=) for RPST.
Per-run: acc/MF1/AUROC/kappa + logits saved to outputs/logits/.
Final: Wilcoxon signed-rank (one-sided, WaPIGT > baseline) per task/regime, n=5.
-> outputs/g2_definitive_results.json
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.baselines import BASELINE_REGISTRY
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss
import scripts.test_tsm_aug as T
from scripts.test_tsm_aug import TSMAugJNUDataset, JNU_SIG_LEN, JNU_FS, reorg_bp
from scripts.metrics_util import evaluate_full

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/g2_definitive_results.json"
LOGITS_DIR = Path("outputs/logits")
LOGITS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1337, 2025, 999, 7]
MODELS = ["WaPIGT", "WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D"]
TASKS = ["JNU_T01", "JNU_T02", "JNU_T03"]


def jnu_loaders(task, augment):
    sp = T.JNU_SPLITS[task]
    root = str(T.config.data.jnu_raw_root)

    def ds(speeds, n, aug=False):
        return TSMAugJNUDataset(
            dataset="JNU", raw_root=root, speed_list=speeds, split="train",
            n_samples_per_bearing=n, signal_length=JNU_SIG_LEN,
            task_name=task, test_speed_list=sp["test"], augment=aug)

    kw = dict(batch_size=32, num_workers=0, pin_memory=False)
    return (DataLoader(ds(sp["train"], T.JNU_N_TRAIN, aug=augment), shuffle=True, **kw),
            DataLoader(ds(sp["val"], T.JNU_N_EVAL), shuffle=False, **kw),
            DataLoader(ds(sp["test"], T.JNU_N_EVAL), shuffle=False, **kw))


@torch.no_grad()
def quick_eval(model, loader, is_wapigt):
    model.eval()
    c = t = 0
    for batch in loader:
        sigs = batch["signal"].to(device)
        labs = batch["label"].to(device)
        if is_wapigt:
            bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
            logits, *_ = model(sigs, bp, fs_sampling=JNU_FS)
        else:
            out = model(sigs)
            logits = out[0] if isinstance(out, tuple) else out
        c += (logits.argmax(1) == labs).sum().item()
        t += labs.size(0)
    return c / max(t, 1)


def train_one(name, loaders, seed, tie_keep):
    train_l, val_l, test_l = loaders
    set_all_seeds(seed)
    is_wapigt = name == "WaPIGT"
    if is_wapigt:
        model = WaPIGT(
            n_classes=4, hidden_dim=config.model.hidden_dim,
            n_encoder_layers=config.model.n_encoder_layers,
            n_heads=config.model.n_heads, mlp_dim=config.model.mlp_dim,
            dropout=config.model.dropout, n_gat_heads=config.model.n_gat_heads,
            gat_dropout=config.model.gat_dropout).to(device)
        scr = SpectrumConsistencyRegularizer(sigma=2.0)
        loss_fn = WaPIGTLoss(
            n_classes=4, scr_module=scr, scr_lambda=config.model.scr_lambda,
            scr_warmup_epochs=config.model.scr_warmup_epochs, n_epochs=120,
            triplet_lambda=getattr(config.model, 'triplet_lambda', 0.1),
            triplet_margin=0.5, triplet_warmup_epochs=20)
        opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                          weight_decay=config.training.weight_decay)
        n_epochs = 120
    else:
        model = BASELINE_REGISTRY[name](
            signal_length=JNU_SIG_LEN, n_classes=4, dropout=0.3).to(device)
        loss_fn = None
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        n_epochs = 80
    patience = 20
    best_val, best_state, no_improve = 0.0, None, 0
    for epoch in range(n_epochs):
        if loss_fn:
            loss_fn.set_epoch(epoch)
        model.train()
        for batch in train_l:
            sigs = batch["signal"].to(device)
            labs = batch["label"].to(device)
            opt.zero_grad()
            if is_wapigt:
                bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
                logits, attn, emb = model(sigs, bp, fs_sampling=JNU_FS)
                ffb = batch.get("fault_freq_bins")
                loss = loss_fn(logits, labs, attn,
                               ffb.to(device) if ffb is not None else None,
                               sigs.shape[-1], JNU_FS, embeddings=emb)
            else:
                out = model(sigs)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits, labs)
            if torch.isnan(loss):
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = quick_eval(model, val_l, is_wapigt)
        if v > best_val:
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
            no_improve = 0
        elif tie_keep and v == best_val:
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
            no_improve += 1
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    metrics, logits, labels = evaluate_full(model, test_l, device,
                                            is_wapigt=is_wapigt, fs=JNU_FS,
                                            reorg_bp=reorg_bp)
    return best_val, metrics, logits, labels


def main():
    print(f"Device: {device}")
    results = {"start_time": datetime.now().isoformat(), "runs": []}

    # seed 42 first across everything (validates against known 3-seed numbers),
    # then the remaining seeds
    seed_order = [42, 1337, 2025, 999, 7]
    for regime, augment in [("noaug", False), ("RPST", True)]:
        for task in TASKS:
            loaders = jnu_loaders(task, augment)
            for name in MODELS:
                accs = []
                for seed in seed_order:
                    try:
                        val, metrics, logits, labels = train_one(
                            name, loaders, seed, tie_keep=augment)
                        tag = f"{name}_{task}_{regime}_s{seed}"
                        np.savez(LOGITS_DIR / f"g2_{tag}.npz",
                                 logits=logits, labels=labels)
                        print(f"{name}[{regime}] {task} seed={seed}: "
                              f"acc={metrics['acc']:.4f} mf1={metrics['mf1']:.4f} "
                              f"auroc={metrics['auroc']:.4f} kappa={metrics['kappa']:.4f}",
                              flush=True)
                        results["runs"].append({"model": name, "task": task,
                                                "regime": regime, "seed": seed,
                                                "status": "success", "val_acc": val,
                                                **metrics})
                        accs.append(metrics["acc"])
                    except Exception as e:
                        print(f"{name}[{regime}] {task} seed={seed} FAILED: {e}", flush=True)
                        results["runs"].append({"model": name, "task": task,
                                                "regime": regime, "seed": seed,
                                                "status": "failed", "error": str(e)})
                    with open(OUTPUT, "w") as f:
                        json.dump(results, f, indent=2)
                if accs:
                    print(f">>> {name}[{regime}] {task}: "
                          f"{np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n",
                          flush=True)

    # Wilcoxon: WaPIGT vs each baseline, paired over seeds, per task+regime
    ok = [r for r in results["runs"] if r.get("status") == "success"]
    acc = {}
    for r in ok:
        acc[(r["model"], r["task"], r["regime"], r["seed"])] = r["acc"]
    stats = {}
    for regime in ["noaug", "RPST"]:
        for task in TASKS:
            for m in MODELS[1:]:
                w = [acc.get(("WaPIGT", task, regime, s)) for s in SEEDS]
                b = [acc.get((m, task, regime, s)) for s in SEEDS]
                if None in w or None in b:
                    continue
                diffs = np.array(w) - np.array(b)
                if np.all(diffs == 0):
                    p = 1.0
                else:
                    try:
                        p = float(wilcoxon(w, b, alternative="greater").pvalue)
                    except Exception:
                        p = float("nan")
                stats[f"{regime}|{task}|WaPIGT_vs_{m}"] = {
                    "wapigt_mean": float(np.mean(w)), "baseline_mean": float(np.mean(b)),
                    "p_one_sided_greater": p}
    results["wilcoxon"] = stats
    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
