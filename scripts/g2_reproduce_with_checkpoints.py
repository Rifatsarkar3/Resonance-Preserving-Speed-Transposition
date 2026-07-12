"""G2 checkpoint reproduction: re-runs the exact g2_definitive_runs.py training procedure
(same seeds, same architectures, same training loop) SOLELY to produce loadable weight
files for the code release. Does NOT touch outputs/g2_definitive_results.json (the
authoritative results file the manuscript's tables were computed from) -- writes its own
results to outputs/g2_reproduction_results.json for comparison/audit only.

6 models x 3 tasks x 2 regimes (noaug / RPST) x 5 seeds (42,1337,2025,999,7) = 180 runs.
Saves model_state_dict only (no optimizer/scheduler) per run to
outputs/checkpoints_release/{model}_{task}_{regime}_s{seed}.pt
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
import scripts.test_tsm_aug as T
from scripts.test_tsm_aug import TSMAugJNUDataset, JNU_SIG_LEN, JNU_FS, reorg_bp
from scripts.metrics_util import evaluate_full
from scripts.g2_definitive_runs import jnu_loaders, train_one, SEEDS, MODELS, TASKS

OUTPUT = "outputs/g2_reproduction_results.json"
CKPT_DIR = Path("outputs/checkpoints_release")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_with_checkpoint(name, loaders, seed, tie_keep):
    """Copy of g2_definitive_runs.train_one that also returns best_state for saving."""
    import torch.optim as optim
    import torch.nn.functional as F
    from src.utils.reproducibility import set_all_seeds
    from src.baselines import BASELINE_REGISTRY
    from src.models.wapigt import WaPIGT
    from src.models.scr import SpectrumConsistencyRegularizer
    from src.training.loss import WaPIGTLoss

    config = Config.from_yaml("config.yaml")
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
        from scripts.g2_definitive_runs import quick_eval as qe
        v = qe(model, val_l, is_wapigt)
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
    return best_val, best_state, metrics


def main():
    print(f"Device: {device}", flush=True)
    results = {"start_time": datetime.now().isoformat(), "runs": [],
               "note": "Reproduction run for checkpoint release only; "
                       "outputs/g2_definitive_results.json (paper source of truth) untouched."}
    t0 = time.time()
    n_done = 0
    n_total = len(MODELS) * len(TASKS) * 2 * len(SEEDS)

    for regime, augment in [("noaug", False), ("RPST", True)]:
        for task in TASKS:
            loaders = jnu_loaders(task, augment)
            for name in MODELS:
                for seed in SEEDS:
                    run_t0 = time.time()
                    tag = f"{name}_{task}_{regime}_s{seed}"
                    try:
                        val, best_state, metrics = train_one_with_checkpoint(
                            name, loaders, seed, tie_keep=augment)
                        ckpt_path = CKPT_DIR / f"{tag}.pt"
                        torch.save({"model_state_dict": best_state,
                                    "model": name, "task": task,
                                    "regime": regime, "seed": seed,
                                    "test_metrics": metrics}, ckpt_path)
                        results["runs"].append({"model": name, "task": task,
                                                "regime": regime, "seed": seed,
                                                "status": "success", "val_acc": val,
                                                "checkpoint": str(ckpt_path),
                                                **metrics})
                        n_done += 1
                        elapsed = time.time() - t0
                        run_dt = time.time() - run_t0
                        eta = (elapsed / n_done) * (n_total - n_done)
                        print(f"[{n_done}/{n_total}] {tag}: acc={metrics['acc']:.4f} "
                              f"({run_dt:.0f}s this run, ETA {eta/60:.0f} min)", flush=True)
                    except Exception as e:
                        print(f"[{n_done+1}/{n_total}] {tag} FAILED: {e}", flush=True)
                        results["runs"].append({"model": name, "task": task,
                                                "regime": regime, "seed": seed,
                                                "status": "failed", "error": str(e)})
                        n_done += 1
                    with open(OUTPUT, "w") as f:
                        json.dump(results, f, indent=2)

    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT, "and", CKPT_DIR)


if __name__ == "__main__":
    main()
