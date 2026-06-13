"""
CWRU Cross-Load Transfer Comparison: WaPIGT vs 5 Baselines
8 tasks × 5 seeds × 6 models = 240 runs. Saves intermediate results after every run.
"""
import sys, json, logging, numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.data_loaders.raw_dataset import RawBearingDataset
from src.baselines import BASELINE_REGISTRY
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("cwru_comparison.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

SEEDS = [42, 1337, 2025]                          # 3 seeds
TASKS = ["CWRU_T01", "CWRU_T05"]                 # 2 representative tasks (T01: 0+1HP→3HP, T05: 0+2HP→1HP)
BASELINES = ["WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D"]
N_EPOCHS_BASELINE = 50
N_EPOCHS_WAPIGT = 50
PATIENCE_WAPIGT = 15
N_TRAIN = 10    # samples per bearing file (fast but enough)
N_EVAL  = 5    # samples per bearing file for val/test
OUTPUT_FILE = "outputs/cwru_comparison_results.json"


def load_cwru_loaders(task: str, n_train: int = 20, n_eval: int = 10, bs: int = 32):
    splits_file = Path(config.data.splits_root) / "cwru_splits.json"
    with open(splits_file) as f:
        splits = json.load(f)
    t = splits[task]
    raw_root = Path(config.data.cwru_raw_root)

    def make_ds(key, split, n):
        return RawBearingDataset(
            dataset="CWRU", raw_root=raw_root,
            bearing_list=t[key], split=split,
            n_samples_per_bearing=n, signal_length=12000,
        )

    train_ds = make_ds("train", "train", n_train)
    val_ds   = make_ds("val",   "val",   n_eval)
    test_ds  = make_ds("test",  "test",  n_eval)
    kw = dict(batch_size=bs, num_workers=0, pin_memory=False)
    n_classes = t["n_classes"]
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw),
            n_classes)


CWRU_FS = 12000.0  # CWRU Drive End sensor sampling rate


def reorganize_bearing_params(batch_bp, batch_size):
    """Convert DataLoader-collated {key: tensor} to [{key: scalar}, ...]."""
    params_list = []
    for i in range(batch_size):
        d = {}
        for key, val in batch_bp.items():
            if isinstance(val, torch.Tensor):
                d[key] = val[i].item() if val.dim() > 0 else val.item()
            elif isinstance(val, (list, tuple)):
                d[key] = val[i]
            else:
                d[key] = val
        params_list.append(d)
    return params_list


def eval_model(model, loader, is_wapigt=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            sigs = batch["signal"].to(device)
            labs = batch["label"].to(device)
            if is_wapigt:
                bp_list = reorganize_bearing_params(batch["bearing_params"], sigs.shape[0])
                logits, *_ = model(sigs, bp_list, fs_sampling=CWRU_FS)
            else:
                out = model(sigs)
                logits = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == labs).sum().item()
            total += labs.size(0)
    return correct / total if total > 0 else 0.0


def train_baseline(name, task, seed):
    set_all_seeds(seed)
    train_l, val_l, test_l, n_classes = load_cwru_loaders(task, n_train=N_TRAIN, n_eval=N_EVAL)
    if train_l is None:
        return {"status": "failed"}

    model_cls = BASELINE_REGISTRY[name]
    model = model_cls(signal_length=12000, n_classes=n_classes, dropout=0.3).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val = 0.0
    best_state = None
    for epoch in range(N_EPOCHS_BASELINE):
        model.train()
        for batch in train_l:
            sigs = batch["signal"].to(device)
            labs = batch["label"].to(device)
            opt.zero_grad()
            out = model(sigs)
            logits = out[0] if isinstance(out, tuple) else out
            F.cross_entropy(logits, labs).backward()
            opt.step()
        val_acc = eval_model(model, val_l)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    test_acc = eval_model(model, test_l)
    return {"status": "success", "val_acc": best_val, "test_acc": test_acc}


def train_wapigt(task, seed):
    set_all_seeds(seed)
    train_l, val_l, test_l, n_classes = load_cwru_loaders(task, n_train=N_TRAIN, n_eval=N_EVAL)
    if train_l is None:
        return {"status": "failed"}

    model = WaPIGT(
        n_classes=n_classes, hidden_dim=config.model.hidden_dim,
        n_encoder_layers=config.model.n_encoder_layers, n_heads=config.model.n_heads,
        mlp_dim=config.model.mlp_dim, dropout=config.model.dropout,
        n_gat_heads=config.model.n_gat_heads, gat_dropout=config.model.gat_dropout,
        filter_order=config.model.filter_order, n_frames=config.model.n_frames,
    ).to(device)

    scr = SpectrumConsistencyRegularizer(sigma=2.0)
    loss_fn = WaPIGTLoss(
        n_classes=n_classes, scr_module=scr,
        scr_lambda=config.model.scr_lambda,
        scr_warmup_epochs=config.model.scr_warmup_epochs,
        n_epochs=N_EPOCHS_WAPIGT,
        triplet_lambda=getattr(config.model, 'triplet_lambda', 0.1),
        triplet_margin=getattr(config.model, 'triplet_margin', 0.5),
    )
    opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)

    best_val = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(N_EPOCHS_WAPIGT):
        loss_fn.set_epoch(epoch)
        model.train()
        for batch in train_l:

            sigs = batch["signal"].to(device)
            labs = batch["label"].to(device)
            bp_list = reorganize_bearing_params(batch["bearing_params"], sigs.shape[0])
            opt.zero_grad()
            logits, attn, embeddings = model(sigs, bp_list, fs_sampling=CWRU_FS)
            loss = loss_fn(logits, labs, attn,
                          batch["fault_freq_bins"].to(device) if "fault_freq_bins" in batch else None,
                          sigs.shape[-1], CWRU_FS,
                          embeddings=embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_acc = eval_model(model, val_l, is_wapigt=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_WAPIGT:
                break

    if best_state:
        model.load_state_dict(best_state)
    test_acc = eval_model(model, test_l, is_wapigt=True)
    return {"status": "success", "val_acc": best_val, "test_acc": test_acc}


def main():
    # Build all runs: WaPIGT first, then baselines
    runs = []
    for task in TASKS:
        for seed in SEEDS:
            runs.append(("WaPIGT", task, seed))
    for bl in BASELINES:
        for task in TASKS:
            for seed in SEEDS:
                runs.append((bl, task, seed))

    total = len(runs)
    logger.info(f"CWRU Cross-Load Comparison: {total} runs")

    results = {"start_time": datetime.now().isoformat(), "runs": []}
    Path("outputs").mkdir(exist_ok=True)

    for i, (model_name, task, seed) in enumerate(runs, 1):
        logger.info(f"[{i}/{total}] {model_name:12s} | {task} | seed={seed}")
        try:
            if model_name == "WaPIGT":
                res = train_wapigt(task, seed)
            else:
                res = train_baseline(model_name, task, seed)
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            res = {"status": "failed", "error": str(e)}

        if res.get("status") == "success":
            logger.info(f"  val={res['val_acc']:.4f}  test={res['test_acc']:.4f}")

        results["runs"].append({"model": model_name, "task": task, "seed": seed, **res})

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    results["end_time"] = datetime.now().isoformat()
    successes = [r for r in results["runs"] if r["status"] == "success"]

    logger.info("\n" + "=" * 70)
    logger.info("CWRU CROSS-LOAD RESULTS")
    logger.info("=" * 70)
    by_model = {}
    for r in successes:
        by_model.setdefault(r["model"], []).append(r["test_acc"])
    for m, accs in sorted(by_model.items()):
        logger.info(f"  {m:12s}: {np.mean(accs):.4f} ± {np.std(accs):.4f}  (n={len(accs)})")

    results["model_summary"] = {
        m: {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "n": len(accs)}
        for m, accs in by_model.items()
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Done. Results → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
