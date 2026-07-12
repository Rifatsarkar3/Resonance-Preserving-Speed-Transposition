"""Order-Normalized Resampling (ONR) test: WaPIGT-MS + ONR on JNU T01/T02/T03.

ONR = computed order tracking applied as input normalization. Every signal
(train/val/test) is resampled to a canonical shaft speed (800 rpm) using the
known rpm, so impulse density per window is speed-invariant. JNU speeds
600:800:1000 = 3:4:5 -> exact integer polyphase ratios.

Also runs TICNN+ONR as a fairness control.
Saves incrementally to outputs/onr_results.json.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.data_loaders.raw_dataset import RawBearingDataset
from src.baselines import BASELINE_REGISTRY
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/onr_results.json"
SEEDS = [42, 1337, 2025]

JNU_FS = 50000.0
JNU_SIG_LEN = 12000
JNU_N_TRAIN = 200
JNU_N_EVAL = 50
FAULT_LABELS = {"n": 0, "ib": 1, "ob": 2, "tb": 3}
SPEED_TO_FS = {"600rpm": 10.0, "800rpm": 13.33, "1000rpm": 16.67}
F_REF = 13.33  # canonical shaft frequency (800 rpm)
# resample_poly(x, up, down): new_len = len * up/down = len * f_actual/f_ref
ONR_RATIO = {"600rpm": (3, 4), "800rpm": (1, 1), "1000rpm": (5, 4)}

JNU_SPLITS = {
    "JNU_T01": {"train": ["600rpm"], "val": ["800rpm"], "test": ["1000rpm"]},
    "JNU_T02": {"train": ["800rpm"], "val": ["600rpm"], "test": ["1000rpm"]},
    "JNU_T03": {"train": ["1000rpm"], "val": ["800rpm"], "test": ["600rpm"]},
}


class ONRJNUDataset(RawBearingDataset):
    """JNU dataset with Order-Normalized Resampling to canonical 800 rpm."""

    def _load_jnu_data(self):
        jnu_root = self.raw_root / "JNU"
        for csv_file in sorted(jnu_root.glob("*.csv")):
            try:
                stem = csv_file.stem
                prefix = stem.split("_")[0]
                fault = "".join(c for c in prefix if c.isalpha()).lower()
                spd = "".join(c for c in prefix if c.isdigit())
                speed = f"{spd}rpm" if spd else "unknown"
                if self.speed_list and speed not in self.speed_list:
                    continue
                label = FAULT_LABELS.get(fault, 0)
                signal = pd.read_csv(csv_file).iloc[:, 0].values.astype(np.float32)
                up, down = ONR_RATIO.get(speed, (1, 1))
                if (up, down) != (1, 1):
                    signal = resample_poly(signal, up, down).astype(np.float32)
                max_start = max(1, len(signal) - self.signal_length)
                for i in range(self.n_samples_per_bearing):
                    start = (i * self.signal_length) % max_start
                    window = signal[start: start + self.signal_length]
                    if len(window) < self.signal_length:
                        window = np.pad(window, (0, self.signal_length - len(window)))
                    self.samples.append(window)
                    self.labels.append(label)
                    self.bearing_ids.append(stem)
                    self.speeds.append(speed)
                    # after ONR every signal is at the canonical shaft speed
                    self.shaft_frequencies.append(F_REF)
            except Exception as e:
                print(f"JNU load error {csv_file.name}: {e}")


def jnu_loaders(task, batch_size=32):
    sp = JNU_SPLITS[task]
    root = str(config.data.jnu_raw_root)

    def ds(speeds, n):
        return ONRJNUDataset(
            dataset="JNU", raw_root=root, speed_list=speeds, split="train",
            n_samples_per_bearing=n, signal_length=JNU_SIG_LEN,
            task_name=task, test_speed_list=sp["test"],
        )

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (DataLoader(ds(sp["train"], JNU_N_TRAIN), shuffle=True, **kw),
            DataLoader(ds(sp["val"], JNU_N_EVAL), shuffle=False, **kw),
            DataLoader(ds(sp["test"], JNU_N_EVAL), shuffle=False, **kw))


def reorg_bp(batch_bp, bs):
    out = []
    for i in range(bs):
        d = {}
        for k, v in batch_bp.items():
            if isinstance(v, torch.Tensor):
                d[k] = v[i].item() if v.dim() > 0 else v.item()
            elif isinstance(v, (list, tuple)):
                d[k] = v[i]
            else:
                d[k] = v
        out.append(d)
    return out


@torch.no_grad()
def evaluate(model, loader, is_wapigt=False):
    model.eval()
    correct = total = 0
    for batch in loader:
        sigs = batch["signal"].to(device)
        labs = batch["label"].to(device)
        if is_wapigt:
            bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
            logits, *_ = model(sigs, bp, fs_sampling=JNU_FS)
        else:
            out = model(sigs)
            logits = out[0] if isinstance(out, tuple) else out
        correct += (logits.argmax(1) == labs).sum().item()
        total += labs.size(0)
    return correct / max(total, 1)


def train_wapigt(loaders, seed, n_epochs=120, patience=20):
    train_l, val_l, test_l = loaders
    set_all_seeds(seed)
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
        scr_warmup_epochs=config.model.scr_warmup_epochs, n_epochs=n_epochs,
        triplet_lambda=getattr(config.model, 'triplet_lambda', 0.1),
        triplet_margin=getattr(config.model, 'triplet_margin', 0.5),
        triplet_warmup_epochs=getattr(config.model, 'triplet_warmup_epochs', 20),
    )
    opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)
    best_val, best_state, no_improve = 0.0, None, 0
    for epoch in range(n_epochs):
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
        v = evaluate(model, val_l, is_wapigt=True)
        if v > best_val:
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_val, evaluate(model, test_l, is_wapigt=True)


def train_baseline(name, loaders, seed, n_epochs=80, patience=20):
    train_l, val_l, test_l = loaders
    set_all_seeds(seed)
    sig_len = train_l.dataset[0]["signal"].shape[-1]
    model = BASELINE_REGISTRY[name](
        signal_length=sig_len, n_classes=4, dropout=0.3
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val, best_state, no_improve = 0.0, None, 0
    for epoch in range(n_epochs):
        model.train()
        for batch in train_l:
            sigs = batch["signal"].to(device)
            labs = batch["label"].to(device)
            opt.zero_grad()
            out = model(sigs)
            logits = out[0] if isinstance(out, tuple) else out
            F.cross_entropy(logits, labs).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, val_l)
        if v > best_val:
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_val, evaluate(model, test_l)


def main():
    print(f"Device: {device}")
    Path("outputs").mkdir(exist_ok=True)
    results = {"start_time": datetime.now().isoformat(), "runs": []}

    # Priority order: WaPIGT on T01 (weakest task) first, then T03/T02,
    # then TICNN+ONR control on all tasks.
    jobs = [("WaPIGT-MS+ONR", t) for t in ["JNU_T01", "JNU_T03", "JNU_T02"]]
    jobs += [("TICNN+ONR", t) for t in ["JNU_T01", "JNU_T03", "JNU_T02"]]

    for model_name, task in jobs:
        accs = []
        for seed in SEEDS:
            loaders = jnu_loaders(task)
            try:
                if model_name.startswith("WaPIGT"):
                    val, test = train_wapigt(loaders, seed)
                else:
                    val, test = train_baseline("TICNN", loaders, seed)
                print(f"{model_name} {task} seed={seed}: val={val:.4f} test={test:.4f}", flush=True)
                results["runs"].append({"model": model_name, "task": task,
                                        "seed": seed, "status": "success",
                                        "val_acc": val, "test_acc": test})
                accs.append(test)
            except Exception as e:
                print(f"{model_name} {task} seed={seed} FAILED: {e}", flush=True)
                results["runs"].append({"model": model_name, "task": task,
                                        "seed": seed, "status": "failed",
                                        "error": str(e)})
            with open(OUTPUT, "w") as f:
                json.dump(results, f, indent=2)
        if accs:
            print(f">>> {model_name} {task}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n", flush=True)

    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
