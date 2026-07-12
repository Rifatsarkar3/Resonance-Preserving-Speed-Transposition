"""G1: PU cross-speed validation of RPST (second dataset, MSSP requirement).

Tasks (only speed differs; torque and radial force held fixed at M07/F10):
  PU_S1: train N09_M07_F10 (900 rpm) -> test N15_M07_F10 (1500 rpm), up x1.67
  PU_S2: train N15_M07_F10 -> test N09_M07_F10, down x0.60
Val = held-out repetition files at the TRAIN condition (no test-speed leakage).

Corrected loader (legacy _load_pu_data bugs: every bearing matched startswith("K")
-> all labels 0; N09/N15 mapped to 9/15 Hz instead of 15/25 Hz; only mat_files[0]
read regardless of condition; silent synthetic-data fallback).
Bearing: 6203 (N=8, d=6.75 mm, D=29.05 mm).

Models: WaPIGT (canonical full config), TICNN, ViT-1D; +/-RPST; 3 seeds.
Logits saved; metrics: acc/MF1/AUROC/kappa -> outputs/g1_pu_results.json
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime

import scipy.io as sio
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.data_loaders.raw_dataset import RawBearingDataset
from src.baselines import BASELINE_REGISTRY
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss
from scripts.metrics_util import evaluate_full

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/g1_pu_results.json"
LOGITS_DIR = Path("outputs/logits")
LOGITS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1337, 2025]

PU_FS = 64000.0
PU_SIG_LEN = 16000
PU_BEARING = {"N_balls": 8, "d_mm": 6.75, "D_mm": 29.05, "alpha_deg": 0.0}
COND_FS = {"N09_M07_F10": 15.0, "N15_M07_F10": 25.0}
COND_RPM = {"N09_M07_F10": 900.0, "N15_M07_F10": 1500.0}
CLASS_BEARINGS = {
    0: ["K001", "K002", "K003", "K004"],      # healthy
    1: ["KA01", "KA03", "KA04", "KA05"],      # outer race
    2: ["KI01", "KI03", "KI04", "KI05"],      # inner race
    3: ["KB23", "KB24", "KB27"],              # combined
}
TRAIN_REPS = [1, 2]
VAL_REPS = [15]
TEST_REPS = [1, 2]
N_TRAIN_PER_FILE = 30
N_VAL_PER_FILE = 15
N_TEST_PER_FILE = 15

TSM_N_FFT = 256
TSM_HOP = 64

TASKS = {
    "PU_S1": {"train": "N09_M07_F10", "test": "N15_M07_F10"},
    "PU_S2": {"train": "N15_M07_F10", "test": "N09_M07_F10"},
}

_signal_cache = {}


def load_vibration(path):
    if path in _signal_cache:
        return _signal_cache[path]
    d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    key = next(k for k in d if not k.startswith("__"))
    sig = None
    for ch in d[key].Y:
        if getattr(ch, "Name", "") == "vibration_1":
            sig = np.asarray(ch.Data, dtype=np.float32).flatten()
            break
    if sig is None:
        raise RuntimeError(f"vibration_1 channel not found in {path}")
    _signal_cache[path] = sig
    return sig


def tsm_stretch(y, rate):
    D = librosa.stft(y, n_fft=TSM_N_FFT, hop_length=TSM_HOP)
    D2 = librosa.phase_vocoder(D, rate=rate, hop_length=TSM_HOP)
    return librosa.istft(D2, hop_length=TSM_HOP, n_fft=TSM_N_FFT,
                         length=int(len(y) / rate)).astype(np.float32)


class PUCrossSpeedDataset(RawBearingDataset):
    """Condition-filtered PU dataset with optional RPST augmentation."""

    def __init__(self, condition, reps, n_per_file, augment=False, aug_target=None):
        self._condition = condition
        self._reps = reps
        self._n_per_file = n_per_file
        self._augment = augment
        self._aug_target = aug_target
        super().__init__(dataset="PU", raw_root=str(config.data.pu_raw_root),
                         bearing_list=[], split="train",
                         n_samples_per_bearing=n_per_file,
                         signal_length=PU_SIG_LEN)
        # corrected bearing geometry (6203) and sampling rate
        self.bearing_params = dict(PU_BEARING)
        self.fs_sampling = PU_FS

    def _load_pu_data(self):
        pu_root = self.raw_root / "PU"
        f_src = COND_FS[self._condition]
        for label, bearings in CLASS_BEARINGS.items():
            for bid in bearings:
                for rep in self._reps:
                    path = pu_root / bid / f"{self._condition}_{bid}_{rep}.mat"
                    if not path.exists():
                        continue
                    sig = load_vibration(path)
                    variants = [(sig, f_src)]
                    if self._augment and self._aug_target:
                        rate = COND_RPM[self._aug_target] / COND_RPM[self._condition]
                        variants.append((tsm_stretch(sig, rate),
                                         COND_FS[self._aug_target]))
                    for s, f_shaft in variants:
                        max_start = max(1, len(s) - self.signal_length)
                        for i in range(self._n_per_file):
                            start = (i * self.signal_length) % max_start
                            w = s[start: start + self.signal_length]
                            if len(w) < self.signal_length:
                                w = np.pad(w, (0, self.signal_length - len(w)))
                            self.samples.append(w.astype(np.float32))
                            self.labels.append(label)
                            self.bearing_ids.append(f"{bid}@{f_shaft}")
                            self.speeds.append(self._condition)
                            self.shaft_frequencies.append(f_shaft)


def pu_loaders(task, augment):
    t = TASKS[task]
    kw = dict(batch_size=32, num_workers=0, pin_memory=False)
    tr = PUCrossSpeedDataset(t["train"], TRAIN_REPS, N_TRAIN_PER_FILE,
                             augment=augment, aug_target=t["test"] if augment else None)
    vl = PUCrossSpeedDataset(t["train"], VAL_REPS, N_VAL_PER_FILE)
    te = PUCrossSpeedDataset(t["test"], TEST_REPS, N_TEST_PER_FILE)
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(vl, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw))


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
def quick_eval(model, loader, is_wapigt):
    model.eval()
    c = t = 0
    for batch in loader:
        sigs = batch["signal"].to(device)
        labs = batch["label"].to(device)
        if is_wapigt:
            bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
            logits, *_ = model(sigs, bp, fs_sampling=PU_FS)
        else:
            out = model(sigs)
            logits = out[0] if isinstance(out, tuple) else out
        c += (logits.argmax(1) == labs).sum().item()
        t += labs.size(0)
    return c / max(t, 1)


def train_model(name, loaders, seed, tie_keep):
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
            signal_length=PU_SIG_LEN, n_classes=4, dropout=0.3).to(device)
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
                logits, attn, emb = model(sigs, bp, fs_sampling=PU_FS)
                ffb = batch.get("fault_freq_bins")
                loss = loss_fn(logits, labs, attn,
                               ffb.to(device) if ffb is not None else None,
                               sigs.shape[-1], PU_FS, embeddings=emb)
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
                                            is_wapigt=is_wapigt, fs=PU_FS,
                                            reorg_bp=reorg_bp)
    return best_val, metrics, logits, labels


def main():
    print(f"Device: {device}")
    results = {"start_time": datetime.now().isoformat(), "runs": []}
    for task in ["PU_S1", "PU_S2"]:
        for regime, augment in [("noaug", False), ("RPST", True)]:
            loaders = pu_loaders(task, augment)
            print(f"{task} {regime}: train={len(loaders[0].dataset)} "
                  f"val={len(loaders[1].dataset)} test={len(loaders[2].dataset)}", flush=True)
            for name in ["WaPIGT", "TICNN", "ViT1D"]:
                accs = []
                for seed in SEEDS:
                    try:
                        val, metrics, logits, labels = train_model(
                            name, loaders, seed, tie_keep=augment)
                        tag = f"{name}_{task}_{regime}_s{seed}"
                        np.savez(LOGITS_DIR / f"g1_{tag}.npz",
                                 logits=logits, labels=labels)
                        print(f"{name}[{regime}] {task} seed={seed}: val={val:.4f} "
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
                          f"{np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n", flush=True)
    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
