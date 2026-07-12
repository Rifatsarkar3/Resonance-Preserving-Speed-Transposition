"""Resonance-preserving speed augmentation (TSM) test on JNU.

Plain resampling scales impulse rate AND all frequency content; a real speed
change scales only the impulse rate (structural resonances are fixed in Hz).
Phase-vocoder time-scale modification changes tempo without shifting
frequencies -> physically faithful speed transposition.

Each training signal is replicated at all 3 JNU speeds via TSM with the shaft
frequency scaled to the simulated speed. Val/test untouched.
Identical protocol to test_speed_aug.py except the warp operator.

Saves incrementally to outputs/tsm_aug_results.json.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.data_loaders.raw_dataset import RawBearingDataset
from src.models.wapigt import WaPIGT
from src.models.scr import SpectrumConsistencyRegularizer
from src.training.loss import WaPIGTLoss

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/tsm_aug_results.json"
SEEDS = [42, 1337, 2025]

JNU_FS = 50000.0
JNU_SIG_LEN = 12000
JNU_N_TRAIN = 200
JNU_N_EVAL = 50
FAULT_LABELS = {"n": 0, "ib": 1, "ob": 2, "tb": 3}
SPEED_TO_FS = {"600rpm": 10.0, "800rpm": 13.33, "1000rpm": 16.67}
SPEED_RPM = {"600rpm": 600.0, "800rpm": 800.0, "1000rpm": 1000.0}
ALL_SPEEDS = ["600rpm", "800rpm", "1000rpm"]

# short windows: impulse spacing at 600 rpm is ~100 ms; ringing ~ms scale.
# n_fft=256 @ 50 kHz = 5.1 ms -> preserves transients, ~195 Hz freq resolution
TSM_N_FFT = 256
TSM_HOP = 64

JNU_SPLITS = {
    "JNU_T01": {"train": ["600rpm"], "val": ["800rpm"], "test": ["1000rpm"]},
    "JNU_T02": {"train": ["800rpm"], "val": ["600rpm"], "test": ["1000rpm"]},
    "JNU_T03": {"train": ["1000rpm"], "val": ["800rpm"], "test": ["600rpm"]},
}


def tsm_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-scale modification: rate>1 speeds up (shorter output),
    frequency content preserved."""
    D = librosa.stft(y, n_fft=TSM_N_FFT, hop_length=TSM_HOP)
    D2 = librosa.phase_vocoder(D, rate=rate, hop_length=TSM_HOP)
    return librosa.istft(D2, hop_length=TSM_HOP, n_fft=TSM_N_FFT,
                         length=int(len(y) / rate)).astype(np.float32)


class TSMAugJNUDataset(RawBearingDataset):
    def __init__(self, *args, augment=False, **kwargs):
        self.augment = augment
        super().__init__(*args, **kwargs)

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

                variants = [(signal, speed)]
                if self.augment:
                    for tgt_speed in ALL_SPEEDS:
                        if tgt_speed == speed:
                            continue
                        rate = SPEED_RPM[tgt_speed] / SPEED_RPM[speed]
                        warped = tsm_stretch(signal, rate)
                        variants.append((warped, tgt_speed))

                for sig, eff_speed in variants:
                    f_shaft = SPEED_TO_FS[eff_speed]
                    max_start = max(1, len(sig) - self.signal_length)
                    for i in range(self.n_samples_per_bearing):
                        start = (i * self.signal_length) % max_start
                        window = sig[start: start + self.signal_length]
                        if len(window) < self.signal_length:
                            window = np.pad(window, (0, self.signal_length - len(window)))
                        self.samples.append(window)
                        self.labels.append(label)
                        self.bearing_ids.append(f"{stem}@{eff_speed}")
                        self.speeds.append(eff_speed)
                        self.shaft_frequencies.append(f_shaft)
            except Exception as e:
                print(f"JNU load error {csv_file.name}: {e}")


def jnu_loaders(task, batch_size=32):
    sp = JNU_SPLITS[task]
    root = str(config.data.jnu_raw_root)

    def ds(speeds, n, aug=False):
        return TSMAugJNUDataset(
            dataset="JNU", raw_root=root, speed_list=speeds, split="train",
            n_samples_per_bearing=n, signal_length=JNU_SIG_LEN,
            task_name=task, test_speed_list=sp["test"], augment=aug,
        )

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (DataLoader(ds(sp["train"], JNU_N_TRAIN, aug=True), shuffle=True, **kw),
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
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        sigs = batch["signal"].to(device)
        labs = batch["label"].to(device)
        bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
        logits, *_ = model(sigs, bp, fs_sampling=JNU_FS)
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
        v = evaluate(model, val_l)
        # keep LATEST tie checkpoint; reset patience only on strict improvement
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
    return best_val, evaluate(model, test_l)


def main():
    print(f"Device: {device}")
    Path("outputs").mkdir(exist_ok=True)
    results = {"start_time": datetime.now().isoformat(), "runs": []}

    for task in ["JNU_T01", "JNU_T03", "JNU_T02"]:
        accs = []
        loaders = jnu_loaders(task)  # TSM is deterministic; share across seeds
        for seed in SEEDS:
            try:
                val, test = train_wapigt(loaders, seed)
                print(f"WaPIGT-MS+TSM {task} seed={seed}: val={val:.4f} test={test:.4f}", flush=True)
                results["runs"].append({"model": "WaPIGT-MS+TSM", "task": task,
                                        "seed": seed, "status": "success",
                                        "val_acc": val, "test_acc": test})
                accs.append(test)
            except Exception as e:
                print(f"WaPIGT-MS+TSM {task} seed={seed} FAILED: {e}", flush=True)
                results["runs"].append({"model": "WaPIGT-MS+TSM", "task": task,
                                        "seed": seed, "status": "failed",
                                        "error": str(e)})
            with open(OUTPUT, "w") as f:
                json.dump(results, f, indent=2)
        if accs:
            print(f">>> WaPIGT-MS+TSM {task}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n", flush=True)

    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
