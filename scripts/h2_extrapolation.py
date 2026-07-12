"""H2: Speed extrapolation. Can resonance-preserving transposition generalise the
model to operating speeds NEVER observed -- not even via augmentation?

JNU EX-UP task: train on 600 rpm only, test on 1000 rpm. Arms:
  - none        : no augmentation (train 600 only)
  - rpst_extrap : RPST transpose 600 -> {700,800,900} (dense intermediate; the
                  1000 rpm test speed is NEVER synthesised) -> EXTRAPOLATION
  - sra_extrap  : same targets but plain resampling (resonance shifted) -> control
  - rpst_interp : RPST transpose 600 -> {800,1000} (includes test speed) -> upper bound
Models: WaPIGT, TICNN, ViT1D; 3 seeds. acc/MF1/AUROC/kappa.
-> outputs/h2_extrapolation_results.json
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.signal import resample_poly
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
from scripts.test_tsm_aug import reorg_bp, TSM_N_FFT, TSM_HOP

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/h2_extrapolation_results.json"
LOGITS_DIR = Path("outputs/logits"); LOGITS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1337, 2025]

JNU_FS = 50000.0
JNU_SIG_LEN = 12000
JNU_N_TRAIN = 200
JNU_N_EVAL = 50
FAULT_LABELS = {"n": 0, "ib": 1, "ob": 2, "tb": 3}

TRAIN_RPM = 600
TEST_RPM = 1000
VAL_RPM = 800            # held-out val at an intermediate speed
ARMS = {
    "none":        {"targets": []},
    "rpst_extrap": {"targets": [700, 800, 900], "op": "tsm"},
    "sra_extrap":  {"targets": [700, 800, 900], "op": "resample"},
    "rpst_interp": {"targets": [800, 1000],     "op": "tsm"},
}


def tsm_stretch(y, rate):
    D = librosa.stft(y, n_fft=TSM_N_FFT, hop_length=TSM_HOP)
    D2 = librosa.phase_vocoder(D, rate=rate, hop_length=TSM_HOP)
    return librosa.istft(D2, hop_length=TSM_HOP, n_fft=TSM_N_FFT,
                         length=int(len(y) / rate)).astype(np.float32)


def resample_speed(y, rate):
    # rate = target/source; new length = len/rate -> up=source-ish; use ratio
    from math import gcd
    num = int(round(rate * 1000)); den = 1000
    g = gcd(num, den); num //= g; den //= g
    # resample_poly(x, up, down): len*up/down. For impulse rate x rate, time
    # compresses by 1/rate -> up=den... we want output ~ len/rate -> up=den, down=num.
    return resample_poly(y, den, num).astype(np.float32)


class ExtrapJNU(RawBearingDataset):
    def __init__(self, speed_rpm, n_per, targets=None, op="tsm"):
        self._speed_rpm = speed_rpm
        self._n_per = n_per
        self._targets = targets or []
        self._op = op
        super().__init__(dataset="JNU", raw_root=str(config.data.jnu_raw_root),
                         speed_list=[f"{speed_rpm}rpm"], split="train",
                         n_samples_per_bearing=n_per, signal_length=JNU_SIG_LEN,
                         task_name="EX", test_speed_list=[f"{TEST_RPM}rpm"])

    def _load_jnu_data(self):
        root = self.raw_root / "JNU"
        src_rpm = self._speed_rpm
        for csv_file in sorted(root.glob("*.csv")):
            stem = csv_file.stem
            prefix = stem.split("_")[0]
            fault = "".join(c for c in prefix if c.isalpha()).lower()
            spd = "".join(c for c in prefix if c.isdigit())
            if f"{spd}rpm" != f"{src_rpm}rpm":
                continue
            label = FAULT_LABELS.get(fault, 0)
            sig = pd.read_csv(csv_file).iloc[:, 0].values.astype(np.float32)
            variants = [(sig, src_rpm)]
            for tgt in self._targets:
                rate = tgt / src_rpm
                w = tsm_stretch(sig, rate) if self._op == "tsm" else resample_speed(sig, rate)
                variants.append((w, tgt))
            for s, rpm in variants:
                f_shaft = rpm / 60.0
                ms = max(1, len(s) - self.signal_length)
                for i in range(self._n_per):
                    st = (i * self.signal_length) % ms
                    win = s[st:st + self.signal_length]
                    if len(win) < self.signal_length:
                        win = np.pad(win, (0, self.signal_length - len(win)))
                    self.samples.append(win)
                    self.labels.append(label)
                    self.bearing_ids.append(f"{stem}@{rpm}")
                    self.speeds.append(f"{rpm}rpm")
                    self.shaft_frequencies.append(f_shaft)


def loaders(arm):
    cfg = ARMS[arm]
    kw = dict(batch_size=32, num_workers=0, pin_memory=False)
    tr = ExtrapJNU(TRAIN_RPM, JNU_N_TRAIN, cfg["targets"], cfg.get("op", "tsm"))
    vl = ExtrapJNU(VAL_RPM, JNU_N_EVAL)
    te = ExtrapJNU(TEST_RPM, JNU_N_EVAL)
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(vl, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw))


@torch.no_grad()
def quick_eval(model, loader, is_w):
    model.eval(); c = t = 0
    for batch in loader:
        sigs = batch["signal"].to(device); labs = batch["label"].to(device)
        if is_w:
            bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
            logits, *_ = model(sigs, bp, fs_sampling=JNU_FS)
        else:
            out = model(sigs); logits = out[0] if isinstance(out, tuple) else out
        c += (logits.argmax(1) == labs).sum().item(); t += labs.size(0)
    return c / max(t, 1)


def train(name, lds, seed, tie_keep):
    train_l, val_l, test_l = lds
    set_all_seeds(seed)
    is_w = name == "WaPIGT"
    if is_w:
        model = WaPIGT(n_classes=4, hidden_dim=config.model.hidden_dim,
                       n_encoder_layers=config.model.n_encoder_layers,
                       n_heads=config.model.n_heads, mlp_dim=config.model.mlp_dim,
                       dropout=config.model.dropout, n_gat_heads=config.model.n_gat_heads,
                       gat_dropout=config.model.gat_dropout).to(device)
        scr = SpectrumConsistencyRegularizer(sigma=2.0)
        loss_fn = WaPIGTLoss(n_classes=4, scr_module=scr, scr_lambda=config.model.scr_lambda,
                             scr_warmup_epochs=config.model.scr_warmup_epochs, n_epochs=120,
                             triplet_lambda=0.0, triplet_margin=0.5, triplet_warmup_epochs=9999)
        opt = optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                          weight_decay=config.training.weight_decay)
        n_epochs = 120
    else:
        model = BASELINE_REGISTRY[name](signal_length=JNU_SIG_LEN, n_classes=4, dropout=0.3).to(device)
        loss_fn = None
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        n_epochs = 80
    best_val, best_state, no_improve = 0.0, None, 0
    for epoch in range(n_epochs):
        if loss_fn:
            loss_fn.set_epoch(epoch)
        model.train()
        for batch in train_l:
            sigs = batch["signal"].to(device); labs = batch["label"].to(device)
            opt.zero_grad()
            if is_w:
                bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
                logits, attn, emb = model(sigs, bp, fs_sampling=JNU_FS)
                loss = loss_fn(logits, labs, attn, None, sigs.shape[-1], JNU_FS, embeddings=emb)
            else:
                out = model(sigs); logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits, labs)
            if torch.isnan(loss):
                opt.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = quick_eval(model, val_l, is_w)
        if v >= best_val:
            if v > best_val:
                no_improve = 0
            else:
                no_improve += 1
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= 20:
            break
    if best_state:
        model.load_state_dict(best_state)
    return best_val, evaluate_full(model, test_l, device, is_wapigt=is_w, fs=JNU_FS, reorg_bp=reorg_bp)


def main():
    print(f"Device: {device}  EX-UP: train {TRAIN_RPM} -> test {TEST_RPM} rpm")
    results = {"start_time": datetime.now().isoformat(), "runs": []}
    for arm in ["none", "rpst_extrap", "sra_extrap", "rpst_interp"]:
        lds = loaders(arm)
        tie = arm != "none"
        print(f"[{arm}] train={len(lds[0].dataset)} (targets {ARMS[arm]['targets']})", flush=True)
        for name in ["WaPIGT", "TICNN", "ViT1D"]:
            accs = []
            for seed in SEEDS:
                try:
                    val, (m, lg, lb) = train(name, lds, seed, tie)
                    np.savez(LOGITS_DIR / f"h2_{name}_{arm}_s{seed}.npz", logits=lg, labels=lb)
                    print(f"{name}[{arm}] seed={seed}: val={val:.3f} acc={m['acc']:.4f} "
                          f"mf1={m['mf1']:.4f} auroc={m['auroc']:.4f} kappa={m['kappa']:.4f}", flush=True)
                    results["runs"].append({"model": name, "arm": arm, "seed": seed,
                                            "status": "success", "val_acc": val, **m})
                    accs.append(m["acc"])
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results["runs"].append({"model": name, "arm": arm, "seed": seed,
                                            "status": "failed", "error": str(e)})
                with open(OUTPUT, "w") as f:
                    json.dump(results, f, indent=2)
            if accs:
                print(f">>> {name}[{arm}]: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%\n", flush=True)
    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
