"""G5: baseline learning-rate sweep (principled-comparison defense).

Each baseline: lr in {3e-4, 1e-3 (default, already measured), 3e-3} on JNU T02
no-aug, seed 42. If any alternative beats the default by >1pp val, flag it.
-> outputs/g5_lr_sweep_results.json
"""
import sys, json
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_all_seeds
from src.baselines import BASELINE_REGISTRY
import scripts.test_tsm_aug as T
from scripts.test_tsm_aug import TSMAugJNUDataset, JNU_SIG_LEN

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/g5_lr_sweep_results.json"
MODELS = ["WDCNN", "TICNN", "MSCNN", "PhysFormer", "ViT1D"]
LRS = [3e-4, 3e-3]
TASK = "JNU_T02"
SEED = 42


def loaders():
    sp = T.JNU_SPLITS[TASK]
    root = str(T.config.data.jnu_raw_root)

    def ds(speeds, n):
        return TSMAugJNUDataset(
            dataset="JNU", raw_root=root, speed_list=speeds, split="train",
            n_samples_per_bearing=n, signal_length=JNU_SIG_LEN,
            task_name=TASK, test_speed_list=sp["test"], augment=False)

    kw = dict(batch_size=32, num_workers=0, pin_memory=False)
    return (DataLoader(ds(sp["train"], T.JNU_N_TRAIN), shuffle=True, **kw),
            DataLoader(ds(sp["val"], T.JNU_N_EVAL), shuffle=False, **kw),
            DataLoader(ds(sp["test"], T.JNU_N_EVAL), shuffle=False, **kw))


@torch.no_grad()
def ev(model, loader):
    model.eval()
    c = t = 0
    for b in loader:
        out = model(b["signal"].to(device))
        lg = out[0] if isinstance(out, tuple) else out
        c += (lg.argmax(1) == b["label"].to(device)).sum().item()
        t += b["label"].size(0)
    return c / max(t, 1)


def train(name, ld, lr, n_epochs=80, patience=20):
    train_l, val_l, test_l = ld
    set_all_seeds(SEED)
    model = BASELINE_REGISTRY[name](signal_length=JNU_SIG_LEN, n_classes=4,
                                    dropout=0.3).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val, best_state, ni = 0.0, None, 0
    for ep in range(n_epochs):
        model.train()
        for b in train_l:
            sigs = b["signal"].to(device)
            labs = b["label"].to(device)
            opt.zero_grad()
            out = model(sigs)
            lg = out[0] if isinstance(out, tuple) else out
            F.cross_entropy(lg, labs).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = ev(model, val_l)
        if v > best_val:
            best_val, ni = v, 0
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            ni += 1
            if ni >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_val, ev(model, test_l)


def main():
    results = {"start_time": datetime.now().isoformat(), "runs": []}
    ld = loaders()
    for name in MODELS:
        for lr in LRS:
            val, test = train(name, ld, lr)
            print(f"{name} lr={lr:g}: val={val:.4f} test={test:.4f}", flush=True)
            results["runs"].append({"model": name, "lr": lr, "task": TASK,
                                    "seed": SEED, "val_acc": val, "test_acc": test})
            with open(OUTPUT, "w") as f:
                json.dump(results, f, indent=2)
    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Done ->", OUTPUT)


if __name__ == "__main__":
    main()
