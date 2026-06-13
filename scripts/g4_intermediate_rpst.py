"""G4: leakage control — RPST transposing ONLY to the intermediate speed.

JNU T01 (train 600rpm, test 1000rpm): augment 600->800 ONLY (never the test
speed). If most of the RPST gain survives, the "augmenting to the known test
speed" critique fails. WaPIGT canonical config, 3 seeds.
-> outputs/g4_intermediate_results.json
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.test_tsm_aug as T
from scripts.test_tsm_aug import TSMAugJNUDataset, train_wapigt, JNU_SIG_LEN

OUTPUT = "outputs/g4_intermediate_results.json"
SEEDS = [42, 1337, 2025]


class IntermediateTSMDataset(TSMAugJNUDataset):
    """Transpose only to 800rpm (intermediate), never to the 1000rpm test speed."""
    pass


def loaders():
    # restrict augmentation targets to the intermediate speed only
    T.ALL_SPEEDS = ["600rpm", "800rpm"]  # source 600 -> target 800 only
    sp = T.JNU_SPLITS["JNU_T01"]
    root = str(T.config.data.jnu_raw_root)

    def ds(speeds, n, aug=False):
        return IntermediateTSMDataset(
            dataset="JNU", raw_root=root, speed_list=speeds, split="train",
            n_samples_per_bearing=n, signal_length=JNU_SIG_LEN,
            task_name="JNU_T01", test_speed_list=sp["test"], augment=aug)

    kw = dict(batch_size=32, num_workers=0, pin_memory=False)
    return (DataLoader(ds(sp["train"], T.JNU_N_TRAIN, aug=True), shuffle=True, **kw),
            DataLoader(ds(sp["val"], T.JNU_N_EVAL), shuffle=False, **kw),
            DataLoader(ds(sp["test"], T.JNU_N_EVAL), shuffle=False, **kw))


def main():
    results = {"start_time": datetime.now().isoformat(), "runs": []}
    ld = loaders()
    tgt_speeds = sorted(set(ld[0].dataset.speeds))
    print("train speeds present (must NOT include 1000rpm):", tgt_speeds, flush=True)
    assert "1000rpm" not in tgt_speeds, "leakage: test speed present in train"
    accs = []
    for seed in SEEDS:
        val, test = train_wapigt(ld, seed)
        print(f"WaPIGT+RPST(800-only) JNU_T01 seed={seed}: val={val:.4f} test={test:.4f}", flush=True)
        results["runs"].append({"model": "WaPIGT+RPST-intermediate", "task": "JNU_T01",
                                "seed": seed, "val_acc": val, "test_acc": test})
        accs.append(test)
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
    print(f">>> WaPIGT+RPST(800-only) JNU_T01: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%")
    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
