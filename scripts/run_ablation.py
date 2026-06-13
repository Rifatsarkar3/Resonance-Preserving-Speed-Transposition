"""
Ablation study for WaPIGT components on JNU T03 (the winning task).
4 configs: ABL-0 (Base Transformer) → ABL-1 (+MST) → ABL-2 (+MST+PIFFG) → ABL-3 (Full)
3 seeds. Results saved to outputs/ablation_results.json.
"""
import sys, json, logging, time, numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import Config
from src.utils.reproducibility import set_all_seeds
from src.data_loaders.raw_dataset import RawBearingDataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler("ablation.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

config = Config.from_yaml("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "outputs/ablation_results.json"
SEEDS = [42, 1337, 2025]

SPEED_TO_FS = {"600rpm": 10.0, "800rpm": 13.33, "1000rpm": 16.67}
FAULT_LABELS = {"n": 0, "ib": 1, "ob": 2, "tb": 3}
JNU_FS = 50000.0
JNU_SIG_LEN = 12000
JNU_N_TRAIN = 200
JNU_N_EVAL = 50


class FixedJNUDataset(RawBearingDataset):
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
                f_shaft = SPEED_TO_FS.get(speed, 16.67)
                df = pd.read_csv(csv_file)
                signal = df.iloc[:, 0].values.astype(np.float32)
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
                    self.shaft_frequencies.append(f_shaft)
            except Exception as e:
                pass


def get_loaders(batch_size=32):
    root = str(config.data.jnu_raw_root)
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    def ds(speeds, n):
        return FixedJNUDataset(
            dataset="JNU", raw_root=root,
            speed_list=speeds, split="train",
            n_samples_per_bearing=n, signal_length=JNU_SIG_LEN,
            task_name="JNU_T03", test_speed_list=["600rpm"],
        )
    return (DataLoader(ds(["1000rpm"], JNU_N_TRAIN), shuffle=True, **kw),
            DataLoader(ds(["800rpm"], JNU_N_EVAL), shuffle=False, **kw),
            DataLoader(ds(["600rpm"], JNU_N_EVAL), shuffle=False, **kw))


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


def build_abl0(n_classes, seed, hidden_dim=96):
    """ABL-0: Base Transformer with ViT-style patch embedding (no MST, no PIFFG)."""
    import torch.nn as nn
    set_all_seeds(seed)

    class BaseTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            patch_size = 48  # 12000 / 250 = 48
            n_tokens = JNU_SIG_LEN // patch_size  # 250 tokens
            self.patch_embed = nn.Linear(patch_size, hidden_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.pos_enc = nn.Parameter(torch.randn(1, n_tokens + 1, hidden_dim) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=384,
                dropout=0.1, batch_first=True, activation='gelu', norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, n_classes)

        def forward(self, x, *args, **kwargs):
            B, L = x.shape
            ps = JNU_SIG_LEN // 250
            n_tok = L // ps
            x = x[:, :n_tok * ps].reshape(B, n_tok, ps)
            t = self.patch_embed(x)
            cls = self.cls_token.expand(B, -1, -1)
            t = torch.cat([cls, t], dim=1) + self.pos_enc[:, :n_tok+1]
            t = self.encoder(t)
            t = self.norm(t[:, 0])
            return self.head(t), None, t

    return BaseTransformer().to(device)


def build_abl1(n_classes, seed):
    """ABL-1: MST only (no PIFFG injection, no SCR)."""
    from src.models.ms_tokenizer import MultiScaleTokenizer
    import torch.nn as nn
    set_all_seeds(seed)
    hidden_dim = 96

    class MSTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = MultiScaleTokenizer(hidden_dim=hidden_dim, n_tokens=256)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.pos_enc = nn.Parameter(torch.randn(1, 257, hidden_dim) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=384,
                dropout=0.1, batch_first=True, activation='gelu', norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, n_classes)

        def forward(self, x, *args, **kwargs):
            B = x.shape[0]
            t = self.tokenizer(x)
            cls = self.cls_token.expand(B, -1, -1)
            t = torch.cat([cls, t], dim=1) + self.pos_enc
            t = self.encoder(t)
            t = self.norm(t[:, 0])
            return self.head(t), None, t

    return MSTTransformer().to(device)


def build_abl2(n_classes, seed, bearing_params_example=None):
    """ABL-2: MST + PIFFG injection (no SCR, no triplet)."""
    from src.models.wapigt import WaPIGT
    from src.models.scr import SpectrumConsistencyRegularizer
    from src.training.loss import WaPIGTLoss
    set_all_seeds(seed)
    model = WaPIGT(
        n_classes=n_classes,
        hidden_dim=config.model.hidden_dim,
        n_encoder_layers=config.model.n_encoder_layers,
        n_heads=config.model.n_heads,
        mlp_dim=config.model.mlp_dim,
        dropout=config.model.dropout,
        n_gat_heads=config.model.n_gat_heads,
        gat_dropout=config.model.gat_dropout,
    ).to(device)
    # Loss: no SCR, no triplet
    loss_fn = WaPIGTLoss(
        n_classes=n_classes, scr_module=None,
        scr_lambda=0.0, scr_warmup_epochs=999,
        n_epochs=120, triplet_lambda=0.0,
        triplet_margin=0.5, triplet_warmup_epochs=999,
    )
    return model, loss_fn


def build_abl3(n_classes, seed):
    """ABL-3: MST + PIFFG + SCR (no triplet)."""
    from src.models.wapigt import WaPIGT
    from src.models.scr import SpectrumConsistencyRegularizer
    from src.training.loss import WaPIGTLoss
    set_all_seeds(seed)
    model = WaPIGT(
        n_classes=n_classes,
        hidden_dim=config.model.hidden_dim,
        n_encoder_layers=config.model.n_encoder_layers,
        n_heads=config.model.n_heads,
        mlp_dim=config.model.mlp_dim,
        dropout=config.model.dropout,
        n_gat_heads=config.model.n_gat_heads,
        gat_dropout=config.model.gat_dropout,
    ).to(device)
    scr = SpectrumConsistencyRegularizer(sigma=2.0)
    loss_fn = WaPIGTLoss(
        n_classes=n_classes, scr_module=scr,
        scr_lambda=config.model.scr_lambda,
        scr_warmup_epochs=config.model.scr_warmup_epochs,
        n_epochs=120, triplet_lambda=0.0,
        triplet_margin=0.5, triplet_warmup_epochs=999,
    )
    return model, loss_fn


def train_simple(model, train_l, val_l, test_l, seed, n_epochs=120, patience=20,
                 is_wapigt=False, loss_fn=None):
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_val, best_state, no_imp = 0.0, None, 0
    for epoch in range(n_epochs):
        if loss_fn is not None and hasattr(loss_fn, 'set_epoch'):
            loss_fn.set_epoch(epoch)
        model.train()
        for batch in train_l:
            sigs = batch["signal"].to(device)
            labs = batch["label"].to(device)
            opt.zero_grad()
            if is_wapigt:
                bp = reorg_bp(batch["bearing_params"], sigs.shape[0])
                logits, attn, embeddings = model(sigs, bp, fs_sampling=JNU_FS)
                if loss_fn is not None:
                    ffb = batch.get("fault_freq_bins")
                    loss = loss_fn(logits, labs, attn,
                                   ffb.to(device) if ffb is not None else None,
                                   sigs.shape[-1], JNU_FS, embeddings=embeddings)
                else:
                    loss = F.cross_entropy(logits, labs)
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
        v = evaluate(model, val_l, is_wapigt=is_wapigt)
        if v > best_val:
            best_val = v
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_val, evaluate(model, test_l, is_wapigt=is_wapigt)


def main():
    logger.info(f"Device: {device}")
    train_l, val_l, test_l = get_loaders()
    n_classes = 4

    results = {"start_time": datetime.now().isoformat(), "configs": {}}

    configs = {
        "ABL-0_BaseTransformer": {"is_wapigt": False},
        "ABL-1_+MST": {"is_wapigt": False},
        "ABL-2_+MST+PIFFG": {"is_wapigt": True},
        "ABL-3_+MST+PIFFG+SCR": {"is_wapigt": True},
        "ABL-4_Full_WaPIGT": {"is_wapigt": True},
    }

    for cfg_name in configs:
        seed_accs = []
        for seed in SEEDS:
            logger.info(f"  {cfg_name} seed={seed}")
            try:
                if cfg_name == "ABL-0_BaseTransformer":
                    model = build_abl0(n_classes, seed)
                    v, t = train_simple(model, train_l, val_l, test_l, seed)
                elif cfg_name == "ABL-1_+MST":
                    model = build_abl1(n_classes, seed)
                    v, t = train_simple(model, train_l, val_l, test_l, seed)
                elif cfg_name == "ABL-2_+MST+PIFFG":
                    model, loss_fn = build_abl2(n_classes, seed)
                    v, t = train_simple(model, train_l, val_l, test_l, seed,
                                        is_wapigt=True, loss_fn=loss_fn)
                elif cfg_name == "ABL-3_+MST+PIFFG+SCR":
                    model, loss_fn = build_abl3(n_classes, seed)
                    v, t = train_simple(model, train_l, val_l, test_l, seed,
                                        is_wapigt=True, loss_fn=loss_fn)
                else:  # ABL-4 Full WaPIGT
                    from src.models.wapigt import WaPIGT
                    from src.models.scr import SpectrumConsistencyRegularizer
                    from src.training.loss import WaPIGTLoss
                    set_all_seeds(seed)
                    model = WaPIGT(
                        n_classes=n_classes,
                        hidden_dim=config.model.hidden_dim,
                        n_encoder_layers=config.model.n_encoder_layers,
                        n_heads=config.model.n_heads,
                        mlp_dim=config.model.mlp_dim,
                        dropout=config.model.dropout,
                        n_gat_heads=config.model.n_gat_heads,
                        gat_dropout=config.model.gat_dropout,
                    ).to(device)
                    scr = SpectrumConsistencyRegularizer(sigma=2.0)
                    loss_fn = WaPIGTLoss(
                        n_classes=n_classes, scr_module=scr,
                        scr_lambda=config.model.scr_lambda,
                        scr_warmup_epochs=config.model.scr_warmup_epochs,
                        n_epochs=120,
                        triplet_lambda=getattr(config.model, 'triplet_lambda', 0.1),
                        triplet_margin=getattr(config.model, 'triplet_margin', 0.5),
                        triplet_warmup_epochs=getattr(config.model, 'triplet_warmup_epochs', 20),
                    )
                    v, t = train_simple(model, train_l, val_l, test_l, seed,
                                        is_wapigt=True, loss_fn=loss_fn)
                logger.info(f"    val={v:.4f} test={t:.4f}")
                seed_accs.append({"seed": seed, "val": v, "test": t})
            except Exception as e:
                logger.error(f"    FAILED: {e}", exc_info=True)
                seed_accs.append({"seed": seed, "error": str(e)})

        ok = [s["test"] for s in seed_accs if "test" in s]
        results["configs"][cfg_name] = {
            "seeds": seed_accs,
            "mean_test": float(np.mean(ok)) if ok else None,
            "std_test": float(np.std(ok)) if ok else None,
        }
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  {cfg_name}: mean={np.mean(ok)*100:.1f}% std={np.std(ok)*100:.1f}%")

    results["end_time"] = datetime.now().isoformat()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation done -> {OUTPUT}")

    # Print ablation table
    logger.info("\n=== ABLATION RESULTS ===")
    for cfg, v in results["configs"].items():
        if v["mean_test"] is not None:
            logger.info(f"  {cfg:40s}: {v['mean_test']*100:.1f}% +/- {v['std_test']*100:.1f}%")


if __name__ == "__main__":
    main()
