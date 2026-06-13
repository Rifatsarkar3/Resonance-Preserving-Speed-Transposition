# Resonance-Preserving Speed Transposition (RPST)

Code and reproducibility scripts for the paper:

> **Resonance-Preserving Speed Transposition: A Physically Grounded, Model-Agnostic
> Augmentation for Cross-Speed Bearing Fault Diagnosis**

*This repository is anonymized for double-blind peer review. It contains no
author-identifying information.*

---

## Overview

Deep models for rolling-element bearing fault diagnosis degrade when the shaft speed at
test time differs from training. A genuine change of shaft speed scales the
**fault-impulse rate** and its modulation sidebands, but leaves the **structural
resonance bands** fixed in absolute frequency (they are set by the housing and
transmission path, not by rotation). Signal resampling and angular-domain order tracking
both violate this invariant: they rescale *all* spectral content and displace the
resonance carrier.

**RPST** is a training-time augmentation that transposes a vibration record to other
shaft speeds using phase-vocoder time-scale modification — changing the impulse rate
while holding the resonance bands fixed, and scaling the kinematic metadata to the
simulated speed. It is model-agnostic and requires only the nominal operating speeds.

This repository also provides:

- a **controlled mechanism study** (no augmentation vs. order-tracking vs. resampling
  vs. RPST) isolating resonance preservation as the causal property of the gain;
- a **speed-extrapolation** study showing RPST generalizes to operating speeds never
  synthesized, where resampling at identical targets degrades below baseline;
- **RPCL** (Resonance-Preserving Consistency Learning), a training objective that
  promotes the same invariance at the representation level;
- **WaPIGT**, an interpretable physics-informed graph Transformer used as one of the
  evaluated models and for the interpretability analysis.

## Repository structure

```
src/            Library code
  models/       WaPIGT (multi-scale tokenizer, physics-informed fault-frequency graph,
                spectrum-consistency regularizer) and components
  baselines/    WDCNN, TICNN, MSCNN, PhysFormer, ViT-1D and registry
  data_loaders/ JNU / PU / CWRU dataset loaders and split strategies
  training/     trainer and loss
  evaluation/   metrics and statistical tests
  utils/        config, reproducibility, fault-frequency formulas
scripts/        Reproducibility scripts (see below)
results/        Result JSONs backing the paper tables
figures/        Paper figures (vector PDF + 300+ DPI PNG)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```
Python ≥ 3.9, PyTorch ≥ 2.0, PyTorch-Geometric ≥ 2.3.

## Datasets

The benchmarks are public and are **not** redistributed here (size). Download and place
them under `data/`, then build splits:

- **JNU** (Jiangnan University): 50 kHz, 600/800/1000 rpm. `scripts/download_jnu.py`
- **PU** (Paderborn University): 64 kHz, 900/1500 rpm. `scripts/extract_pu_rars.py`
- **CWRU** (Case Western Reserve University): 12 kHz, cross-load.

```bash
python scripts/organize_datasets.py     # arrange raw files
python scripts/preprocess_all.py        # windowing / normalization
python scripts/generate_splits.py       # bearing-instance-level, leakage-free splits
```

## Reproducing the results

| Result | Script |
|---|---|
| JNU cross-speed main tables (6 models × 3 tasks × 2 regimes, 5 seeds) | `scripts/g2_definitive_runs.py` |
| Mechanism study (order tracking / resampling / RPST) | `scripts/test_onr.py`, `scripts/test_speed_aug.py`, `scripts/test_tsm_aug.py` |
| Leakage control (transpose to intermediate speed only) | `scripts/g4_intermediate_rpst.py` |
| Baseline learning-rate fairness sweep | `scripts/g5_baseline_lr_sweep.py` |
| Paderborn cross-speed generalization | `scripts/g1_pu_cross_speed.py` |
| WaPIGT triplet-free variant (5 seeds) | `scripts/g9_wapigt_notriplet_5seed.py` |
| RPCL consistency learning | `scripts/h1_rpcl_pu.py`, `scripts/h3_rpcl_anchored.py` |
| Speed extrapolation to unseen speeds | `scripts/h2_extrapolation.py` |
| Architecture ablation | `scripts/run_ablation.py` |
| CWRU cross-load | `scripts/run_cwru_comparison.py` |
| Figures | `scripts/make_g2_figures.py`, `scripts/make_h2_figure.py`, `scripts/make_paper_figures.py` |

Train a single model:
```bash
python scripts/train_wapigt.py          # WaPIGT
python scripts/run_baselines.py         # baseline architectures
```

Reported numbers (mean ± std over seeds, with macro-F1 / AUROC / Cohen's κ and paired
Wilcoxon significance tests) are stored in `results/` for cross-checking.

## License

Released under the MIT License (see [LICENSE](LICENSE)).
