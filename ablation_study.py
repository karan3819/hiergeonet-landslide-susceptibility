"""
Component ablation study for HierGeoNet.

This script is extracted from the original monolithic `run_hiergeonet.py`
pipeline (the ablation logic previously lived as nested closures inside that
script's `main()` — see "Provenance" below). It is a standalone, runnable
script so the ablation study can be re-run independently of the main
training script, and so it is not silently lost or left inconsistent as the
main pipeline evolves (see `train_hiergeonet_multiseed.py`, which superseded
`run_hiergeonet.py` for the primary benchmark/results but does not contain
any ablation code).

WHAT THIS SCRIPT DOES
----------------------
Trains and evaluates 6 model variants on the same train/validation split:
  1. HierGeoNet (full)         - all components enabled
  2. w/o cross-scale bridges   - fine/medium/coarse features are fused with
                                  zeros instead of the learned bridge modules
  3. w/o coarse Transformer    - the coarse graph's GAT output is used
                                  directly, skipping the geo-positional
                                  encoding + Transformer encoder stage
  4-6. w/o medium graph        - the medium-resolution graph branch is
                                  disabled entirely, run at 3 different
                                  random seeds (42, 43, 44) specifically to
                                  check whether the effect of removing this
                                  component is stable or seed-dependent
                                  (see NOTES below - it is NOT fully stable)

All 6 configurations are evaluated on the VALIDATION split (245 patches) at
a FIXED 0.5 decision threshold. This is deliberately different from the
F1-optimal-threshold, held-out-test-set evaluation used for the headline
HierGeoNet result elsewhere in this repository — an ablation study needs a
single, configuration-independent operating point for the comparison to be
meaningful, so threshold selection is turned off here.

PROVENANCE
----------
Extracted, with no logic changes, from the PREVIOUS version of this
repository's main script (archived as `legacy/run_hiergeonet.py`), STAGE 4
block (`_train_ablation_variant`, `_eval_ablation`, and the driving
`ablation_configs` loop, which were originally nested functions/closures
inside that script's `main()`). The current main script, `train.py`
(formerly `run_hier_latest.py`), does not contain any ablation code or the
`HierGeoNetAblation` class — it is a from-scratch rewrite focused on
multi-seed benchmarking of the FULL model only. The only changes made during
extraction from the legacy script:
  - Closures were converted into top-level functions with their captured
    variables (train_loader, val_loader, graph topology tensors, criterion,
    the LR schedule function, etc.) passed as explicit parameters, so the
    functions are independently callable and testable.
  - Graph topology construction (fine/medium/coarse coordinate grids, edge
    building, cross-scale index maps) and dataset/dataloader construction,
    which the original closures relied on implicitly via `main()`'s scope,
    are reproduced here in `build_graph_topology()` and `build_dataloaders()`.
  - `HierGeoNetAblation` did not exist in the legacy script as a top-level
    importable name either — it was already a top-level class there (unlike
    the training closures), so no conversion was needed for it specifically.
    It is defined in this file rather than in `train.py` because it does
    not exist in the current main script and this task's brief was to keep
    `train.py` as given, matching `run_hier_latest.py` exactly. It
    SUBCLASSES the `HierGeoNet` imported from `train.py` (the base
    architecture is identical between the legacy and current scripts), so
    this file has a real, intentional dependency on `train.py` for the
    base model, dataset class, and config, but not for anything
    ablation-specific.
  - No hyperparameter, architecture, loss, threshold, or seed value was
    changed. `results/metrics/ablation_study.csv` in this repository is the
    unmodified output of this exact logic as it existed in the legacy
    script, and has NOT been re-run against the current `train.py`
    codepath. Since the base `HierGeoNet` architecture is identical between
    the two scripts, this is not expected to matter — but if you modify
    `train.py`'s `HierGeoNet` class, re-run this script before trusting
    the CSV against your changes.

NOTES ON READING THE RESULTS
------------------------------
- The ablation training objective is plain weighted BCE only (no Dice loss,
  no geographic contrastive term), unlike the main HierGeoNet training run
  elsewhere in this repository, which combines all three. This is a
  deliberate difference in the original script, not an oversight introduced
  during extraction, and means ablation numbers should not be directly
  compared to the main model's headline test-set numbers.
- The "w/o medium graph" F1 delta ranges from -0.0101 (seed 42) to -0.0413
  (seed 43) across the 3 seeds tested — a 4x spread. Any claim about the
  *magnitude* of the medium-graph component's contribution should cite this
  full range, not a single seed. See results/metrics/README.md for the full
  discussion.

USAGE
-----
    python ablation_study.py

Expects the same `images/` and `annotations/` directory layout documented in
the repository root README. Writes `outputs/ablation_study.csv`.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from train import CFG, DEVICE, AMP_DEVICE_TYPE, USE_AMP, USE_MULTI_GPU
from train import L4SDataset, HierGeoNet, build_edges, gpu_scatter_mean

SEED = 42  # main seed; medium-graph configs additionally use 43 and 44 (see below)


class HierGeoNetAblation(HierGeoNet):
    """Subclasses the current HierGeoNet (from train.py) and adds three
    boolean switches so individual architectural components can be disabled
    at construction time, without duplicating the base model's layers.

    use_bridges=False:     fine/medium/coarse features are fused with zeros
                            in place of the learned CrossScaleBridge outputs.
    use_transformer=False: the coarse graph's GAT output is used directly,
                            skipping the geo-positional encoding + Transformer
                            encoder stage.
    use_medium=False:      the medium-resolution graph branch (proj_med +
                            gat_med + bridge_c2m/bridge_m2f) is skipped
                            entirely; the fine branch bridges directly from
                            the coarse branch instead.

    This is an unmodified copy of the HierGeoNetAblation class as it existed
    in the legacy run_hiergeonet.py script.
    """

    def __init__(self, cfg, use_bridges=True, use_transformer=True, use_medium=True):
        super().__init__(cfg)
        self.use_bridges = use_bridges
        self.use_transformer = use_transformer
        self.use_medium = use_medium

    def forward(self, x_f, ei_f, x_m, ei_m, x_c, ei_c, coords_c, f2m, m2c, f2c, return_embeddings=False):
        h_f = self.gat_fine(self.proj_fine(x_f), ei_f)
        h_m = self.gat_med(self.proj_med(x_m), ei_m) if self.use_medium else None
        h_c = self.gat_coarse(self.proj_coarse(x_c), ei_c)

        if self.use_transformer:
            pe = self.geo_pe(coords_c)
            h_cg = self.transformer((h_c + pe).unsqueeze(0)).squeeze(0)
        else:
            h_cg = h_c

        if self.use_bridges:
            if self.use_medium and h_m is not None:
                h_m2 = self.bridge_c2m(h_m, h_cg[m2c])
                h_f2 = self.bridge_m2f(h_f, h_m2[f2m])
            else:
                h_f2 = h_f
            h_fs = self.bridge_c2f(h_f, h_cg[f2c])
            h_final = self.fusion(torch.cat([h_f2, h_fs], dim=-1))
        else:
            zeros = torch.zeros_like(h_f)
            h_final = self.fusion(torch.cat([h_f, zeros], dim=-1))

        logits = self.head(h_final).squeeze(-1)
        if return_embeddings:
            return logits, h_final
        return logits

ROOT = Path(".")
for _d in ["models", "outputs"]:
    (ROOT / _d).mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "images"
MASK_DIR = PROJECT_ROOT / "annotations"


def build_graph_topology():
    """Reconstructs the fine/medium/coarse graph topology exactly as in the
    main pipeline (128x128 patches; medium = every 3rd pixel; coarse = every
    9th pixel; proximity-radius edges at 1.5 / 4.5 / 13.5 px respectively)."""
    H, W = 128, 128
    y_f, x_f = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords_fine = np.column_stack([y_f.ravel(), x_f.ravel()]).astype(np.float32)
    y_m, x_m = np.meshgrid(np.arange(0, H, 3), np.arange(0, W, 3), indexing="ij")
    coords_med = np.column_stack([y_m.ravel(), x_m.ravel()]).astype(np.float32)
    y_c, x_c = np.meshgrid(np.arange(0, H, 9), np.arange(0, W, 9), indexing="ij")
    coords_coarse = np.column_stack([y_c.ravel(), x_c.ravel()]).astype(np.float32)

    N_med, N_coarse = len(coords_med), len(coords_coarse)

    ei_f, _ = build_edges(coords_fine, prox_dist=1.5)
    ei_m, _ = build_edges(coords_med, prox_dist=4.5)
    ei_c, _ = build_edges(coords_coarse, prox_dist=13.5)

    _, fine2med = cKDTree(coords_med).query(coords_fine, k=1)
    _, med2coarse = cKDTree(coords_coarse).query(coords_med, k=1)
    _, fine2coarse = cKDTree(coords_coarse).query(coords_fine, k=1)

    def _t(arr, dtype):
        return torch.tensor(arr, dtype=dtype).to(DEVICE)

    topo = dict(
        ei_f=ei_f.to(DEVICE), ei_m=ei_m.to(DEVICE), ei_c=ei_c.to(DEVICE),
        f2m_t=_t(fine2med, torch.long), m2c_t=_t(med2coarse, torch.long), f2c_t=_t(fine2coarse, torch.long),
        coords_c_t=_t(coords_coarse, torch.float32),
        N_med=N_med, N_coarse=N_coarse,
    )
    return topo


def build_dataloaders():
    """Reconstructs train/validation dataloaders with the same normalization
    procedure (mean/std over a 200-patch training sample) as the main
    pipeline. The ablation study only needs train+validation; it does not
    touch the held-out test split."""
    set_seed(SEED)
    _raw_ds = L4SDataset(PROJECT_ROOT, "train", norm_stats=None)
    _sample_idx = np.random.choice(len(_raw_ds), size=min(200, len(_raw_ds)), replace=False)
    _imgs = torch.cat([_raw_ds[i][0] for i in _sample_idx], dim=0)
    norm_mean, norm_std = _imgs.mean(0), _imgs.std(0).clamp(min=1e-8)
    del _imgs, _raw_ds

    train_dataset = L4SDataset(PROJECT_ROOT, "train", norm_stats=(norm_mean, norm_std))
    val_dataset = L4SDataset(PROJECT_ROOT, "validation", norm_stats=(norm_mean, norm_std))

    loader_kwargs = dict(
        batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
        persistent_workers=(CFG.NUM_WORKERS > 0), prefetch_factor=(2 if CFG.NUM_WORKERS > 0 else None),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lr_lambda():
    warmup, total = CFG.WARMUP_EPOCHS, CFG.EPOCHS
    min_r = 1e-6 / CFG.LR

    def _lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        prog = (epoch - warmup) / max(1, total - warmup)
        return max(min_r, 0.5 * (1.0 + np.cos(np.pi * prog)))

    return _lr_lambda


def train_ablation_variant(name, seed_val, topo, train_loader, val_loader, criterion, lr_lambda, **kwargs):
    """Trains one HierGeoNetAblation configuration to convergence (early
    stopping on validation AUC, same PATIENCE/EPOCHS as the main model).
    Resumable via a 5-minute checkpoint, exactly as in the original script."""
    set_seed(seed_val)

    safe_name = name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
    best_ckpt = MODEL_DIR / f"ablation_{safe_name}_best.pt"
    resume_ckpt = MODEL_DIR / f"ablation_{safe_name}_resume.pt"

    m = HierGeoNetAblation(CFG, **kwargs).to(DEVICE)
    if USE_MULTI_GPU:
        m = nn.DataParallel(m)

    if best_ckpt.exists():
        print(f"[*] {name} ablation already trained. Loading best checkpoint...")
        m.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        return m

    print(f"Training ablation variant: {name} (seed={seed_val})...")
    opt = AdamW(m.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)

    start_epoch = 1
    best_val_auc, best_state, epochs_since_best = -1.0, None, 0
    last_backup_time = time.time()

    if resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        sched.load_state_dict(ckpt["scheduler_state"])
        best_val_auc = ckpt["best_val_auc"]
        epochs_since_best = ckpt["epochs_since_best"]
        best_state = ckpt["best_state"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[*] Resumed {name} from epoch {start_epoch - 1}. Best AUC so far: {best_val_auc:.4f}")

    ei_f, ei_m, ei_c = topo["ei_f"], topo["ei_m"], topo["ei_c"]
    f2m_t, m2c_t, f2c_t = topo["f2m_t"], topo["m2c_t"], topo["f2c_t"]
    coords_c_t = topo["coords_c_t"]
    N_med, N_coarse = topo["N_med"], topo["N_coarse"]

    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        m.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
            y_batch = y_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
            opt.zero_grad(set_to_none=True)
            all_logits, all_y = [], []
            with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                for b in range(x_batch.shape[0]):
                    xf = x_batch[b]
                    xm = gpu_scatter_mean(xf, f2m_t, N_med)
                    xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                    all_logits.append(m(xf, ei_f, xm, ei_m, xc, ei_c, coords_c_t, f2m_t, m2c_t, f2c_t))
                    all_y.append(y_batch[b])
                loss = criterion(torch.cat(all_logits), torch.cat(all_y))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        sched.step()

        m.eval()
        v_probs, v_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
                for b in range(x_batch.shape[0]):
                    xf = x_batch[b]
                    xm = gpu_scatter_mean(xf, f2m_t, N_med)
                    xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                    with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                        lg = m(xf, ei_f, xm, ei_m, xc, ei_c, coords_c_t, f2m_t, m2c_t, f2c_t)
                    v_probs.extend(torch.sigmoid(lg).float().cpu().tolist())
                v_true.extend(y_batch[b].tolist())
        v_probs, v_true = np.array(v_probs), np.array(v_true, dtype=int)
        cur_auc = roc_auc_score(v_true, v_probs) if len(np.unique(v_true)) > 1 else 0.0

        if cur_auc > best_val_auc:
            best_val_auc = cur_auc
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= CFG.PATIENCE:
                break

        if (time.time() - last_backup_time) >= 300:
            last_backup_time = time.time()
            torch.save({
                "epoch": epoch, "model_state": m.state_dict(), "optimizer_state": opt.state_dict(),
                "scheduler_state": sched.state_dict(), "best_val_auc": best_val_auc,
                "epochs_since_best": epochs_since_best, "best_state": best_state,
            }, resume_ckpt)
            print(f"    [Backup] state saved for {name} (epoch={epoch})")

    if best_state is not None:
        torch.save(best_state, best_ckpt)
        m.load_state_dict(best_state)
    if resume_ckpt.exists():
        resume_ckpt.unlink()

    return m


def eval_ablation(m, topo, val_loader):
    """Evaluates a trained ablation variant on the validation set at a FIXED
    0.5 threshold (intentionally not F1-optimal — see module docstring)."""
    ei_f, ei_m, ei_c = topo["ei_f"], topo["ei_m"], topo["ei_c"]
    f2m_t, m2c_t, f2c_t = topo["f2m_t"], topo["m2c_t"], topo["f2c_t"]
    coords_c_t = topo["coords_c_t"]
    N_med, N_coarse = topo["N_med"], topo["N_coarse"]

    m.eval()
    ps, ls = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
            for b in range(x_batch.shape[0]):
                xf = x_batch[b]
                xm = gpu_scatter_mean(xf, f2m_t, N_med)
                xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                    lg = m(xf, ei_f, xm, ei_m, xc, ei_c, coords_c_t, f2m_t, m2c_t, f2c_t)
                ps.extend(torch.sigmoid(lg).float().cpu().tolist())
                ls.extend(y_batch[b].tolist())
    ps, ls = np.array(ps), np.array(ls, dtype=int)
    preds = (ps > 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(ls, ps)) if len(np.unique(ls)) > 1 else float("nan"),
        "F1": float(f1_score(ls, preds, average="binary", zero_division=0)),
        "Prec": float(precision_score(ls, preds, zero_division=0)),
        "Rec": float(recall_score(ls, preds, zero_division=0)),
    }


ABLATION_CONFIGS = [
    ("HierGeoNet (full)",          dict(use_bridges=True,  use_transformer=True,  use_medium=True,  seed_val=SEED)),
    ("w/o cross-scale bridges",    dict(use_bridges=False, use_transformer=True,  use_medium=True,  seed_val=SEED)),
    ("w/o coarse Transformer",     dict(use_bridges=True,  use_transformer=False, use_medium=True,  seed_val=SEED)),
    ("w/o medium graph (Seed 42)", dict(use_bridges=True,  use_transformer=True,  use_medium=False, seed_val=42)),
    ("w/o medium graph (Seed 43)", dict(use_bridges=True,  use_transformer=True,  use_medium=False, seed_val=43)),
    ("w/o medium graph (Seed 44)", dict(use_bridges=True,  use_transformer=True,  use_medium=False, seed_val=44)),
]


def main():
    print("Building graph topology...")
    topo = build_graph_topology()
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders()

    pos_wt = torch.tensor([CFG.POS_WEIGHT], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
    lr_lambda = make_lr_lambda()

    ablation_rows = []
    for name, kw in ABLATION_CONFIGS:
        kw = dict(kw)
        s_val = kw.pop("seed_val")
        trained_model = train_ablation_variant(name, s_val, topo, train_loader, val_loader, criterion, lr_lambda, **kw)
        res = eval_ablation(trained_model, topo, val_loader)
        ablation_rows.append({"Configuration": name, **res})
        print(f"    {name}: final AUC={res['AUC']:.4f}  F1={res['F1']:.4f}")

    # Restore main seed for reproducibility of anything run afterward
    set_seed(SEED)

    ablation_df = pd.DataFrame(ablation_rows).set_index("Configuration")
    full_row = ablation_df.loc["HierGeoNet (full)"]
    ablation_df["\u0394AUC"] = ablation_df["AUC"] - full_row["AUC"]
    ablation_df["\u0394F1"] = ablation_df["F1"] - full_row["F1"]
    ablation_df.to_csv(OUT_DIR / "ablation_study.csv")
    print("\nSaved outputs/ablation_study.csv")
    print(ablation_df)


if __name__ == "__main__":
    main()
