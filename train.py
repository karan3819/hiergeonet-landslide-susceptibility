# %%
# ============================================================
# IMPORTS & GLOBAL DEVICE SETUP
# ============================================================
import os, sys, warnings, time, json, math, copy, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
import h5py
import joblib

warnings.filterwarnings('ignore')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve, confusion_matrix,
    average_precision_score, precision_recall_curve,
    jaccard_score, accuracy_score
)

# Global Device Config (Silent at module level to prevent worker spam)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    AMP_DEVICE_TYPE = 'cuda'
    USE_MULTI_GPU = (torch.cuda.device_count() > 1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
else:
    DEVICE = torch.device('cpu')
    AMP_DEVICE_TYPE = 'cpu'
    USE_MULTI_GPU = False

USE_AMP = torch.cuda.is_available()
from torch.amp import GradScaler, autocast

ROOT = Path('.')
for d in ['data', 'models', 'outputs']:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / 'models'
OUT_DIR   = ROOT / 'outputs'

# %%
# ============================================================
# CONFIGURATION & GLOBAL PATHS
# ============================================================
class Config:
    IN_DIM   = 14
    D_FINE   = 128
    D_MED    = 192
    D_COARSE = 256
    GAT_HEADS  = 4
    GAT_LAYERS = 2
    TF_HEADS   = 8
    TF_LAYERS  = 3
    FF_DIM     = 512
    DROPOUT    = 0.15

    LAMBDA_CONTRAST     = 0.1
    CONTRAST_MARGIN     = 2.0
    CONTRAST_RADIUS_PIX = 5.0

    EPOCHS        = 100
    BATCH_SIZE    = 32
    LR            = 5e-4
    WEIGHT_DECAY  = 1e-4
    POS_WEIGHT    = 3.5
    GRAD_CLIP     = 1.0
    PATIENCE      = 20
    WARMUP_EPOCHS = 15

    NUM_WORKERS  = 8
    PIN_MEMORY   = True

    SEEDS = [42, 43, 44, 45, 46, 47]
    CHECKPOINT_INTERVAL_SEC = 180   # 3 minutes

CFG = Config()
if not torch.cuda.is_available():
    CFG.PIN_MEMORY = False

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "images"
MASK_DIR = PROJECT_ROOT / "annotations"
TRAIN_IMG_DIR = IMAGE_DIR / "train"
VALID_IMG_DIR = IMAGE_DIR / "validation"
TEST_IMG_DIR = IMAGE_DIR / "test"

scaler = GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# %%
# ============================================================
# GLOBAL CLASSES & HELPER FUNCTIONS
# ============================================================
def build_edges(coords, prox_dist):
    tree  = cKDTree(coords)
    pairs = tree.query_pairs(r=prox_dist, output_type='ndarray')
    if len(pairs) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
    ii, jj  = pairs[:, 0], pairs[:, 1]
    dists   = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    sigma2  = (prox_dist * 0.5) ** 2
    weights = np.exp(-dists**2 / (2 * sigma2))
    src = np.concatenate([ii, jj])
    dst = np.concatenate([jj, ii])
    wts = np.concatenate([weights, weights])
    return (torch.tensor(np.stack([src, dst]), dtype=torch.long),
            torch.tensor(wts, dtype=torch.float32))

def gpu_scatter_mean(src, index, dim_size):
    D   = src.shape[1]
    idx = index.unsqueeze(1).expand(-1, D)
    s   = torch.zeros(dim_size, D, dtype=src.dtype, device=src.device)
    c   = torch.zeros(dim_size, D, dtype=src.dtype, device=src.device)
    s.scatter_add_(0, idx, src)
    c.scatter_add_(0, idx, torch.ones_like(src))
    return s / c.clamp(min=1)

def contrastive_geo_loss(embeddings, coords, y, radius=CFG.CONTRAST_RADIUS_PIX, margin=CFG.CONTRAST_MARGIN, n_pairs=2048):
    N   = embeddings.size(0)
    idx = torch.randperm(N, device=embeddings.device)[:min(n_pairs * 2, N)]
    emb, crd, ys = embeddings[idx], coords[idx], y[idx]
    if emb.size(0) < 4:
        return torch.zeros(1, device=embeddings.device).squeeze()
    ed = torch.cdist(emb, emb.detach(), p=2.0)
    gd = torch.cdist(crd, crd, p=2.0)
    same = (ys.unsqueeze(1) == ys.unsqueeze(0))
    pos_mask = same  & (gd < radius);       pos_mask.fill_diagonal_(False)
    neg_mask = (~same) | (gd > 2*radius);   neg_mask.fill_diagonal_(False)
    lp = (ed[pos_mask] ** 2).mean()                    if pos_mask.any() else ed.new_tensor(0.0)
    ln = (F.relu(margin - ed[neg_mask]) ** 2).mean()   if neg_mask.any() else ed.new_tensor(0.0)
    return lp + ln

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    dice = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1.0 - dice

def _unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m

class L4SDataset(Dataset):
    def __init__(self, data_dir, split, norm_stats=None):
        if split.lower() == "train":
            self.img_dir = IMAGE_DIR / "train"
            self.mask_dir = MASK_DIR / "train"
        elif split.lower() == "validation":
            self.img_dir = IMAGE_DIR / "validation"
            self.mask_dir = MASK_DIR / "validation"
        elif split.lower() == "test":
            self.img_dir = IMAGE_DIR / "test"
            self.mask_dir = MASK_DIR / "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.files = sorted(self.img_dir.glob("*.h5"))
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            fp      = self.files[idx]
            mask_fp = self.mask_dir / fp.name.replace('image', 'mask')
            with h5py.File(fp, 'r') as f:
                img  = f['img'][:]
            with h5py.File(mask_fp, 'r') as f:
                mask = f['mask'][:]
        except Exception:
            img = np.zeros((14, 128, 128), dtype=np.float32)
            mask = np.zeros((128, 128), dtype=np.float32)

        if img.ndim == 3 and img.shape[0] == 14:
            img = img.transpose(1, 2, 0)
        img  = img.reshape(-1, 14).astype(np.float32)
        mask = mask.ravel().astype(np.float32)

        img  = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        if self.norm_stats is not None:
            mean, std = self.norm_stats
            img = (img - mean) / std
        else:
            img = (img - img.mean(0)) / (img.std(0) + 1e-8)
        return img, mask

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dropout):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim // heads, heads=heads,
                            dropout=dropout, concat=True, add_self_loops=True)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.GELU()

    def forward(self, x, edge_index):
        return self.norm(self.act(self.conv(x, edge_index)) + self.skip(x))

class GATStack(nn.Module):
    def __init__(self, in_dim, out_dim, heads, n_layers, dropout):
        super().__init__()
        dims = [in_dim] + [out_dim] * n_layers
        self.layers = nn.ModuleList([
            GATLayer(dims[i], dims[i+1], heads, dropout)
            for i in range(n_layers)
        ])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

class CrossScaleBridge(nn.Module):
    def __init__(self, d_fine, d_coarse):
        super().__init__()
        self.proj    = nn.Linear(d_coarse, d_fine)
        self.gate    = nn.Linear(d_fine * 2, d_fine)
        self.context = nn.Linear(d_fine, d_fine)
        self.norm    = nn.LayerNorm(d_fine)

    def forward(self, h_fine, h_coarse_bcast):
        hc   = self.proj(h_coarse_bcast)
        gate = torch.sigmoid(self.gate(torch.cat([h_fine, hc], dim=-1)))
        ctx  = torch.tanh(self.context(hc))
        return self.norm(h_fine + gate * ctx)

class GeoPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        k   = torch.arange(d_model // 4, dtype=torch.float32)
        div = 10000.0 ** (2.0 * k / d_model)
        self.register_buffer('div', div)

    def forward(self, coords):
        lat = coords[:, 0:1]
        lon = coords[:, 1:2]
        d   = self.div.unsqueeze(0)
        pe  = torch.cat([
            torch.sin(lat / d), torch.cos(lat / d),
            torch.sin(lon / d), torch.cos(lon / d),
        ], dim=-1)
        return pe * self.scale.unsqueeze(0)

class HierGeoNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_f, d_m, d_c = cfg.D_FINE, cfg.D_MED, cfg.D_COARSE
        self.proj_fine   = nn.Sequential(nn.Linear(cfg.IN_DIM, d_f), nn.LayerNorm(d_f), nn.GELU())
        self.proj_med    = nn.Sequential(nn.Linear(cfg.IN_DIM, d_m), nn.LayerNorm(d_m), nn.GELU())
        self.proj_coarse = nn.Sequential(nn.Linear(cfg.IN_DIM, d_c), nn.LayerNorm(d_c), nn.GELU())

        self.gat_fine   = GATStack(d_f, d_f, cfg.GAT_HEADS, cfg.GAT_LAYERS, cfg.DROPOUT)
        self.gat_med    = GATStack(d_m, d_m, cfg.GAT_HEADS, cfg.GAT_LAYERS, cfg.DROPOUT)
        self.gat_coarse = GATStack(d_c, d_c, cfg.GAT_HEADS, cfg.GAT_LAYERS, cfg.DROPOUT)

        self.geo_pe = GeoPositionalEncoding(d_c)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_c, nhead=cfg.TF_HEADS, dim_feedforward=cfg.FF_DIM,
            dropout=cfg.DROPOUT, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.TF_LAYERS, enable_nested_tensor=False)

        self.bridge_c2m  = CrossScaleBridge(d_m, d_c)
        self.bridge_m2f  = CrossScaleBridge(d_f, d_m)
        self.bridge_c2f  = CrossScaleBridge(d_f, d_c)

        self.fusion = nn.Sequential(nn.Linear(d_f * 2, d_f), nn.LayerNorm(d_f), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d_f, 64), nn.GELU(), nn.Dropout(cfg.DROPOUT), nn.Linear(64, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_f, ei_f, x_m, ei_m, x_c, ei_c, coords_c, f2m, m2c, f2c, return_embeddings=False):
        h_f = self.gat_fine  (self.proj_fine  (x_f), ei_f)
        h_m = self.gat_med   (self.proj_med   (x_m), ei_m)
        h_c = self.gat_coarse(self.proj_coarse(x_c), ei_c)

        pe         = self.geo_pe(coords_c)
        h_cg       = self.transformer((h_c + pe).unsqueeze(0)).squeeze(0)

        h_m2 = self.bridge_c2m(h_m, h_cg[m2c])
        h_f2 = self.bridge_m2f(h_f, h_m2[f2m])
        h_fs = self.bridge_c2f(h_f, h_cg[f2c])

        h_final = self.fusion(torch.cat([h_f2, h_fs], dim=-1))
        logits  = self.head(h_final).squeeze(-1)

        if return_embeddings:
            return logits, h_final
        return logits

# ============================================================
# CANONICAL BASELINES
# ============================================================
class CNNBaseline(nn.Module):
    MAX_EPOCHS = CFG.EPOCHS
    PATIENCE   = CFG.PATIENCE
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    def make_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    def forward(self, x, *args, **kwargs):
        spatial_dim = int(math.sqrt(x.shape[0]))
        x_img = x.T.view(1, -1, spatial_dim, spatial_dim)
        out = self.net(x_img)
        return out.view(-1)

class GCNBaseline(nn.Module):
    HIDDEN_DIM   = 16
    DROPOUT      = 0.5
    LR           = 0.01
    WEIGHT_DECAY = 5e-4
    MAX_EPOCHS   = 200
    PATIENCE     = 30
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, self.HIDDEN_DIM)
        self.conv2 = GCNConv(self.HIDDEN_DIM, 1)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x, ei, *args, **kwargs):
        h = F.dropout(x, p=self.DROPOUT, training=self.training)
        h = F.relu(self.conv1(h, ei))
        h = F.dropout(h, p=self.DROPOUT, training=self.training)
        return self.conv2(h, ei).squeeze(-1)
    def make_optimizer(self):
        first_layer_params = list(self.conv1.parameters())
        other_params = list(self.conv2.parameters())
        return torch.optim.Adam([
            {'params': first_layer_params, 'weight_decay': self.WEIGHT_DECAY},
            {'params': other_params,       'weight_decay': 0.0},
        ], lr=self.LR)

class GraphSAGEBaseline(nn.Module):
    HIDDEN_DIM = 128
    DROPOUT    = 0.5
    LR         = 0.01
    MAX_EPOCHS = 200
    PATIENCE   = 10
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, self.HIDDEN_DIM, aggr='mean')
        self.conv2 = SAGEConv(self.HIDDEN_DIM, 1, aggr='mean')
    def forward(self, x, ei, *args, **kwargs):
        h = F.relu(self.conv1(x, ei))
        h = F.dropout(h, p=self.DROPOUT, training=self.training)
        return self.conv2(h, ei).squeeze(-1)
    def make_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

class GATBaseline(nn.Module):
    HEADS_L1     = 8
    FEATS_L1     = 8
    DROPOUT      = 0.6
    LR           = 0.005
    WEIGHT_DECAY = 5e-4
    MAX_EPOCHS   = 300
    PATIENCE     = 40
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, self.FEATS_L1, heads=self.HEADS_L1, dropout=self.DROPOUT, concat=True)
        self.conv2 = GATConv(self.FEATS_L1 * self.HEADS_L1, 1, heads=1, dropout=self.DROPOUT, concat=False)
    def forward(self, x, ei, *args, **kwargs):
        h = F.dropout(x, p=self.DROPOUT, training=self.training)
        h = F.elu(self.conv1(h, ei))
        h = F.dropout(h, p=self.DROPOUT, training=self.training)
        return self.conv2(h, ei).squeeze(-1)
    def make_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)

# ============================================================
# PER-SEED TRAINING FUNCTIONS
# ============================================================
def train_hiergeonet_one_seed(seed, train_loader, val_loader, test_loader, ei_f, ei_m, ei_c, coords_c_t, coords_f_t, f2m_t, m2c_t, f2c_t, N_med, N_coarse, criterion):
    set_all_seeds(seed)
    BEST_CKPT   = MODEL_DIR / f'hiergeonet_seed{seed}_best.pt'
    RESUME_CKPT = MODEL_DIR / f'hiergeonet_seed{seed}_resume.pt'
    DONE_MARKER = MODEL_DIR / f'hiergeonet_seed{seed}_test_metrics.json'
    
    if DONE_MARKER.exists():
        print(f"[seed {seed}] HierGeoNet already fully evaluated. Loading saved test metrics.")
        with open(DONE_MARKER) as f: return json.load(f)

    model = HierGeoNet(CFG).to(DEVICE)
    if USE_MULTI_GPU: model = nn.DataParallel(model)
    _net = _unwrap(model)

    optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.999))
    _warmup, _total = CFG.WARMUP_EPOCHS, CFG.EPOCHS
    _min_r = 1e-6 / CFG.LR

    def _lr_lambda(epoch):
        if epoch < _warmup: return float(epoch + 1) / float(_warmup)
        prog = (epoch - _warmup) / max(1, _total - _warmup)
        return max(_min_r, 0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = LambdaLR(optimizer, _lr_lambda)
    local_scaler = GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)

    history  = dict(train_loss=[], val_auc=[], val_f1=[], lr=[])
    best_auc, patience, start_epoch = 0.0, 0, 1

    if RESUME_CKPT.exists():
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        _net.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        local_scaler.load_state_dict(ckpt['scaler_state'])
        best_auc, patience, history = ckpt['best_auc'], ckpt['patience'], ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"[seed {seed}] Resumed from epoch {start_epoch-1}. Best AUC so far: {best_auc:.4f}")

    last_backup_time = time.time()
    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
            y_batch = y_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
            B = x_batch.shape[0]
            optimizer.zero_grad(set_to_none=True)
            all_logits, all_embeds, all_y = [], [], []
            with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                for b in range(B):
                    xf = x_batch[b]
                    xm = gpu_scatter_mean(xf, f2m_t, N_med)
                    xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                    lg, em = _net(xf, ei_f, xm, ei_m, xc, ei_c, coords_c_t, f2m_t, m2c_t, f2c_t, return_embeddings=True)
                    all_logits.append(lg); all_embeds.append(em); all_y.append(y_batch[b])
                logits_cat, embeds_cat, y_cat = torch.cat(all_logits), torch.cat(all_embeds), torch.cat(all_y)
                loss = criterion(logits_cat, y_cat) + (0.5 * dice_loss(logits_cat, y_cat)) + CFG.LAMBDA_CONTRAST * contrastive_geo_loss(embeds_cat, coords_f_t.repeat(B, 1), y_cat)

            local_scaler.scale(loss).backward(); local_scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP); local_scaler.step(optimizer); local_scaler.update()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss, cur_lr = epoch_loss / len(train_loader), scheduler.get_last_lr()[0]

        model.eval()
        v_probs, v_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
                y_batch = y_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
                for b in range(x_batch.shape[0]):
                    xf = x_batch[b]
                    xm = gpu_scatter_mean(xf, f2m_t, N_med)
                    xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                    with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                        lg = _net(xf, ei_f, xm, ei_m, xc, ei_c, coords_c_t, f2m_t, m2c_t, f2c_t)
                    v_probs.extend(torch.sigmoid(lg).float().cpu().tolist())
                    v_true.extend(y_batch[b].float().cpu().tolist())

        v_probs, v_true = np.array(v_probs, dtype=np.float32), np.array(v_true, dtype=np.int32)
        if len(np.unique(v_true)) > 1:
            val_auc = float(roc_auc_score(v_true, v_probs))
            prec_arr, rec_arr, thr_arr = precision_recall_curve(v_true, v_probs)
            denom = prec_arr[:-1] + rec_arr[:-1]
            f1_array = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
            val_f1 = float(f1_array[int(np.argmax(f1_array))])
        else: val_auc, val_f1 = 0.0, 0.0

        history['train_loss'].append(avg_loss); history['val_auc'].append(val_auc); history['val_f1'].append(val_f1); history['lr'].append(cur_lr)

        if val_auc > best_auc:
            best_auc, patience = val_auc, 0
            torch.save({'epoch': epoch, 'val_auc': val_auc, 'val_f1': val_f1, 'model_state': _unwrap(model).state_dict()}, BEST_CKPT)
            print(f'[seed {seed}] epoch {epoch}: new best val AUC {best_auc:.4f} (saved)')
        else:
            patience += 1
            if patience >= CFG.PATIENCE: print(f'[seed {seed}] Early stop at epoch {epoch}'); break

        current_time = time.time()
        if (current_time - last_backup_time) >= CFG.CHECKPOINT_INTERVAL_SEC:
            last_backup_time = current_time
            torch.save({'epoch': epoch, 'model_state': _net.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'scaler_state': local_scaler.state_dict(), 'best_auc': best_auc, 'patience': patience, 'history': history}, RESUME_CKPT)

    try: ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
    except TypeError: ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
    eval_model = HierGeoNet(CFG).to(DEVICE)
    eval_model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state"].items()}, strict=True)
    eval_model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
            for b in range(x_batch.shape[0]):
                xf, xm = x_batch[b], gpu_scatter_mean(x_batch[b], f2m_t, N_med)
                xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                    lg = eval_model(xf, ei_f, xm, ei_m, xc, ei_c, coords_c_t, f2m_t, m2c_t, f2c_t)
                all_probs.extend(torch.sigmoid(lg).float().cpu().tolist()); all_labels.extend(y_batch[b].tolist())

    all_probs, all_labels = np.array(all_probs, dtype=np.float32), np.array(all_labels, dtype=np.int32)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(all_labels, all_probs)
    denom  = prec_arr[:-1] + rec_arr[:-1]
    f1_arr = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1_arr))
    BEST_THRESHOLD = float(thr_arr[best_idx])
    preds_bin = (all_probs >= BEST_THRESHOLD).astype(np.int32)

    metrics = {
        "seed": seed, "AUC": float(roc_auc_score(all_labels, all_probs)),
        "F1": float(f1_score(all_labels, preds_bin, average="binary", zero_division=0)),
        "IoU": float(jaccard_score(all_labels, preds_bin, average='binary', zero_division=0)),
        "Recall": float(recall_score(all_labels, preds_bin, zero_division=0)),
        "Precision": float(precision_score(all_labels, preds_bin, zero_division=0)),
        "MCC": float(matthews_corrcoef(all_labels, preds_bin)),
        "Accuracy": float(accuracy_score(all_labels, preds_bin)),
        "threshold": BEST_THRESHOLD
    }
    with open(DONE_MARKER, 'w') as f: json.dump(metrics, f, indent=2)
    if RESUME_CKPT.exists(): RESUME_CKPT.unlink()
    return metrics

def train_gnn_baseline_one_seed(name, model_cls, seed, train_loader, val_loader, test_loader, ei_f, criterion):
    set_all_seeds(seed)
    safe_name = name.replace("(", "").replace(")", "")
    BEST_CKPT   = MODEL_DIR / f'baseline_{safe_name}_seed{seed}_best.pt'
    RESUME_CKPT = MODEL_DIR / f'baseline_{safe_name}_seed{seed}_resume.pt'
    DONE_MARKER = MODEL_DIR / f'baseline_{safe_name}_seed{seed}_test_metrics.json'

    if DONE_MARKER.exists():
        print(f"[seed {seed}] {name} already fully evaluated. Loading saved test metrics.")
        with open(DONE_MARKER) as f: return json.load(f)

    b_model = model_cls(CFG.IN_DIM).to(DEVICE)
    b_opt = b_model.make_optimizer()
    start_epoch, best_val_auc, best_state, epochs_since_best = 1, -1.0, None, 0
    last_backup_time = time.time()

    if RESUME_CKPT.exists():
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        b_model.load_state_dict(ckpt['model_state'])
        b_opt.load_state_dict(ckpt['optimizer_state'])
        best_val_auc, epochs_since_best, best_state = ckpt['best_val_auc'], ckpt['epochs_since_best'], ckpt['best_state']
        start_epoch = ckpt['epoch'] + 1
        print(f"[seed {seed}] Resumed {name} from epoch {start_epoch-1}.")

    for ep in range(start_epoch, model_cls.MAX_EPOCHS + 1):
        b_model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            b_opt.zero_grad()
            loss = 0
            for b in range(x_batch.shape[0]): loss = loss + criterion(b_model(x_batch[b], ei_f), y_batch[b])
            loss.backward(); b_opt.step()

        b_model.eval()
        v_probs, v_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                for b in range(x_batch.shape[0]):
                    v_probs.extend(torch.sigmoid(b_model(x_batch[b], ei_f)).cpu().tolist()); v_true.extend(y_batch[b].tolist())
        v_probs, v_true = np.array(v_probs), np.array(v_true, dtype=int)
        cur_auc = roc_auc_score(v_true, v_probs) if len(np.unique(v_true)) > 1 else 0.0

        if cur_auc > best_val_auc: best_val_auc, best_state, epochs_since_best = cur_auc, {k: v.detach().clone() for k, v in b_model.state_dict().items()}, 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= model_cls.PATIENCE: print(f"[seed {seed}] {name} early stop at ep {ep}"); break

        if (time.time() - last_backup_time) >= CFG.CHECKPOINT_INTERVAL_SEC:
            last_backup_time = time.time()
            torch.save({'epoch': ep, 'model_state': b_model.state_dict(), 'optimizer_state': b_opt.state_dict(), 'best_val_auc': best_val_auc, 'epochs_since_best': epochs_since_best, 'best_state': best_state}, RESUME_CKPT)

    if best_state is not None:
        torch.save(best_state, BEST_CKPT)
        b_model.load_state_dict(best_state)
    if RESUME_CKPT.exists(): RESUME_CKPT.unlink()

    b_model.eval()
    t_probs, t_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            for b in range(x_batch.shape[0]):
                t_probs.extend(torch.sigmoid(b_model(x_batch[b], ei_f)).cpu().tolist()); t_labels.extend(y_batch[b].tolist())
    t_probs, t_labels = np.array(t_probs), np.array(t_labels, dtype=int)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(t_labels, t_probs)
    denom = prec_arr[:-1] + rec_arr[:-1]
    f1_arr = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1_arr)) if len(f1_arr) > 0 else 0
    thresh = float(thr_arr[best_idx]) if len(thr_arr) > 0 else 0.5
    t_preds = (t_probs >= thresh).astype(int)

    metrics = {
        "seed": seed, "AUC": float(roc_auc_score(t_labels, t_probs)) if len(np.unique(t_labels)) > 1 else 0.0,
        "F1": float(f1_score(t_labels, t_preds, average='binary', zero_division=0)),
        "IoU": float(jaccard_score(t_labels, t_preds, average='binary', zero_division=0)),
        "Recall": float(recall_score(t_labels, t_preds, zero_division=0)),
        "Precision": float(precision_score(t_labels, t_preds, zero_division=0)),
        "MCC": float(matthews_corrcoef(t_labels, t_preds)),
        "Accuracy": float(accuracy_score(t_labels, t_preds)),
        "threshold": float(thresh)
    }
    with open(DONE_MARKER, 'w') as f: json.dump(metrics, f, indent=2)
    return metrics

def fit_flat_baseline_with_resume(clf, name, seed, X_tr, y_tr, X_te, y_te):
    ckpt_path = MODEL_DIR / f'flatbaseline_{name}_seed{seed}.joblib'
    DONE_MARKER = MODEL_DIR / f'baseline_{name}_seed{seed}_test_metrics.json'

    if DONE_MARKER.exists():
        print(f"[seed {seed}] {name} already fully evaluated. Loading saved test metrics.")
        with open(DONE_MARKER) as f: return json.load(f)

    if ckpt_path.exists():
        clf = joblib.load(ckpt_path)
        print(f"[seed {seed}] Loaded saved model for {name}.")
    else:
        print(f"[seed {seed}] Training {name}...")
        clf.fit(X_tr, y_tr)
        joblib.dump(clf, ckpt_path)

    pr = clf.predict_proba(X_te)[:, 1]
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_te, pr)
    denom = prec_arr[:-1] + rec_arr[:-1]
    f1_arr = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1_arr)) if len(f1_arr) > 0 else 0
    thresh = float(thr_arr[best_idx]) if len(thr_arr) > 0 else 0.5
    pd_ = (pr >= thresh).astype(int)

    metrics = {
        'seed': seed,
        'AUC': float(roc_auc_score(y_te, pr)) if len(np.unique(y_te)) > 1 else 0.0,
        'F1': float(f1_score(y_te, pd_, average='binary', zero_division=0)),
        'IoU': float(jaccard_score(y_te, pd_, average='binary', zero_division=0)),
        'Recall': float(recall_score(y_te, pd_, zero_division=0)),
        'Precision': float(precision_score(y_te, pd_, zero_division=0)),
        'MCC': float(matthews_corrcoef(y_te, pd_)),
        'Accuracy': float(accuracy_score(y_te, pd_)),
        'threshold': float(thresh)
    }
    with open(DONE_MARKER, 'w') as f: json.dump(metrics, f, indent=2)
    return metrics

def aggregate_seed_results(name, seed_metrics_list):
    keys = [k for k in seed_metrics_list[0].keys() if k not in ('seed', 'threshold', 'Threshold')]
    row = {'Model': name}
    for k in keys:
        vals = np.array([m[k] for m in seed_metrics_list], dtype=np.float64)
        row[f'{k}_mean'], row[f'{k}_std'] = float(vals.mean()), float(vals.std())
    row['n_seeds'] = len(seed_metrics_list)
    return row

# ============================================================
# MAIN PROCEDURAL EXECUTION (PROTECTED SCOPE)
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', help='HierGeoNet, GCN, GraphSAGE, GAT, CNN, LR, RF, XGBoost, Flat, all')
    parser.add_argument('--seed', type=int, default=None, help='Specific seed to run')
    parser.add_argument('--aggregate_only', action='store_true', help='Only aggregate existing JSON outputs')
    args = parser.parse_args()

    # Device & Setup banner only in main process
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f' GPUs detected: {n_gpus}')
        for i in range(n_gpus):
            nm = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f'   GPU {i}: {nm}  {mem:.1f} GB VRAM')
    else:
        print('  No GPU detected - running on CPU')
    
    print(f' DEVICE={DEVICE} | AMP_DEVICE_TYPE={AMP_DEVICE_TYPE!r} | USE_AMP={USE_AMP} | Multi-GPU={USE_MULTI_GPU}')

    if not args.aggregate_only:
        print(f"Targeting Model: {args.model} | Target Seed: {args.seed}")

    train_img_files = sorted(TRAIN_IMG_DIR.glob("*.h5"))
    val_img_files = sorted(VALID_IMG_DIR.glob("*.h5"))
    test_img_files = sorted(TEST_IMG_DIR.glob("*.h5"))
    if len(train_img_files) == 0: raise FileNotFoundError('No .h5 files found in image directories.')

    H, W = 128, 128
    y_f, x_f = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords_fine   = np.column_stack([y_f.ravel(), x_f.ravel()]).astype(np.float32)
    y_m, x_m = np.meshgrid(np.arange(0, H, 3), np.arange(0, W, 3), indexing='ij')
    coords_med    = np.column_stack([y_m.ravel(), x_m.ravel()]).astype(np.float32)
    y_c, x_c = np.meshgrid(np.arange(0, H, 9), np.arange(0, W, 9), indexing='ij')
    coords_coarse = np.column_stack([y_c.ravel(), x_c.ravel()]).astype(np.float32)

    N_fine, N_med, N_coarse = len(coords_fine), len(coords_med), len(coords_coarse)
    ei_f, ew_f = build_edges(coords_fine, prox_dist=1.5)
    ei_m, ew_m = build_edges(coords_med, prox_dist=4.5)
    ei_c, ew_c = build_edges(coords_coarse, prox_dist=13.5)

    _, fine2med = cKDTree(coords_med).query(coords_fine, k=1)
    _, med2coarse = cKDTree(coords_coarse).query(coords_med, k=1)
    _, fine2coarse = cKDTree(coords_coarse).query(coords_fine, k=1)

    def _t(arr, dtype): return torch.tensor(arr, dtype=dtype).to(DEVICE)
    f2m_t = _t(fine2med, torch.long); m2c_t = _t(med2coarse, torch.long); f2c_t = _t(fine2coarse, torch.long)
    coords_f_t, coords_c_t = _t(coords_fine, torch.float32), _t(coords_coarse, torch.float32)
    ei_f, ew_f = ei_f.to(DEVICE), ew_f.to(DEVICE)
    ei_m, ew_m = ei_m.to(DEVICE), ew_m.to(DEVICE)
    ei_c, ew_c = ei_c.to(DEVICE), ew_c.to(DEVICE)

    set_all_seeds(CFG.SEEDS[0])
    _raw_ds = L4SDataset(PROJECT_ROOT, 'train', norm_stats=None)
    _sample_idx = np.random.choice(len(_raw_ds), size=min(200, len(_raw_ds)), replace=False)
    _imgs = torch.cat([_raw_ds[i][0] for i in _sample_idx], dim=0)
    NORM_MEAN, NORM_STD = _imgs.mean(0), _imgs.std(0).clamp(min=1e-8)
    del _imgs, _raw_ds

    train_dataset = L4SDataset(PROJECT_ROOT, 'train', norm_stats=(NORM_MEAN, NORM_STD))
    val_dataset   = L4SDataset(PROJECT_ROOT, 'validation', norm_stats=(NORM_MEAN, NORM_STD))
    test_dataset  = L4SDataset(PROJECT_ROOT, 'test', norm_stats=(NORM_MEAN, NORM_STD))

    _loader_kwargs = dict(batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY, persistent_workers=(CFG.NUM_WORKERS > 0), prefetch_factor=(2 if CFG.NUM_WORKERS > 0 else None))
    train_loader = DataLoader(train_dataset, shuffle=True, **_loader_kwargs)
    val_loader   = DataLoader(val_dataset, shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_dataset, shuffle=False, **_loader_kwargs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CFG.POS_WEIGHT], device=DEVICE))

    target_seeds = [args.seed] if args.seed is not None else CFG.SEEDS

    # Aggregation
    if args.aggregate_only:
        all_rows = []
        all_candidate_models = ['HierGeoNet', 'GCN', 'GraphSAGE', 'GAT', 'CNN', 'LR', 'RF', 'XGBoost']
        for model_name in all_candidate_models:
            seed_metrics = []
            for s in CFG.SEEDS:
                marker = MODEL_DIR / f'{"hiergeonet" if model_name=="HierGeoNet" else f"baseline_{model_name}"}_seed{s}_test_metrics.json'
                if marker.exists():
                    with open(marker) as f: seed_metrics.append(json.load(f))
            if seed_metrics:
                all_rows.append(aggregate_seed_results(model_name, seed_metrics))
        
        if all_rows:
            df_bench = pd.DataFrame(all_rows).set_index('Model').sort_values('AUC_mean', ascending=False)
            df_bench.to_csv(OUT_DIR / 'benchmark_comparison_multiseed.csv')
            print("\n" + "="*72)
            print(df_bench[['AUC_mean', 'AUC_std', 'F1_mean', 'F1_std', 'n_seeds']].to_string())
            print("="*72)
        else:
            print("No test metric JSON files found to aggregate.")
        return

    # Train HierGeoNet
    if args.model in ['HierGeoNet', 'all']:
        for s in target_seeds: train_hiergeonet_one_seed(s, train_loader, val_loader, test_loader, ei_f, ei_m, ei_c, coords_c_t, coords_f_t, f2m_t, m2c_t, f2c_t, N_med, N_coarse, criterion)

    # Train GNN/CNN Baselines
    gnn_classes = {'GCN': GCNBaseline, 'GraphSAGE': GraphSAGEBaseline, 'GAT': GATBaseline, 'CNN': CNNBaseline}
    for gname, gcls in gnn_classes.items():
        if args.model in [gname, 'all']:
            for s in target_seeds: train_gnn_baseline_one_seed(gname, gcls, s, train_loader, val_loader, test_loader, ei_f, criterion)

    # Train Flat Baselines (LR, RF, XGBoost)
    if args.model in ['Flat', 'LR', 'RF', 'XGBoost', 'all']:
        def _collect_flat(loader, max_batches=None):
            Xs, Ys = [], []
            for i, (x, y) in enumerate(loader):
                if max_batches and i >= max_batches: break
                Xs.append(x.reshape(-1, CFG.IN_DIM).numpy())
                Ys.append(y.reshape(-1).numpy())
            return np.nan_to_num(np.concatenate(Xs).astype(np.float32)), np.concatenate(Ys).astype(np.int32)

        X_tr, y_tr = _collect_flat(train_loader)
        X_te, y_te = _collect_flat(test_loader)
        
        # Subsample training data if too large to fit in memory
        if len(X_tr) > 2_000_000:
            np.random.seed(CFG.SEEDS[0])
            sub_idx = np.random.choice(len(X_tr), 2_000_000, replace=False)
            X_tr, y_tr = X_tr[sub_idx], y_tr[sub_idx]

        custom_weight = {0: 1.0, 1: float(CFG.POS_WEIGHT)}

        for s in target_seeds:
            if args.model in ['Flat', 'LR', 'all']:
                fit_flat_baseline_with_resume(
                    LogisticRegression(max_iter=500, class_weight=custom_weight, random_state=s),
                    'LR', s, X_tr, y_tr, X_te, y_te
                )
            if args.model in ['Flat', 'RF', 'all']:
                fit_flat_baseline_with_resume(
                    RandomForestClassifier(n_estimators=200, class_weight=custom_weight, n_jobs=4, random_state=s),
                    'RF', s, X_tr, y_tr, X_te, y_te
                )
            if args.model in ['Flat', 'XGBoost', 'all']:
                fit_flat_baseline_with_resume(
                    xgb.XGBClassifier(n_estimators=200, scale_pos_weight=CFG.POS_WEIGHT, n_jobs=4, random_state=s, verbosity=0, eval_metric='logloss'),
                    'XGBoost', s, X_tr, y_tr, X_te, y_te
                )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()