from __future__ import annotations
import math
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from poker_transformer import Normalizer,PokerTransformerClassifier,PokerTransformerConfig

from typing import Optional, Union


pid_to_idx = {f"p{i}": i - 1 for i in range(1, 7)}


class RunningStandardizer(nn.Module):
    """Stores mean/std for continuous columns and applies (x - mean) / std with std>0."""

    def __init__(self, n_features: int, eps: float = 1e-6, trainable: bool = False):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.eps = eps
        if trainable:
            self.mean = nn.Parameter(self.mean)
            self.std = nn.Parameter(self.std)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        X: [N, T, F], mask: [N, T] or None (True for valid)
        Computes mean/std across valid time steps only.
        """
        if mask is not None:
            m = mask.float().unsqueeze(-1)  # [N,T,1]
            total = m.sum(dim=(0, 1))  # [1]
            total = torch.clamp(total, min=1.0)
            mu = (X * m).sum(dim=(0, 1)) / total
            var = ((X - mu) ** 2 * m).sum(dim=(0, 1)) / total
        else:
            mu = X.mean(dim=(0, 1))
            var = X.var(dim=(0, 1), unbiased=False)
        std = torch.sqrt(var + self.eps)
        self.mean.copy_(mu)
        self.std.copy_(std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / (self.std + self.eps)

# ================================
# Model: Input projection -> Mamba -> Classification head
# ================================
@dataclass
class PokerMambaConfig:
    d_model: int = 256
    n_layers: int = 4
    dropout: float = 0.1
    # Optional categorical support (indices in feature dim and their cardinalities)
    categorical_idx: Optional[List[int]] = None
    categorical_cardinalities: Optional[List[int]] = None
    embed_dim: int = 32  # per categorical feature


class PokerMambaClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, cfg: PokerMambaConfig):
        super().__init__()
        self.cfg = cfg
        cat_idx = cfg.categorical_idx or []
        self.cat_idx = torch.tensor(cat_idx, dtype=torch.long) if len(cat_idx) else None
        self.cont_idx = (
            [i for i in range(n_features) if i not in cat_idx]
            if len(cat_idx)
            else list(range(n_features))
        )

        # Embeddings for categorical features
        self.embeds = None
        embed_total = 0
        if len(cat_idx):
            cards = cfg.categorical_cardinalities
            assert cards and len(cards) == len(cat_idx), "Provide cardinalities for categorical_idx"
            self.embeds = nn.ModuleList([nn.Embedding(c, cfg.embed_dim) for c in cards])
            embed_total = len(cat_idx) * cfg.embed_dim

        # Linear projection for continuous features
        self.cont_in = None
        cont_dim = len(self.cont_idx)
        if cont_dim > 0:
            self.cont_in = nn.Linear(cont_dim, max(1, cfg.d_model - embed_total))

        # If only embeddings exist or projection smaller than d_model, final fuse to d_model
        fused_in = (cfg.d_model - embed_total if self.cont_in else 0) + embed_total
        self.fuse = (
            nn.Identity()
            if fused_in == cfg.d_model
            else nn.Linear(fused_in, cfg.d_model)
        )

        # Mamba encoder (sequence in/out [B,T,d_model])
        mcfg = MambaConfig(d_model=cfg.d_model, n_layers=cfg.n_layers)
        self.backbone = Mamba(mcfg)

        # Head
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.d_model, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, T, F] -> logits: [B, T, C]"""
        B, T, F = X.shape
        pieces = []
        if self.embeds is not None and self.cat_idx is not None and len(self.cat_idx) > 0:
            X_int = X.long()  # assume categorical features already integer-coded
            cats = [X_int[..., i] for i in self.cat_idx.tolist()]  # each [B,T]
            emb = [emb(c) for emb, c in zip(self.embeds, cats)]  # each [B,T,embed_dim]
            pieces.append(torch.cat(emb, dim=-1))
        if self.cont_in is not None:
            cont = X[..., self.cont_idx].float()
            cont = self.cont_in(cont)
            pieces.append(cont)
        H = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
        H = self.fuse(H)
        H = self.backbone(H)
        H = self.dropout(H)
        logits = self.head(H)
        return logits


# ================================
# Loss with masking & legal action suppression
# ================================
class MaskedLegalCrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, legal: torch.Tensor):
        """
        logits: [B,T,C], target: [B,T], mask: [B,T] bool, legal: [B,T,C] bool
        """
        # Suppress illegal action logits before softmax
        very_neg = torch.finfo(logits.dtype).min / 2
        logits = torch.where(legal, logits, very_neg)

        B, T, C = logits.shape
        logits = logits.reshape(B * T, C)
        target = target.reshape(B * T)
        mask = mask.reshape(B * T)

        if self.label_smoothing > 0:
            # CrossEntropy with label smoothing via log_softmax + nll
            logp = F.log_softmax(logits, dim=-1)
            nll = -logp[torch.arange(B * T, device=logits.device), target]
            smooth = -logp.mean(dim=-1)
            loss_vec = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            loss_vec = F.cross_entropy(logits, target, reduction='none', weight=self.class_weights)

        loss = (loss_vec * mask.float()).sum() / (mask.float().sum() + 1e-8)
        return loss


# ================================
# Metrics
# ================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total, correct, legal_total, legal_correct = 0.0, 0.0, 0.0, 0.0
    per_class_tp = torch.zeros(3)
    per_class_fp = torch.zeros(3)
    per_class_fn = torch.zeros(3)

    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        m = batch["mask"].to(device)
        legal = batch["legal"].to(device)
        logits = model(X)
        very_neg = torch.finfo(logits.dtype).min / 2
        logits = torch.where(legal, logits, very_neg)
        pred = logits.argmax(dim=-1)

        valid = m
        total += valid.float().sum().item()
        correct += ((pred == y) & valid).float().sum().item()

        # masked-legal accuracy (same as above since we zero illegal logits)
        legal_total += valid.float().sum().item()
        legal_correct += ((pred == y) & valid).float().sum().item()

        # per-class precision/recall components
        for c in range(3):
            tp = (((pred == c) & (y == c)) & valid).float().sum().item()
            fp = (((pred == c) & (y != c)) & valid).float().sum().item()
            fn = (((pred != c) & (y == c)) & valid).float().sum().item()
            per_class_tp[c] += tp
            per_class_fp[c] += fp
            per_class_fn[c] += fn

    acc = correct / max(1.0, total)
    legal_acc = legal_correct / max(1.0, legal_total)
    precision = per_class_tp / torch.clamp(per_class_tp + per_class_fp, min=1.0)
    recall = per_class_tp / torch.clamp(per_class_tp + per_class_fn, min=1.0)
    macro_f1 = (2 * precision * recall / torch.clamp(precision + recall, min=1e-8)).mean().item()
    return {
        "acc": acc,
        "legal_acc": legal_acc,
        "macro_f1": macro_f1,
        "precision_fold": precision[0].item(),
        "precision_call": precision[1].item(),
        "precision_raise": precision[2].item(),
        "recall_fold": recall[0].item(),
        "recall_call": recall[1].item(),
        "recall_raise": recall[2].item(),
    }

# ================================
# Train loop
# ================================
@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 2e-4
    max_epochs: int = 10
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 2
    ckpt_dir: str = "checkpoints"


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train(
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    n_classes: int = 3,
    model_cfg: PokerMambaConfig = PokerMambaConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    class_weights: Optional[List[float]] = None,
    save_name: str = "poker_mamba",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SeqDictDataset(train_data)
    val_ds = SeqDictDataset(val_data)

    n_features = train_ds.X.size(-1)

    # Fit normalizer on train only
    normalizer = RunningStandardizer(n_features)
    normalizer.fit(train_ds.X, train_ds.mask)

    # Apply normalization lazily inside a wrapper collate function (avoid storing extra copy)
    def collate_fn(samples: List[Dict[str, torch.Tensor]]):
        # Type-guarded collate: only stack when every element is a Tensor
        batch: Dict[str, object] = {}
        keys = list(samples[0].keys())
        for k in keys:
            vals = [s[k] for s in samples]
            if k == "X":
                # X must be stacked & normalized
                X = torch.stack(vals)
                B, T, F = X.shape
                X = X.view(B * T, F)
                X = normalizer(X).view(B, T, F)
                batch["X"] = X
            elif all(torch.is_tensor(v) for v in vals):
                batch[k] = torch.stack(vals)
            else:
                # e.g., hand_id is list[str]; keep as list to carry-through
                batch[k] = vals
        return batch

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    model = PokerMambaClassifier(n_features=n_features, n_classes=n_classes, cfg=model_cfg).to(device)

    # Optimizer & scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, train_cfg.max_epochs))

    # Loss
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    criterion = MaskedLegalCrossEntropy(label_smoothing=train_cfg.label_smoothing, class_weights=cw)

    best_val = -float("inf")
    os.makedirs(train_cfg.ckpt_dir, exist_ok=True)

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            logits = model(batch["X"])  # [B,T,C]
            loss = criterion(logits, batch["y"], batch["mask"], batch["legal"])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()
            total_loss += loss.item()
            steps += 1
        sched.step()

        val_metrics = evaluate(model, val_loader, device)
        train_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_metrics['acc']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}")

        score = val_metrics["macro_f1"]
        if score > best_val:
            best_val = score
            ckpt = {
                "model_state": model.state_dict(),
                "normalizer": {
                    "mean": normalizer.mean.cpu(),
                    "std": normalizer.std.cpu(),
                },
                "model_cfg": asdict(model_cfg),
                "train_cfg": asdict(train_cfg),
                "n_features": n_features,
                "n_classes": n_classes,
            }
            path = os.path.join(train_cfg.ckpt_dir, f"{save_name}.pt")
            torch.save(ckpt, path)
            print(f"Saved checkpoint -> {path}")

    # Final eval report
    val_metrics = evaluate(model, val_loader, device)
    print("Final validation:", json.dumps(val_metrics, indent=2))



# ================================
# Inference helper (single batch, with legal mask)
# ================================
@torch.no_grad()
def infer_logits(model: nn.Module, normalizer: RunningStandardizer, batch: Dict[str, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    X = batch["X"].to(device)
    m = batch.get("mask")
    if m is not None:
        B, T, F = X.shape
        X = X.view(B * T, F)
        X = normalizer(X).view(B, T, F)
    else:
        X = normalizer(X)
    logits = model(X)
    legal = batch.get("legal")
    if legal is not None:
        legal = legal.to(device)
        very_neg = torch.finfo(logits.dtype).min / 2
        logits = torch.where(legal, logits, very_neg)
    return logits

# ================================
# Data utilities
# ================================
class SeqDictDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.X     = torch.as_tensor(data["X"]).float()
        self.y     = torch.as_tensor(data["y"]).long()
        self.mask  = torch.as_tensor(data["mask"]).bool()
        self.legal = torch.as_tensor(data["legal"]).bool()

        hid = data.get("hand_id", data.get("hand_ids"))
        if hid is None:
            hid = list(range(self.X.shape[0]))
        self.hand_id = hid  # keep as list[str] or list[int]; no torch.as_tensor here


        assert self.X.ndim == 3, "X must be [N, T, F]"
        assert self.y.shape[:2] == self.X.shape[:2]
        assert self.mask.shape[:2] == self.X.shape[:2]
        assert self.legal.shape[:2] == self.X.shape[:2]
        assert self.legal.size(-1) == 3, "legal last dim must be 3 (fold/call/raise)"

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return {
            "X": self.X[idx],
            "y": self.y[idx],
            "mask": self.mask[idx],
            "legal": self.legal[idx],
            "hand_id": self.hand_id[idx],
        }

# ================================
# (rest of the code unchanged)
# ================================

@torch.no_grad()
def split_by_hand(data: Dict[str, torch.Tensor], val_ratio: float = 0.1, seed: int = 1337):
    import numpy as np

    X = data["X"]
    N = X.shape[0] if not torch.is_tensor(X) else X.size(0)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)
    n_val = max(1, int(N * val_ratio))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    def subset_like(v, ix):
        if torch.is_tensor(v) and v.shape[:1] == (N,):
            return v[ix]
        if isinstance(v, np.ndarray) and v.shape[:1] == (N,):
            return v[ix.cpu().numpy()]
        if isinstance(v, (list, tuple)) and len(v) == N:
            return [v[i.item()] for i in ix]
        return v  # metadata or mismatched shape → carry through

    def take(ix):
        out = {}
        for k, v in data.items():
            if k in ("feature_keys",):
                out[k] = v
            else:
                out[k] = subset_like(v, ix)
        return out

    return take(tr_idx), take(val_idx)


import torch
from typing import Dict, Optional

# --- Rebuild config/normalizer and model from a checkpoint ---
def load_poker_mamba(ckpt_path: str, device: Optional[torch.device] = None):
    from dataclasses import is_dataclass
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) Rebuild model_cfg dataclass
    model_cfg_obj = ckpt.get("model_cfg")
    if isinstance(model_cfg_obj, dict):
        model_cfg_obj = PokerMambaConfig(**model_cfg_obj)
    elif not is_dataclass(model_cfg_obj):
        raise ValueError("model_cfg in checkpoint must be a dict or PokerMambaConfig")

    # 2) Figure out n_features
    if "n_features" in ckpt:
        n_features = int(ckpt["n_features"])
    elif "feature_keys" in ckpt:
        n_features = len(ckpt["feature_keys"])
    else:
        raise ValueError("Checkpoint missing n_features/feature_keys")

    # 3) Rebuild model
    model = PokerMambaClassifier(n_features=n_features, n_classes=3, cfg=model_cfg_obj).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 4) Rebuild normalizer
    norm_stats = ckpt.get("normalizer", {})
    normalizer = RunningStandardizer(n_features=n_features)
    if "mean" in norm_stats and "std" in norm_stats:
        with torch.no_grad():
            normalizer.mean.copy_(torch.as_tensor(norm_stats["mean"]).view(-1))
            normalizer.std.copy_(torch.as_tensor(norm_stats["std"]).view(-1))
    normalizer.eval()

    feature_keys = ckpt.get("feature_keys", None)
    return model, normalizer, feature_keys, model_cfg_obj

# --- Single-step inference (UI) ---
@torch.no_grad()
def infer_single_step(model, normalizer, x_vec: torch.Tensor, legal_mask: torch.Tensor):
    """
    x_vec: [F]    feature vector in the SAME order as `feature_keys`
    legal_mask: [3] bool/int {1=legal,0=illegal} for [fold, call, raise]
    """
    device = next(model.parameters()).device
    x = x_vec.float().view(1, 1, -1).to(device)   # [1,1,F]
    x = normalizer(x.view(-1, x.shape[-1])).view(1, 1, -1)

    logits = model(x)                              # [1,1,3]
    legal = legal_mask.to(device).view(1, 1, -1).bool()
    very_neg = torch.finfo(logits.dtype).min / 2
    logits = torch.where(legal, logits, very_neg)
    action = logits.argmax(dim=-1).item()          # 0=fold,1=call,2=raise
    return action, logits.squeeze(0).squeeze(0).cpu()

#model, normalizer, feature_keys, model_cfg = load_poker_mamba("checkpoints/poker_mamba.pt")

# Build x_vec in EXACT feature_keys order (or the same order your training X had)
# x_vec = torch.tensor([...], dtype=torch.float32)
# legal_mask = torch.tensor([1,1,0], dtype=torch.uint8)  # example

#action, logits = infer_single_step(model, normalizer, x_vec, legal_mask)

"""
Mamba‑ready sequential extractor for PHH hands with simplified 3-class labels: fold, call, raise, plus PHH zip processor.
"""

from typing import Any, Dict, List, Optional, Tuple
import itertools
import numpy as np
import zipfile

def process_phh_zip(zip_path: str, name: str = "Pluribus") -> List[Dict[str, Any]]:
    """Process a .zip of PHH files into Mamba-ready sequences using extract_pluribus_actions_mamba."""
    sequences = []
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for fname in zipf.namelist():
            if not fname.endswith('.phh'):
                continue
            try:
                content = zipf.read(fname).decode('utf-8')
                hand = {}
                for line in content.splitlines():
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        hand[k.strip()] = v.strip()
                seq = extract_pluribus_actions_mamba(hand, name=name, file_path=fname)
                if seq["steps"]:
                    sequences.append(seq)
            except Exception as e:
                print(f"⚠️ Failed on {fname}: {e}")
    return sequences


# --- Simple PHH parser (string -> dict) ---------------------------------------
import ast, zipfile

def parse_phh_string(file_content: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in file_content.splitlines():
        if '=' in line:
            key, val = line.strip().split('=', 1)
            key = key.strip()
            val = val.strip()
            try:
                data[key] = ast.literal_eval(val)
            except Exception:
                data[key] = val
    return data

# --- Zip processor wired to the new extractor ---------------------------------

def process_phh_zip_mamba(
    zip_path: str,
    *,
    target_name: str = "Pluribus",
    include_card_ints: bool = True,
    include_basic_scalars: bool = True,
    collate: bool = False,
    feature_keys: Optional[List[str]] = None,
) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Read .phh files from a zip and build Mamba-ready sequences using
    `extract_pluribus_actions_mamba`.

    If `collate=True`, returns a dict of padded arrays suitable for training.
    Otherwise returns a list of per-hand sequence dicts.
    """
    sequences: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for name in zipf.namelist():
            if not name.endswith('.phh'):
                continue
            try:
                content = zipf.read(name).decode('utf-8')
                raw = parse_phh_string(content)  # typed dict

                hand_for_extractor = {
                    'actions': repr(raw.get('actions', [])),
                    'players': repr(raw.get('players', [])),
                    'blinds_or_straddles': repr(raw.get('blinds_or_straddles', [])),
                    'starting_stacks': repr(raw.get('starting_stacks', [])),
                    'hand': raw.get('hand', name),
                }
                seq = extract_pluribus_actions_mamba(
                    hand_for_extractor,
                    name=target_name,
                    include_card_ints=include_card_ints,
                    include_basic_scalars=include_basic_scalars,
                    file_path=name,
                )
                if seq.get('steps'):
                    sequences.append(seq)
            except Exception as e:
                print(f"⚠️ Failed on {name}: {e}")

    if collate:

        return collate_mamba_batch(sequences, feature_keys=feature_keys)
    return sequences

"""
Mamba‑ready sequential extractor for PHH hands with simplified 3-class labels: fold, call, raise.
"""

from typing import Any, Dict, List, Optional, Tuple
import itertools
import numpy as np

RANK_MAP = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUIT_MAP = {'c': 0, 'd': 1, 'h': 2, 's': 3}

PHASE_IDS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "unknown": -1}

ACTION_VOCAB = ["fold", "call", "raise"]
NUM_ACTIONS = len(ACTION_VOCAB)  # -> 3
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_VOCAB)}


# --- ADD: Chen preflop strength (single, continuous feature) ---
# Ref: "Chen formula" style heuristic. Monotone, pocket pairs valued highly,
# suited/connectors get credit, big gaps penalized.

RANK_ORDER = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
RANK_TO_VAL = {r:i for i, r in enumerate(RANK_ORDER, start=2)}  # 2..14

def chen_strength_2c(hh: str) -> float:
    """
    hh: like 'AhKd' or '2c2d' (exactly 4 chars)
    Returns a continuous strength; AA ~ 20, small offsuit/gapped ~ low.
    """
    r1, s1 = hh[0].upper(), hh[1].lower()
    r2, s2 = hh[2].upper(), hh[3].lower()
    v1, v2 = RANK_TO_VAL[r1], RANK_TO_VAL[r2]
    high, low = max(v1, v2), min(v1, v2)
    suited = (s1 == s2)
    gap = max(0, (high - low - 1))

    # Base = high card value / 2, with Ace rounded up as in classic Chen (≈10)
    base = 10.0 if high == 14 else high / 2.0

    # Pairs: 10 + (rank/2), but tiny pairs get a small bump over high-card base
    if v1 == v2:
        return 10.0 + high / 2.0

    # Suited bonus
    base += 2.0 if suited else 0.0

    # Gap penalties
    if gap == 1: base -= 1.0
    elif gap == 2: base -= 2.0
    elif gap == 3: base -= 4.0
    elif gap >= 4: base -= 5.0

    # Small connector bonus (no gap) for 5-10 region
    if gap == 0 and high <= 10:
        base += 1.0

    return max(0.0, base)


# --- ADD: robust ante detection from blinds list ---
def infer_blinds_antes(blinds: list[int], n_players: int) -> tuple[int, int, int]:
    """
    Attempts: blinds = [SB, BB, (optional uniform ante ...)]
    Returns (sb, bb, ante_per_player). If unknown, ante=0.
    """
    sb = int(blinds[0]) if len(blinds) >= 1 else 0
    bb = int(blinds[1]) if len(blinds) >= 2 else max(1, sb * 2)  # keep >=1
    ante = 0
    if len(blinds) >= 3:
        # Heuristic: if remaining entries equal, treat that value as uniform ante
        rest = blinds[2:]
        if all(x == rest[0] for x in rest):
            ante = int(rest[0])
        else:
            # If they provided n_players values, compute average
            if len(rest) == n_players and sum(rest) > 0:
                ante = int(round(sum(rest) / n_players))
    return max(0, sb), max(1, bb), max(0, ante)


def encode_card(card_str: str) -> Tuple[int, int]:
    return RANK_MAP[card_str[0].upper()], SUIT_MAP[card_str[1].lower()]

def encode_hand(hand_str: str) -> Tuple[int, int, int, int]:
    r1, s1 = encode_card(hand_str[:2])
    r2, s2 = encode_card(hand_str[2:])
    return r1, s1, r2, s2

def _split_cards(token: str) -> List[str]:
    return [token[i:i + 2] for i in range(0, len(token), 2)]

def determine_phase(upto_idx: int, actions: List[str]) -> str:
    revealed = 0
    for i in range(upto_idx + 1):
        a = actions[i]
        if a.startswith("d db "):
            revealed += len(_split_cards("".join(a.split()[2:])))
    if revealed == 0:
        return "preflop"
    if revealed >= 5:
        return "river"
    if revealed == 4:
        return "turn"
    return "flop"

def _board_so_far(upto_idx: int, board_events: List[Tuple[int, List[str]]]) -> List[str]:
    return list(itertools.chain.from_iterable(cards for ei, cards in board_events if ei < upto_idx))

def _to_call(max_bet: int, already_put_in: int) -> int:
    return max(0, max_bet - already_put_in)

def _phase_from_board_len(n: int) -> str:
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n, "unknown")

def _label_preflop(a_type: str, amt: Optional[int], bb_amt: int, stack_remaining: int) -> str:
    if a_type == "f":
        return "fold"
    if a_type == "cc":
        return "call"
    if a_type == "cbr":
        return "raise"
    return "unknown"

def _label_postflop(a_type: str, amt: Optional[int], cur_pot: float, stack_remaining: int) -> str:
    if a_type == "f":
        return "fold"
    if a_type == "cc":
        return "call"
    if a_type == "cbr":
        return "raise"
    return "unknown"

def _legal_mask(_: str) -> List[int]:

    return [1] * NUM_ACTIONS   # now length 3

# --- Preflop all-in tracking ---------------------------------------------------
from dataclasses import dataclass

@dataclass
class PreflopAllInTracker:
    """Tracks preflop shove context up to the hero's decision."""
    allin_seen: bool = False                 # any opponent has shoved already?
    first_aggr_seen: bool = False            # has the first aggressive action happened?
    first_aggr_was_allin: bool = False       # was that first aggression a shove?
    last_allin_total_putin: int = 0          # committed chips by the shover

def update_preflop_allin_tracker(
    tracker: PreflopAllInTracker,
    *,
    pid: str,                # e.g., 'p3'
    a_type: str,             # 'f' / 'cc' / 'cbr'
    a_amt: int | None,
    put_in: dict[str, int],  # current committed per player (mutable outside)
    stacks0: list[int],      # starting stacks
    street: str,             # current street from your loop ('preflop'/...)
):
    """Call this for every OPPONENT action before the hero acts."""
    if street != "preflop":
        return

    if a_type in ("cc", "cbr"):
        # opponent has added chips; their total committed is already in put_in
        committed = put_in.get(pid, 0)
        opp_idx = int(pid[1:]) - 1
        remaining = int(stacks0[opp_idx]) - committed

        # mark first aggression (a raise/bet) if this is 'cbr'
        if a_type == "cbr" and not tracker.first_aggr_seen:
            tracker.first_aggr_seen = True
            tracker.first_aggr_was_allin = (remaining <= 0)

        # if they are all-in now, remember it
        if remaining <= 0 and committed > 0:
            tracker.allin_seen = True
            tracker.last_allin_total_putin = committed

def _facing_allin_preflop(tracker: PreflopAllInTracker, street: str) -> int:
    """1 if any opponent is already all-in before hero acts (preflop only)."""
    return int(street == "preflop" and tracker.allin_seen)

def _is_open_shove_preflop(tracker: PreflopAllInTracker, street: str) -> int:
    """1 if the first aggressive action preflop was an all-in shove."""
    return int(street == "preflop" and tracker.first_aggr_was_allin)


# --- Preflop/hole-card helpers -------------------------------------------------

def is_suited_hole(r1: int, s1: int, r2: int, s2: int) -> int:
    """1 if hero hole cards are suited."""
    return int(s1 == s2)

def gap_size_hole(r1: int, r2: int) -> int:
    """Absolute gap between ranks (A=14). E.g., A5 -> 14 & 5 => gap=8 (big), 76 -> gap=0."""
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    return max(0, hi - lo - 1)

def is_connected_hole(r1: int, r2: int, allow_one_gap: bool = True) -> int:
    """
    1 if connected (gap==0), or (optionally) 1-gap connected like T8 (gap==1).
    """
    g = gap_size_hole(r1, r2)
    return int(g == 0 or (allow_one_gap and g == 1))


def flush_draw_flag(hole_s1: int, hole_s2: int, board_suits: list[int]) -> int:
    """
    Returns 1 if there are exactly 4 (or more) of the same suit among
    hole + visible board (i.e., a 4-flush draw before river).
    board_suits entries are {0,1,2,3} for known cards and -1 for unknown slots.
    """
    counts = [0, 0, 0, 0]
    for s in (hole_s1, hole_s2):
        if 0 <= s <= 3:
            counts[s] += 1
    for s in board_suits:
        if 0 <= s <= 3:
            counts[s] += 1
    return int(max(counts) >= 4)  # 4-to-a-flush or better


def straight_draw_flag(hole_r1: int, hole_r2: int, board_ranks: list[int]) -> int:
    """
    Very compact rank-bit heuristic:
    - Build presence mask over ranks 2..A (2..14)
    - If any 5-card straight window (A-5 .. T-A) has >=4 distinct ranks present -> draw.
    (This will flag OESD and most gutshots; duplicates don’t count twice.)
    """
    present = [0] * 15  # index by rank, ignore 0/1
    for r in (hole_r1, hole_r2):
        if 2 <= r <= 14:
            present[r] = 1
    for r in board_ranks:
        if 2 <= r <= 14:
            present[r] = 1

    # windows: [A-5] is 14,5,4,3,2; we’ll also check 10..14 (T..A)
    # Implement as numeric windows 2..6, 3..7, ..., 10..14; and a special wheel (A-5)
    def window_count(lo: int, hi: int) -> int:
        return sum(present[r] for r in range(lo, hi + 1))

    # normal windows
    for lo in range(2, 11):  # 2..10 inclusive -> windows [2..6]..[10..14]
        if window_count(lo, lo + 4) >= 4:
            return 1

    # wheel A-5 (A=14 plus 2..5)
    if present[14] + sum(present[r] for r in (2, 3, 4, 5)) >= 4:
        return 1

    return 0


from typing import Any, Dict, List, Optional, Set, Tuple

def extract_pluribus_actions_mamba(
    hand: Dict[str, Any],
    *,
    name: str = "Pluribus",
    include_card_ints: bool = True,
    include_basic_scalars: bool = True,
    file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a hand dict into Mamba-style sequential steps with features and labels.
    Adds preflop push/fold EV features using a 'functional all-in' definition:
    any opponent action that makes the current price to continue >= SHOVE_THRESHOLD
    of hero's remaining stack.
    """

    # ── Parse raw fields (kept as in your original) ────────────────────────────
    actions: List[str] = eval(hand.get("actions", []))
    players: List[str] = eval(hand.get("players", []))
    blinds: List[int] = eval(hand.get("blinds_or_straddles", []))
    stacks0: List[int] = eval(hand.get("starting_stacks", []))

    n_players = len(players) if players else 6
    sb_amt, bb_amt_inferred, ante_pp = infer_blinds_antes(blinds, n_players)
    # Prefer inferred BB, fall back to literal in blinds, then 1
    bb_amt = (blinds[1] if len(blinds) > 1 else 0) or bb_amt_inferred or 1

    # Guard: hero must be seated
    if not players or name not in players:
        return {"hand_id": f"{file_path}_{hand.get('hand', 'unknown')}", "steps": []}

    hero_tag = f"p{players.index(name) + 1}"
    p_idx = players.index(name)

    # ── State trackers ────────────────────────────────────────────────────────
    put_in: Dict[str, float] = {f"p{i}": 0.0 for i in range(1, 7)}
    cur_pot: float = float(sum(blinds))
    hero_stack: float = float(stacks0[p_idx])

    # Board reveal events to know current street by index
    board_events = [(i, _split_cards("".join(a.split()[2:])))
                    for i, a in enumerate(actions) if a.startswith("d db ")]

    # Hero hole cards
    hero_hand = next((a.split()[-1] for a in actions if a.startswith(f"d dh {hero_tag} ")), None)
    if hero_hand is None:
        return {"hand_id": f"{file_path}_{hand.get('hand', 'unknown')}", "steps": []}
    r1, s1, r2, s2 = encode_hand(hero_hand)

    steps: List[Dict[str, Any]] = []
    max_bet = 0.0
    current_phase = "preflop"
    live: Set[str] = {f"p{i}" for i in range(1, 7)}

    seat_order = [f"p{i}" for i in range(1, 7)]
    hero_seat = seat_order.index(hero_tag)

    # Keep your prior flags (can be useful elsewhere)
    aggr_preflop_seen = False

    # ── NEW: create preflop tracker ONCE + functional-all-in buffers ─────────
    pf_tracker = PreflopAllInTracker()
    shove_putins: List[float] = []             # total committed by each (functional) all-in actor
    first_shover_pid: Optional[str] = None     # pid of first (functional) all-in

    # ── Iterate through actions ───────────────────────────────────────────────
    for idx, act in enumerate(actions):
        board_sofar = _board_so_far(idx, board_events)
        street = _phase_from_board_len(len(board_sofar))

        # Street change → reset per-street put-ins/max_bet (your original logic)
        if street != current_phase:
            max_bet = 0.0
            for k in put_in:
                put_in[k] = 0.0
            current_phase = street

        # We only care about player actions starting with 'p'
        if not act.startswith("p"):
            continue

        parts = act.split()
        pid = parts[0]
        a_type = parts[1] if len(parts) > 1 else ""
        a_amt = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

        # ── Opponent actions (update pot/put-in/tracker, detect functional all-in)
        if pid != hero_tag:
            if a_type in ("cc", "cbr"):
                amt = max(0, a_amt or 0)
                put_in[pid] += float(amt)
                max_bet = max(max_bet, put_in[pid])
                cur_pot += float(amt)

                # update tracker for preflop all-in semantics you already had
                update_preflop_allin_tracker(
                    pf_tracker, pid=pid, a_type=a_type, a_amt=a_amt,
                    put_in=put_in, stacks0=stacks0, street=street
                )

                # Mark first aggression on preflop
                if street == "preflop" and a_type == "cbr" and not aggr_preflop_seen:
                    aggr_preflop_seen = True

                # ── NEW: functional all-in detection (≥ threshold of hero stack)
                if street == "preflop" and a_type == "cbr":
                    new_putin = float(put_in[pid])
                    hero_already_putin = float(put_in[hero_tag])
                    hero_remaining_stack_now = float(hero_stack)

                    if _is_functional_allin_for_hero(
                        new_putin_opp=new_putin,
                        hero_already_putin=hero_already_putin,
                        hero_remaining_stack=hero_remaining_stack_now,
                        threshold=SHOVE_THRESHOLD,
                    ):
                        shove_putins.append(new_putin)
                        if first_shover_pid is None:
                            first_shover_pid = pid

            # Folds: mark player dead
            if a_type == "f":
                live.discard(pid)

        # ── Hero decision point: build features + label ───────────────────────
        if pid == hero_tag:
            already = float(put_in[hero_tag])
            to_call = float(_to_call(max_bet, already))
            pot_odds = (to_call / max(cur_pot + to_call, 1.0)) if to_call > 0 else 0.0
            spr = (hero_stack / max(cur_pot, 1.0)) if cur_pot > 0 else 0.0
            call_vs_pot = to_call / max(cur_pot, 1.0)

            players_alive = len(live)
            denom = float(sb_amt + bb_amt + ante_pp * players_alive)
            m_ratio = float(hero_stack) / max(1.0, denom)

            # Count players to act after hero this street
            players_to_act_after = 0
            for step_ahead in range(1, 6):
                pid_after = seat_order[(hero_seat + step_ahead) % 6]
                if pid_after in live:
                    players_to_act_after += 1

            # Simple 6-max pos bucket using seat index
            hero_pos_bucket = 0 if p_idx in (0, 1) else (1 if p_idx in (2, 3) else 2)

            # Encode board (optional)
            board_ranks: List[int] = []
            board_suits: List[int] = []
            if include_card_ints:
                for i in range(5):
                    if i < len(board_sofar):
                        rk, st = encode_card(board_sofar[i])
                    else:
                        rk, st = 0, -1
                    board_ranks.append(int(rk))
                    board_suits.append(int(st))

            # Base feature dict
            x: Dict[str, Any] = {"phase_id": PHASE_IDS.get(street, -1)}
            # always define them
            x.setdefault("facing_allin_preflop", 0)
            #x.setdefault("n_allins_preflop", 0)
            #x.setdefault("any_overcall_preflop", 0)
            #x.setdefault("sum_allin_putin_bb", 0.0)
            #x.setdefault("max_allin_putin_bb", 0.0)
            #x.setdefault("eff_stack_vs_first_shove_bb", 0.0)
            #x.setdefault("to_call_vs_first_shove_bb", 0.0)
            #x.setdefault("players_behind_who_can_overcall", 0)
            #x.setdefault("max_stack_behind_bb", 0.0)
            #x.setdefault("sum_stack_behind_bb", 0.0)
            #x.setdefault("pot_odds_vs_shove", 0.0)
            #x.setdefault("req_eq_no_rake", 0.0)
            #x.setdefault("req_eq_with_rake", 0.0)
            x.setdefault("is_pair", 0)
            x.setdefault("pair_rank_bucket", 0)
            #x.setdefault("has_ace", 0)
            x.setdefault("has_broadway", 0)
            #x.setdefault("suited_connector_bucket", 0)
            #x.setdefault("aces_blocker_combo", 0)

            x.update({
                "hero_pos_bucket": int(hero_pos_bucket),
            })

            if include_basic_scalars:
                x.update({
                    "spr": float(spr),
                    "pot_odds": float(pot_odds),
                    "opponents_live": int(len(live) - 1 if hero_tag in live else len(live)),
                    "call_vs_pot": float(call_vs_pot),
                    # add m_ratio if you later want it:
                    # "m_ratio": float(m_ratio),
                })

            if include_card_ints:
                x.update({
                    "hole_r1": int(r1), "hole_s1": int(s1),
                    "hole_r2": int(r2), "hole_s2": int(s2),
                    "b1_r": board_ranks[0], "b1_s": board_suits[0],
                    "b2_r": board_ranks[1], "b2_s": board_suits[1],
                    "b3_r": board_ranks[2], "b3_s": board_suits[2],
                    "b4_r": board_ranks[3], "b4_s": board_suits[3],
                    "b5_r": board_ranks[4], "b5_s": board_suits[4],
                })

            # Hole-card booleans + draw hints (board-aware)
            x["is_suited"] = is_suited_hole(r1, s1, r2, s2)
            x["is_connected"] = is_connected_hole(r1, r2, allow_one_gap=True)

            board_ranks_vis = [rk for rk in board_ranks if rk > 0]
            board_suits_vis = [st for st in board_suits if st >= 0]
            x["flush_draw"] = flush_draw_flag(s1, s2, board_suits_vis)
            x["straight_draw"] = straight_draw_flag(r1, r2, board_ranks_vis)

            x["hand_strength_chen"] = float(chen_strength_2c(hero_hand)) if hero_hand else 0.0

            # ── NEW: push/fold EV features (only preflop) ─────────────────────
            if street == "preflop":
                # Facing all-in per your existing tracker semantics (works fine with functional too)
                facing_allin = _facing_allin_preflop(pf_tracker, street)
                x["facing_allin_preflop"] = int(facing_allin)


                # Multiway all-in context

                # Effective stack and price vs first shover
                eff_vs_first, to_call_vs_first = 0.0, 0.0
                if first_shover_pid is not None:
                    f_idx = int(first_shover_pid[1:]) - 1
                    hero_committed = float(put_in[hero_tag])
                    first_putin = float(put_in[first_shover_pid])
                    # total effective between hero and first shover
                    eff_vs_first = min(hero_stack + hero_committed, float(stacks0[f_idx]))
                    to_call_vs_first = max(0.0, first_putin - hero_committed)


                # Players behind who can still act (and their stacks at risk)
                players_behind: List[str] = []
                for step_ahead in range(1, 6):
                    pid_after = seat_order[(hero_seat + step_ahead) % 6]
                    if pid_after in live and pid_after != hero_tag:
                        players_behind.append(pid_after)
                stacks_behind: List[int] = []
                for p in players_behind:
                    idx = pid_to_idx.get(p, None)
                    base_stack = int(stacks0[idx]) if (idx is not None and idx < len(stacks0)) else 0
                    committed  = int(put_in.get(p, 0))
                    stacks_behind.append(max(0, base_stack - committed))


                # Current price and required equity (with/without rake)
                already_now = float(put_in[hero_tag])
                to_call_now = float(_to_call(max_bet, already_now))
                pot_before_call = float(cur_pot)


                # Cheap equity proxies / blockers
                x.update(_hand_buckets(r1, s1, r2, s2))

            # Label as in your original code
            label_str = (
                _label_preflop(a_type, a_amt, bb_amt, int(hero_stack))
                if street == "preflop"
                else _label_postflop(a_type, a_amt, float(cur_pot), int(hero_stack))
            )
            if label_str not in ACTION_TO_ID:
                # skip steps you don't want to train on (e.g., unknown)
                continue

            y = int(ACTION_TO_ID[label_str])

            # Append step
            steps.append({
                "x": x,
                "y": y,
                "phase": PHASE_IDS.get(street, -1),
                "legal_mask": _legal_mask(street)
            })

            # Apply hero’s action to state
            if a_type in ("cc", "cbr"):
                amt = max(0, a_amt or 0)
                put_in[hero_tag] += float(amt)
                hero_stack = max(0.0, hero_stack - float(amt))
                max_bet = max(max_bet, put_in[hero_tag])
                cur_pot += float(amt)
            if a_type == "f":
                live.discard(hero_tag)

    return {"hand_id": f"{file_path}_{hand.get('hand', 'unknown')}", "steps": steps}

from itertools import permutations
from copy import deepcopy

# ---- tiny helpers that mirror your feature conventions ----

def _recompute_hole_flags(x):
    r1, s1, r2, s2 = x.get("hole_r1",0), x.get("hole_s1",-1), x.get("hole_r2",0), x.get("hole_s2",-1)
    # reuse your own helpers if they’re importable; otherwise quick fallbacks:
    def is_suited(r1,s1,r2,s2): return int(s1 == s2 and s1 >= 0 and s2 >= 0 and r1>0 and r2>0)
    def is_connected(r1,r2,allow_one_gap=True):
        if not (r1 and r2): return 0
        gap = abs(r1 - r2)
        return int(gap == 1 or (allow_one_gap and gap == 2))
    x["is_suited"] = is_suited(r1,s1,r2,s2)
    x["is_connected"] = is_connected(r1,r2,allow_one_gap=True)
    # simple pair/broadway recompute if you use them
    x["is_pair"] = int(r1 == r2 and r1 > 0)
    x["pair_rank_bucket"] = int(max(r1, r2) >= 12)  # example bucket; align to your _hand_buckets
    x["has_broadway"] = int(max(r1, r2) >= 12)
    return x

def _swap_hole_in_x(x):
    x2 = deepcopy(x)
    # swap ranks/suits
    x2["hole_r1"], x2["hole_r2"] = x.get("hole_r2",0), x.get("hole_r1",0)
    x2["hole_s1"], x2["hole_s2"] = x.get("hole_s2",-1), x.get("hole_s1",-1)
    # recompute hole-based flags
    _recompute_hole_flags(x2)
    return x2

def _permute_flop_in_x(x, order):  # order is a tuple with elements from (1,2,3)
    x2 = deepcopy(x)
    br = [x.get("b1_r",0), x.get("b2_r",0), x.get("b3_r",0)]
    bs = [x.get("b1_s",-1), x.get("b2_s",-1), x.get("b3_s",-1)]
    # reorder the first three board cards
    x2["b1_r"], x2["b2_r"], x2["b3_r"] = br[order[0]-1], br[order[1]-1], br[order[2]-1]
    x2["b1_s"], x2["b2_s"], x2["b3_s"] = bs[order[0]-1], bs[order[1]-1], bs[order[2]-1]
    # if you have draw flags that depend on board suits/ranks, recompute quickly:
    # (If you already compute them elsewhere, remove this.)
    return x2

# ---- main augmenter for the extractor output ----

def augment_steps_flip_and_permute(extracted: dict,
                                   do_swap_hole_preflop: bool = True,
                                   do_permute_flop: bool = True,
                                   unique_only: bool = True) -> dict:
    """
    Takes the dict returned by extract_pluribus_actions_mamba and returns a new dict
    with additional augmented steps:
      - Preflop: hole-card swap
      - Flop: all 6 permutations of the 3 board cards
    Keeps 'y', 'phase', and 'legal_mask' untouched.
    """
    base_steps = extracted.get("steps", [])
    aug_steps = []

    # to avoid duplicates when cards share ranks (rare), dedup by a hashable key of the x-vector
    def sig(x):
        # include card ints and a few scalars; adjust to your feature set
        keys = [k for k in x.keys() if k.startswith(("hole_", "b1_", "b2_", "b3_", "b4_", "b5_", "phase_id"))]
        keys += ["spr", "pot_odds", "opponents_live", "call_vs_pot", "hero_pos_bucket",
                 "is_suited", "is_connected", "is_pair", "pair_rank_bucket", "has_broadway"]
        return tuple((k, x.get(k, None)) for k in sorted(set(keys)))

    seen = set()

    for step in base_steps:
        x = step["x"]
        phase_id = step.get("phase", x.get("phase_id", -1))
        # 0:preflop, 1:flop, 2:turn, 3:river (per your PHASE_IDS)
        def maybe_add(s):
            if not unique_only or sig(s["x"]) not in seen:
                if unique_only: seen.add(sig(s["x"]))
                aug_steps.append(s)

        # always include original
        maybe_add(step)

        # preflop hole swap
        if do_swap_hole_preflop and phase_id == 0 and "hole_r1" in x:
            s2 = deepcopy(step)
            s2["x"] = _swap_hole_in_x(x)
            maybe_add(s2)

        # flop permutations
        if do_permute_flop and phase_id == 1 and all(k in x for k in ("b1_r","b2_r","b3_r")):
            for order in permutations((1,2,3), 3):  # 6 permutations
                s2 = deepcopy(step)
                s2["x"] = _permute_flop_in_x(x, order)
                maybe_add(s2)

    return {"hand_id": extracted.get("hand_id"), "steps": aug_steps}







SHOVE_THRESHOLD = 0.80  # 80% of hero remaining stack

def _is_functional_allin_for_hero(new_putin_opp: float,
                                  hero_already_putin: float,
                                  hero_remaining_stack: float,
                                  threshold: float = SHOVE_THRESHOLD) -> bool:
    to_call_vs_opp = max(0.0, float(new_putin_opp) - float(hero_already_putin))
    denom = max(1.0, float(hero_remaining_stack))
    return (to_call_vs_opp / denom) >= float(threshold)

def _required_equity_no_rake(pot_before: float, to_call: float) -> float:
    denom = float(pot_before + to_call)
    return float(to_call) / denom if denom > 0 else 1.0

def _required_equity_with_rake(pot_before: float, to_call: float, rake_pct=0.05, rake_cap=3.0) -> float:
    gross = float(pot_before + to_call)
    rake = min(gross * float(rake_pct), float(rake_cap))
    denom = gross - rake
    return float(to_call) / denom if denom > 0 else 1.0

def _bb(x):  # guard
    return max(1, int(x))

def _hand_buckets(r1, s1, r2, s2):
    is_pair = int(r1 == r2 and r1 > 0)
    pair_rank_bucket = 0
    if is_pair:
        pair_rank_bucket = 0 if r1 <= 6 else (1 if r1 <= 10 else 2)  # low/mid/high
    has_ace = int(r1 == 14 or r2 == 14)
    has_broadway = int((r1 >= 10) or (r2 >= 10))
    suited = int(s1 == s2 and s1 >= 0)
    gap = abs(r1 - r2) if (r1 > 0 and r2 > 0) else 99
    is_sc = int(suited and gap == 1)
    is_s1g = int(suited and gap == 2)
    aces_blocker_combo = int(has_ace and suited)  # Axs proxy
    return {
        "is_pair": is_pair,
        "pair_rank_bucket": pair_rank_bucket,
        #"has_ace": has_ace,
        "has_broadway": has_broadway,
        #"suited_connector_bucket": (2 if is_sc else (1 if is_s1g else 0)),
        #"aces_blocker_combo": aces_blocker_combo,
    }



def _dict_list_to_array(dicts: List[Dict[str, Any]], keys: List[str]) -> np.ndarray:
    return np.asarray([[d[k] for k in keys] for d in dicts], dtype=float)

def collate_mamba_batch(hand_seqs: List[Dict[str, Any]], feature_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if not hand_seqs:
        return {"X": np.zeros((0, 0, 0), dtype=np.float32),
                "y": np.zeros((0, 0), dtype=np.int64),
                "mask": np.zeros((0, 0), dtype=np.uint8),
                "legal": np.zeros((0, 0, NUM_ACTIONS), dtype=np.uint8),
                "hand_ids": []}
    if feature_keys is None:
        # Derive keys that *every* step has (safe intersection)
        common_keys = None
        for hand in hand_seqs:
            for step in hand["steps"]:
                ks = set(step["x"].keys())
                common_keys = ks if common_keys is None else (common_keys & ks)
        feature_keys = sorted(list(common_keys))
    B = len(hand_seqs)
    lengths = [len(h["steps"]) for h in hand_seqs]
    T_max = max(lengths) if lengths else 0
    F = len(feature_keys)
    X = np.zeros((B, T_max, F), dtype=np.float32)
    y = np.zeros((B, T_max), dtype=np.int64)
    mask = np.zeros((B, T_max), dtype=np.uint8)
    legal = np.zeros((B, T_max, NUM_ACTIONS), dtype=np.uint8)
    hand_ids = []
    for b, hand_seq in enumerate(hand_seqs):
        hand_ids.append(hand_seq.get("hand_id", str(b)))
        steps = hand_seq["steps"]
        for t, step in enumerate(steps):
            if t >= T_max:
                break
            X[b, t, :] = _dict_list_to_array([step["x"]], feature_keys)[0]
            y[b, t] = int(step["y"])
            mask[b, t] = 1
            legal[b, t, :] = np.asarray(step["legal_mask"], dtype=np.uint8)
    return {"X": X, "y": y, "mask": mask, "legal": legal, "hand_ids": hand_ids, "feature_keys": feature_keys}



def load_poker_transformer(ckpt_path: str, device: Optional[torch.device] = None):
    """
    Returns:
        model: PokerTransformerClassifier (eval mode, on `device`)
        normalizer: Normalizer | None
        feature_keys: Optional[List[str]]
        model_cfg_obj: PokerTransformerConfig
    """
    from dataclasses import is_dataclass
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) Rebuild model_cfg dataclass
    model_cfg_obj = ckpt.get("model_cfg")
    if isinstance(model_cfg_obj, dict):
        model_cfg_obj = PokerTransformerConfig(**model_cfg_obj)
    elif not is_dataclass(model_cfg_obj):
        raise ValueError("model_cfg in checkpoint must be a dict or PokerTransformerConfig")

    # 2) n_features / n_classes
    if "n_features" in ckpt:
        n_features = int(ckpt["n_features"])
    elif "feature_keys" in ckpt:
        n_features = len(ckpt["feature_keys"])
    else:
        raise ValueError("Checkpoint missing n_features/feature_keys")

    n_classes = int(ckpt.get("n_classes", 3))

    # 3) Rebuild model
    model = PokerTransformerClassifier(n_features=n_features, n_classes=n_classes, cfg=model_cfg_obj).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 4) Rebuild normalizer (may be None if you saved without it)
    norm_state = ckpt.get("normalizer", None)
    normalizer = None
    if norm_state is not None:
        normalizer = Normalizer()
        # Accept tensors, lists, or numpy arrays
        normalizer.load_state_dict({
            "mean": torch.as_tensor(norm_state["mean"]).view(-1),
            "std":  torch.as_tensor(norm_state["std"]).view(-1),
        })

    feature_keys = ckpt.get("feature_keys", None)
    return model, normalizer, feature_keys, model_cfg_obj

from poker_transformer import Normalizer,PokerTransformerClassifier


# --- Single-step inference (Transformer) ---
@torch.no_grad()
def infer_single_step_transformer(
    model,
    normalizer,
    x_seq: torch.Tensor,
    legal_mask: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
):
    """
    Args:
        x_seq: [F], [T,F], or [1,T,F]
               - If [F], treats as a single-timestep sequence (T=1)
               - If [T,F], adds batch dim internally
        legal_mask: [C]  1=legal, 0=illegal
        key_padding_mask: optional [T] or [1,T] bool (True where PAD)

    Returns:
        action_id: int       (0=fold,1=call,2=raise)
        logits_last: Tensor  [C] (after legality masking, on CPU)
    """
    device = next(model.parameters()).device

    # Shape to [B=1, T, F]
    x = x_seq.float()
    if x.dim() == 1:              # [F]
        x = x.view(1, 1, -1)
        kpm = torch.zeros((1, 1), dtype=torch.bool, device=device)
    elif x.dim() == 2:            # [T, F]
        x = x.unsqueeze(0)        # [1, T, F]
        kpm = torch.zeros((1, x.size(1)), dtype=torch.bool, device=device)
    elif x.dim() == 3:            # [1, T, F] expected
        kpm = None  # possibly provided
    else:
        raise ValueError("x_seq must have shape [F], [T,F], or [1,T,F]")

    # Key padding mask (batchify if needed)
    if key_padding_mask is None:
        key_padding_mask = kpm
    else:
        k = key_padding_mask
        if k.dim() == 1:
            k = k.unsqueeze(0)  # [1, T]
        key_padding_mask = k.to(device)

    x = x.to(device)

    # Normalize if available
    if normalizer is not None:
        x = normalizer.transform(x)

    # Forward → take last timestep
    logits = model(x, key_padding_mask=key_padding_mask)   # [1, T, C]
    logits_last = logits[:, -1, :]                         # [1, C]

    # Legality mask
    legal = legal_mask.to(device).view(1, -1).bool()
    very_neg = torch.finfo(logits_last.dtype).min / 2
    logits_last = torch.where(legal, logits_last, very_neg)

    action_id = int(logits_last.argmax(dim=-1).item())
    return action_id, logits_last.squeeze(0).detach().cpu()
