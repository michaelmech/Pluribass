from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Basic containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PackedSequences:
    """Container for a batch of variable-length sequences (already padded)."""
    X: torch.Tensor            # [B, T, F]
    y: torch.Tensor            # [B, T]
    key_padding_mask: torch.Tensor  # [B, T] bool; True where PAD (to match PyTorch Transformer)
    legal: Optional[torch.Tensor]   # [B, T, C] or None
    lengths: torch.Tensor      # [B]


class SequenceDataset(Dataset):
    """Dataset over pre-built per-hand sequences.

    Args:
        sequences: list of dicts as described in module docstring.
    """
    def __init__(self, sequences: Sequence[Dict[str, Any]]):
        self.sequences = list(sequences)
        # Validate and cache feature_keys once
        fk = None
        for s in self.sequences:
            if 'feature_keys' in s and s['feature_keys']:
                fk = s['feature_keys']
                break
        self.feature_keys: Optional[List[str]] = fk

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.sequences[idx]
        return {
            'hand_id': item.get('hand_id', str(idx)),
            'X': torch.from_numpy(np.asarray(item['X'], dtype=np.float32)),
            'y': torch.from_numpy(np.asarray(item['y'], dtype=np.int64)),
            'legal': torch.from_numpy(np.asarray(item['legal'], dtype=np.float32)) if 'legal' in item and item['legal'] is not None else None,
            'mask': torch.from_numpy(np.asarray(item['mask'], dtype=np.float32)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Normalizer
# ──────────────────────────────────────────────────────────────────────────────
class Normalizer:
    """Feature-wise standardization over *all* timesteps.

    Computes mean/std per feature using only training data. Apply to tensors
    with shape [..., F]. Keeps small epsilon to avoid division by zero.
    """
    def __init__(self, eps: float = 1e-6):
        self.mean: Optional[torch.Tensor] = None  # [F]
        self.std: Optional[torch.Tensor] = None   # [F]
        self.eps = eps

    def fit(self, X_list: Iterable[torch.Tensor]) -> None:
        # Concatenate along time & batch to compute per-feature stats
        xs = [x.reshape(-1, x.shape[-1]) for x in X_list]
        big = torch.cat(xs, dim=0)  # [N*T, F]
        self.mean = big.mean(dim=0)
        self.std = big.std(dim=0).clamp_min(self.eps)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None, "Call fit() first"
        return (X - self.mean) / self.std

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.mean = state['mean']
        self.std = state['std']


# ──────────────────────────────────────────────────────────────────────────────
# Collation & padding
# ──────────────────────────────────────────────────────────────────────────────

def pad_and_collate(batch: List[Dict[str, Any]], pad_value: float = 0.0) -> PackedSequences:
    """Pads a list of variable-length sequences to a uniform [B, T_max].

    Returns tensors ready for a Transformer encoder.
    """
    lengths = torch.tensor([b['X'].shape[0] for b in batch], dtype=torch.long)
    T = int(lengths.max().item())
    F = int(batch[0]['X'].shape[1])
    has_legal = batch[0]['legal'] is not None
    C = int(batch[0]['legal'].shape[1]) if has_legal else 0

    X = torch.full((len(batch), T, F), pad_value, dtype=torch.float32)
    y = torch.full((len(batch), T), -100, dtype=torch.long)  # -100 ignored index for CE
    legal = torch.zeros((len(batch), T, C), dtype=torch.float32) if has_legal else None
    key_padding_mask = torch.ones((len(batch), T), dtype=torch.bool)  # True = PAD

    for i, b in enumerate(batch):
        t = b['X'].shape[0]
        X[i, :t] = b['X']
        y[i, :t] = b['y']
        key_padding_mask[i, :t] = False
        if has_legal:
            legal[i, :t] = b['legal']

    return PackedSequences(X=X, y=y, key_padding_mask=key_padding_mask, legal=legal, lengths=lengths)


# ──────────────────────────────────────────────────────────────────────────────
# Splitting utilities
# ──────────────────────────────────────────────────────────────────────────────

def split_by_hand(sequences: Sequence[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Random split at hand granularity."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sequences))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, s in enumerate(sequences):
        (val if i in val_idx else train).append(s)
    return train, val


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader builders
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(train_seq: Sequence[Dict[str, Any]], val_seq: Sequence[Dict[str, Any]], batch_size: int = 16, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Optional[List[str]]]:
    dtrain = SequenceDataset(train_seq)
    dval = SequenceDataset(val_seq)
    feature_keys = dtrain.feature_keys or dval.feature_keys

    train_loader = DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_and_collate, pin_memory=True)
    val_loader = DataLoader(dval, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_and_collate, pin_memory=True)
    return train_loader, val_loader, feature_keys


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_device(p: PackedSequences, device: torch.device) -> PackedSequences:
    return PackedSequences(
        X=p.X.to(device),
        y=p.y.to(device),
        key_padding_mask=p.key_padding_mask.to(device),
        legal=p.legal.to(device) if p.legal is not None else None,
        lengths=p.lengths.to(device),
    )


def flatten_feature_keys(sequences: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    for s in sequences:
        if 'feature_keys' in s and s['feature_keys']:
            return list(s['feature_keys'])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Example adapter for legacy flat dict (optional)
# ──────────────────────────────────────────────────────────────────────────────

def from_legacy_flat_dict(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a legacy dict with keys like 'X','y','mask','legal' (each a list) into
    a list of per-hand sequence dicts. Use this if your extractor returned a single
    big object instead of a list of hands.
    """
    X_list: List[np.ndarray] = data['X']
    y_list: List[np.ndarray] = data['y']
    mask_list: List[np.ndarray] = data['mask']
    legal_list: Optional[List[np.ndarray]] = data.get('legal')
    feature_keys: Optional[List[str]] = data.get('feature_keys')

    sequences: List[Dict[str, Any]] = []
    for i, (X, y, m) in enumerate(zip(X_list, y_list, mask_list)):
        seq = {
            'hand_id': str(data.get('hand_ids', [i]*len(X_list))[i]) if 'hand_ids' in data else f"hand_{i}",
            'X': np.asarray(X, dtype=np.float32),
            'y': np.asarray(y, dtype=np.int64),
            'mask': np.asarray(m, dtype=np.float32),
            'legal': np.asarray(legal_list[i], dtype=np.float32) if legal_list is not None else None,
            'feature_keys': feature_keys,
        }
        sequences.append(seq)
    return sequences


# ──────────────────────────────────────────────────────────────────────────────
# Minimal I/O (torch.save/torch.load left to caller)
# ──────────────────────────────────────────────────────────────────────────────

def pack_for_save(sequences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    feature_keys = flatten_feature_keys(sequences)
    return {
        'sequences': sequences,
        'feature_keys': feature_keys,
    }


def unpack_from_load(obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
    return obj['sequences'], obj.get('feature_keys')


import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,E]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# Configs
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PokerTransformerConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # categorical embedding (optional)
    categorical_idx: Optional[List[int]] = None
    categorical_cardinalities: Optional[List[int]] = None
    embed_dim: int = 32


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    device: str = "cpu"  # 'cuda' if available


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class PokerTransformerClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, cfg: PokerTransformerConfig):
        super().__init__()
        self.cfg = cfg
        cat_idx = cfg.categorical_idx or []
        self.cat_idx = torch.tensor(cat_idx, dtype=torch.long) if len(cat_idx) else None
        self.cont_idx = (
            [i for i in range(n_features) if i not in cat_idx]
            if len(cat_idx)
            else list(range(n_features))
        )

        # Categorical embeddings
        self.embeds = None
        embed_total = 0
        if len(cat_idx):
            cards = cfg.categorical_cardinalities
            assert cards and len(cards) == len(cat_idx), "Provide cardinalities for categorical_idx"
            self.embeds = nn.ModuleList([nn.Embedding(c, cfg.embed_dim) for c in cards])
            embed_total = len(cat_idx) * cfg.embed_dim

        # Continuous projection
        self.cont_in = None
        cont_dim = len(self.cont_idx)
        if cont_dim > 0:
            self.cont_in = nn.Linear(cont_dim, max(1, cfg.d_model - embed_total))

        # Fuse to d_model
        fused_in = (cfg.d_model - embed_total if self.cont_in else 0) + embed_total
        self.fuse = (
            nn.Identity() if fused_in == cfg.d_model else nn.Linear(fused_in, cfg.d_model)
        )

        # Positional encoding + TransformerEncoder
        self.posenc = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, n_classes),
        )

    def _split_features(self, X: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        # X: [B,T,F]
        if self.cat_idx is None or len(self.cat_idx) == 0:
            cont = X
            cats = None
        else:
            cats = [X[..., i].long() for i in self.cat_idx.tolist()]
            cont = X[..., self.cont_idx] if len(self.cont_idx) else None
        return cont, cats

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Prepare embeddings
        cont, cats = self._split_features(X)
        parts: List[torch.Tensor] = []
        if cats is not None and self.embeds is not None:
            embs = [emb(cat) for emb, cat in zip(self.embeds, cats)]  # [B,T,embed]
            parts.append(torch.cat(embs, dim=-1))
        if cont is not None and self.cont_in is not None:
            parts.append(self.cont_in(cont))
        if len(parts) == 0:
            raise RuntimeError("No input features provided")
        h = torch.cat(parts, dim=-1)
        h = self.fuse(h)
        h = self.posenc(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # [B,T,d]
        logits = self.head(h)  # [B,T,C]
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Training / Eval
# ──────────────────────────────────────────────────────────────────────────────

def masked_xent_with_legal(logits: torch.Tensor, targets: torch.Tensor, key_padding_mask: torch.Tensor, legal: Optional[torch.Tensor], label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy ignoring PAD positions and optionally masking illegal actions.

    Args:
        logits: [B,T,C]
        targets: [B,T] with -100 at PAD
        key_padding_mask: [B,T] bool, True where PAD
        legal: [B,T,C] float in {0,1} or None
    """
    if legal is not None:
        logits = logits.masked_fill(legal < 0.5, -1e9)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )
    return loss


def train_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, normalizer: Optional[Normalizer], cfg: TrainConfig, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
    model.train()
    total = 0.0
    n = 0
    device = torch.device(cfg.device)
    for batch in loader:
        batch = batch if isinstance(batch, PackedSequences) else PackedSequences(**batch)  # type: ignore
        batch = batch if isinstance(batch, PackedSequences) else batch  # for mypy
        X = batch.X.to(device)
        y = batch.y.to(device)
        kpm = batch.key_padding_mask.to(device)
        legal = batch.legal.to(device) if batch.legal is not None else None

        if normalizer is not None:
            X = normalizer.transform(X)

        optim.zero_grad(set_to_none=True)
        logits = model(X, key_padding_mask=kpm)
        loss = masked_xent_with_legal(logits, y, kpm, legal, cfg.label_smoothing)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(1, n)


def evaluate(model: nn.Module, loader: DataLoader, normalizer: Optional[Normalizer], cfg: TrainConfig) -> Tuple[float, float]:
    model.eval()
    device = torch.device(cfg.device)
    total = 0.0
    n = 0
    correct = 0
    denom = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch if isinstance(batch, PackedSequences) else PackedSequences(**batch)  # type: ignore
            X = batch.X.to(device)
            y = batch.y.to(device)
            kpm = batch.key_padding_mask.to(device)
            legal = batch.legal.to(device) if batch.legal is not None else None

            if normalizer is not None:
                X = normalizer.transform(X)

            logits = model(X, key_padding_mask=kpm)
            loss = masked_xent_with_legal(logits, y, kpm, legal, 0.0)

            total += loss.item() * X.size(0)
            n += X.size(0)

            # Accuracy on non-pad steps
            preds = logits.argmax(dim=-1)
            mask = ~kpm
            correct += ((preds == y) & mask).sum().item()
            denom += mask.sum().item()
    return total / max(1, n), (correct / max(1, denom))


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing & Inference
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model: nn.Module, normalizer: Optional[Normalizer], model_cfg: PokerTransformerConfig, train_cfg: TrainConfig, n_features: int, n_classes: int, feature_keys: Optional[List[str]] = None) -> None:
    obj = {
        'model_state': model.state_dict(),
        'normalizer': None if normalizer is None else normalizer.state_dict(),
        'model_cfg': asdict(model_cfg),
        'train_cfg': asdict(train_cfg),
        'n_features': n_features,
        'n_classes': n_classes,
        'feature_keys': feature_keys,
    }
    torch.save(obj, path)


def load_checkpoint(path: str, device: str = 'cpu') -> Tuple[nn.Module, Optional[Normalizer], PokerTransformerConfig, TrainConfig, int, int, Optional[List[str]]]:
    ckpt = torch.load(path, map_location=device)
    model_cfg = PokerTransformerConfig(**ckpt['model_cfg'])
    train_cfg = TrainConfig(**ckpt['train_cfg'])
    n_features = ckpt['n_features']
    n_classes = ckpt['n_classes']
    model = PokerTransformerClassifier(n_features, n_classes, model_cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    normalizer = None
    if ckpt.get('normalizer') is not None:
        normalizer = Normalizer()
        normalizer.load_state_dict(ckpt['normalizer'])

    feature_keys = ckpt.get('feature_keys')
    return model, normalizer, model_cfg, train_cfg, n_features, n_classes, feature_keys


def infer_single_step(model: nn.Module, normalizer: Optional[Normalizer], x_seq: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return probabilities for the last timestep in x_seq.

    Args:
        x_seq: [1, T, F] or [T, F]
        key_padding_mask: [1, T] or [T] bool (True where PAD)
    Returns:
        probs: [C]
    """
    model.eval()
    if x_seq.dim() == 2:
        x_seq = x_seq.unsqueeze(0)
    if key_padding_mask is None:
        kpm = torch.zeros((x_seq.size(0), x_seq.size(1)), dtype=torch.bool, device=x_seq.device)
    else:
        kpm = key_padding_mask
        if kpm.dim() == 1:
            kpm = kpm.unsqueeze(0)
        kpm = kpm.to(x_seq.device)

    with torch.no_grad():
        X = x_seq
        if normalizer is not None:
            X = normalizer.transform(X)
        logits = model(X, key_padding_mask=kpm)
        probs = logits[:, -1].softmax(dim=-1).squeeze(0)
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end train loop wrapper
# ──────────────────────────────────────────────────────────────────────────────

def train_loop(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, normalizer: Optional[Normalizer], cfg: TrainConfig) -> Dict[str, List[float]]:
    device = torch.device(cfg.device)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, normalizer, cfg)
        val_loss, val_acc = evaluate(model, val_loader, normalizer, cfg)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")

    return history


'''
# ──────────────────────────────────────────────────────────────────────────────
# CLI example (optional)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────────────
# Explicit-variable entrypoint (no argparse)
  # ──────────────────────────────────────────────────────────────────────────────
  import os
  from typing import List
  import torch

  # ======= 1) Define your inputs here =======
  #data_path      = "/path/to/your/data.pt"   # torch-saved object w/ {'sequences': [...] } OR a list of sequences
  save_path      = "poker_transformer.pt"

  d_model        = 256
  n_layers       = 4
  n_heads        = 8
  ff             = 1024
  dropout        = 0.10
  batch_size     = 16
  epochs         = 10
  lr             = 3e-4
  label_smoothing= 0.05
  device_str     = "cpu"   # e.g., "cuda" if available

  # ======= 2) Load sequences (no parser) =======
  #obj = torch.load(data_path, map_location="cpu")
  if isinstance(obj, dict) and "sequences" in obj:
      sequences = obj["sequences"]
      feature_keys = obj.get("feature_keys")
  elif isinstance(obj, list):
      sequences = obj
      feature_keys = None
  else:
      raise ValueError("data_path must contain either {'sequences': ...} or a list of sequences")

  # ======= 3) Make splits & dataloaders =======
  train_seq, val_seq = split_by_hand(sequences, val_ratio=0.10)
  train_loader, val_loader, detected_keys = make_dataloaders(
      train_seq, val_seq, batch_size=batch_size
  )

  # ======= 4) Fit normalizer on a few batches (RAM-friendly) =======
  normalizer = Normalizer()
  with torch.no_grad():
      seen: List[torch.Tensor] = []
      for i, pack in enumerate(train_loader):
          if i >= 8:  # sample up to ~8 batches
              break
          seen.append(pack.X)
      normalizer.fit(seen)

  # ======= 5) Build model/configs =======
  n_features = train_loader.dataset[0]["X"].shape[-1]  # type: ignore
  n_classes  = 3  # fold/call/raise

  model_cfg = PokerTransformerConfig(
      d_model=d_model,
      n_layers=n_layers,
      n_heads=n_heads,
      dim_feedforward=ff,
      dropout=dropout,
      categorical_idx=None,
      categorical_cardinalities=None,
  )

  train_cfg = TrainConfig(
      batch_size=batch_size,
      max_epochs=epochs,
      lr=lr,
      label_smoothing=label_smoothing,
      device=device_str,
  )

  model = PokerTransformerClassifier(n_features, n_classes, model_cfg)

  # ======= 6) Train =======
  hist = train_loop(model, train_loader, val_loader, normalizer, train_cfg)

  # ======= 7) Save checkpoint =======
  save_checkpoint(
      save_path,
      model,
      normalizer,
      model_cfg,
      train_cfg,
      n_features,
      n_classes,
      feature_keys=feature_keys or detected_keys,
  )

  print(f"Saved checkpoint to {os.path.abspath(save_path)}")
'''
