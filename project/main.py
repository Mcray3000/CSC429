# 0) imports
import copy
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ecu_df = pd.read_parquet("data/cleaned_ecu.parquet")
wustl_df = pd.read_parquet("data/cleaned_wustl.parquet")

# 1) reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2) config
@dataclass
class FLConfig:
    # Data / splitting
    label_col: str = "label"
    test_size: float = 0.2
    n_clients: int = 8
    dirichlet_alpha: float = 0.3
    min_client_size: int = 64

    # Model / local training
    hidden_dim: int = 64
    local_epochs: int = 2
    batch_size: int = 128
    lr: float = 1e-3

    # Federated training
    rounds: int = 25
    clients_per_round: int = 4
    weighted_aggregation: bool = True

    # DP / clipping
    clip_norm: float = 1.0
    base_sigma: float = 0.02
    sigma_min: float = 0.005
    sigma_max: float = 0.10

    # adaptive sigma controls
    dp_mode: str = "adaptive"
    drift_beta: float = 0.75
    imbalance_gamma: float = 1.00
    size_lambda: float = 0.50

    # misc
    verbose: bool = True


# 3) tabular preprocessing
def load_polars_or_pandas(df_like) -> pd.DataFrame:
    if isinstance(df_like, pl.DataFrame):
        return df_like.to_pandas()
    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()
    raise TypeError("Expected a polars or pandas DataFrame.")


def prepare_binary_tabular_data(
    df_like,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    df = load_polars_or_pandas(df_like)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    y = df[label_col].astype(int).to_numpy()
    X_df = df.drop(columns=[label_col]).copy()

    for c in X_df.columns:
        if pd.api.types.is_bool_dtype(X_df[c]):
            X_df[c] = X_df[c].astype(int)

    train_df, test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = pd.get_dummies(train_df, dummy_na=True)
    test_df = pd.get_dummies(test_df, dummy_na=True)
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    feature_names = list(train_df.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.astype(np.float32))
    X_test = scaler.transform(test_df.astype(np.float32))

    return X_train.astype(np.float32), X_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32), feature_names


# 4) dirichlet client splitting
def dirichlet_split_indices(
    y: np.ndarray,
    n_clients: int,
    alpha: float = 0.3,
    min_size: int = 64,
    seed: int = 42,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n_classes = len(np.unique(y))

    while True:
        client_indices = [[] for _ in range(n_clients)]

        for cls in range(n_classes):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)

            proportions = rng.dirichlet(alpha=np.repeat(alpha, n_clients))
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            split_cls = np.split(cls_idx, cuts)

            for client_id, idx_chunk in enumerate(split_cls):
                client_indices[client_id].extend(idx_chunk.tolist())

        client_indices = [np.array(idx, dtype=int) for idx in client_indices]
        sizes = [len(idx) for idx in client_indices]

        if min(sizes) >= min_size:
            return client_indices


def summarize_clients(y: np.ndarray, client_indices: List[np.ndarray]) -> pd.DataFrame:
    rows = []
    global_pos = float(np.mean(y))
    for cid, idx in enumerate(client_indices):
        yc = y[idx]
        pos_rate = float(np.mean(yc)) if len(yc) > 0 else 0.0
        rows.append({
            "client_id": cid,
            "n_samples": len(idx),
            "pos_rate": pos_rate,
            "imbalance_vs_global": abs(pos_rate - global_pos),
        })
    return pd.DataFrame(rows)


# 5) simple binary classifier
class BinaryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# 6) state helpers
def get_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def set_model_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def state_sub(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] - b[k] for k in a.keys()}


def state_add(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] + b[k] for k in a.keys()}


def state_mul(a: Dict[str, torch.Tensor], scalar: float) -> Dict[str, torch.Tensor]:
    return {k: a[k] * scalar for k in a.keys()}


def state_l2_norm(state: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for v in state.values():
        total += float(torch.sum(v.float() ** 2))
    return math.sqrt(total)


def clip_state_by_l2(state: Dict[str, torch.Tensor], clip_norm: float) -> Tuple[Dict[str, torch.Tensor], float]:
    norm = state_l2_norm(state)
    if norm <= clip_norm or norm == 0.0:
        return {k: v.clone() for k, v in state.items()}, norm
    scale = clip_norm / norm
    return {k: v * scale for k, v in state.items()}, norm


def add_gaussian_noise_to_state(
    state: Dict[str, torch.Tensor],
    sigma: float,
    clip_norm: float,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    noisy = {}
    std = sigma * clip_norm
    for k, v in state.items():
        noise = torch.normal(
            mean=0.0,
            std=std,
            size=v.shape,
            device=device
        ).cpu()
        noisy[k] = v + noise
    return noisy


# 7) local training
def make_client_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    Xc = torch.tensor(X[indices], dtype=torch.float32)
    yc = torch.tensor(y[indices], dtype=torch.float32)
    ds = TensorDataset(Xc, yc)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def local_train(
    global_state: Dict[str, torch.Tensor],
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_idx: np.ndarray,
    cfg: FLConfig,
    in_dim: int,
) -> Dict[str, torch.Tensor]:
    model = BinaryMLP(in_dim=in_dim, hidden_dim=cfg.hidden_dim).to(DEVICE)
    set_model_state(model, global_state)

    model.train()
    loader = make_client_loader(X_train, y_train, client_idx, cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    local_state = get_model_state(model)
    delta = state_sub(local_state, global_state)
    return delta


# 8) adaptive sigma rule
def compute_adaptive_sigma(
    base_sigma: float,
    clip_norm: float,
    raw_delta_norm: float,
    client_n: int,
    mean_client_n: float,
    client_pos_rate: float,
    global_pos_rate: float,
    cfg: FLConfig,
) -> float:
    drift_ratio = raw_delta_norm / max(clip_norm, 1e-12)

    imbalance = abs(client_pos_rate - global_pos_rate)

    size_term = math.sqrt(max(mean_client_n, 1.0) / max(client_n, 1.0))

    sigma = base_sigma * (
        1.0
        + cfg.drift_beta * drift_ratio
        + cfg.imbalance_gamma * imbalance
    ) * (1.0 + cfg.size_lambda * (size_term - 1.0))

    sigma = float(np.clip(sigma, cfg.sigma_min, cfg.sigma_max))
    return sigma


# 9) evaluation
@torch.no_grad()
def evaluate_model(
    state: Dict[str, torch.Tensor],
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: FLConfig,
) -> Dict[str, float]:
    model = BinaryMLP(in_dim=X_test.shape[1], hidden_dim=cfg.hidden_dim).to(DEVICE)
    set_model_state(model, state)
    model.eval()

    Xb = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    logits = model(Xb).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_test, probs)
    except ValueError:
        metrics["auc"] = np.nan

    return metrics


# 10) federated training loop
def federated_train_adaptive_dp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    client_indices: List[np.ndarray],
    cfg: FLConfig,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, pd.DataFrame]:
    in_dim = X_train.shape[1]
    global_model = BinaryMLP(in_dim=in_dim, hidden_dim=cfg.hidden_dim).to(DEVICE)
    global_state = get_model_state(global_model)

    client_stats = summarize_clients(y_train, client_indices)
    global_pos_rate = float(np.mean(y_train))
    mean_client_n = float(client_stats["n_samples"].mean())

    history_rows = []
    sigma_rows = []

    for rnd in range(1, cfg.rounds + 1):
        chosen = np.random.choice(
            len(client_indices),
            size=min(cfg.clients_per_round, len(client_indices)),
            replace=False
        )

        client_updates = []
        client_weights = []

        for cid in chosen:
            idx = client_indices[cid]
            client_n = len(idx)
            client_pos = float(np.mean(y_train[idx]))

            # 1) local training
            raw_delta = local_train(
                global_state=global_state,
                X_train=X_train,
                y_train=y_train,
                client_idx=idx,
                cfg=cfg,
                in_dim=in_dim,
            )

            # 2) clip update
            clipped_delta, raw_norm = clip_state_by_l2(raw_delta, cfg.clip_norm)

            # 3) choose sigma
            if cfg.dp_mode == "fixed":
                sigma_i = cfg.base_sigma
            elif cfg.dp_mode == "adaptive":
                sigma_i = compute_adaptive_sigma(
                    base_sigma=cfg.base_sigma,
                    clip_norm=cfg.clip_norm,
                    raw_delta_norm=raw_norm,
                    client_n=client_n,
                    mean_client_n=mean_client_n,
                    client_pos_rate=client_pos,
                    global_pos_rate=global_pos_rate,
                    cfg=cfg,
                )
            else:
                raise ValueError("cfg.dp_mode must be 'fixed' or 'adaptive'.")

            # 4) add client-side gaussian noise before upload
            noisy_delta = add_gaussian_noise_to_state(
                state=clipped_delta,
                sigma=sigma_i,
                clip_norm=cfg.clip_norm,
                device=DEVICE,
            )

            client_updates.append(noisy_delta)
            client_weights.append(client_n if cfg.weighted_aggregation else 1.0)

            sigma_rows.append({
                "round": rnd,
                "client_id": int(cid),
                "client_n": int(client_n),
                "client_pos_rate": float(client_pos),
                "raw_delta_norm": float(raw_norm),
                "sigma": float(sigma_i),
            })

        # 5) aggregate noisy deltas
        total_weight = float(np.sum(client_weights))
        agg_delta = None
        for upd, w in zip(client_updates, client_weights):
            scaled = state_mul(upd, float(w) / total_weight)
            agg_delta = scaled if agg_delta is None else state_add(agg_delta, scaled)

        global_state = state_add(global_state, agg_delta)

        # 6) evaluate
        metrics = evaluate_model(global_state, X_test, y_test, cfg)
        history_rows.append({
            "round": rnd,
            "mode": cfg.dp_mode,
            **metrics
        })

        if cfg.verbose:
            print(
                f"[Round {rnd:02d}] "
                f"acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['auc']:.4f}"
            )

    history_df = pd.DataFrame(history_rows)
    sigma_df = pd.DataFrame(sigma_rows)
    return global_state, history_df, sigma_df


# 11) convenience wrapper (just a wrapper... WIP)
def run_experiment(
    raw_df_like,
    cfg: FLConfig,
    dataset_name: str = "dataset",
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame, pd.DataFrame, List[str], List[np.ndarray]]:
    X_train, X_test, y_train, y_test, feature_names = prepare_binary_tabular_data(
        raw_df_like,
        label_col=cfg.label_col,
        test_size=cfg.test_size,
        random_state=42,
    )

    client_indices = dirichlet_split_indices(
        y=y_train.astype(int),
        n_clients=cfg.n_clients,
        alpha=cfg.dirichlet_alpha,
        min_size=cfg.min_client_size,
        seed=42,
    )

    print(f"\n[{dataset_name}]")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(summarize_clients(y_train, client_indices).describe(include="all"))

    state, history_df, sigma_df = federated_train_adaptive_dp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        client_indices=client_indices,
        cfg=cfg,
    )

    return state, history_df, sigma_df, feature_names, client_indices


# 12) example usage on ecu
cfg_fixed = FLConfig(
    n_clients=8,
    dirichlet_alpha=0.3,
    rounds=20,
    clients_per_round=4,
    local_epochs=2,
    batch_size=128,
    lr=1e-3,
    clip_norm=1.0,
    base_sigma=0.02,
    dp_mode="fixed",
    verbose=True,
)

cfg_adapt = FLConfig(
    n_clients=8,
    dirichlet_alpha=0.3,
    rounds=20,
    clients_per_round=4,
    local_epochs=2,
    batch_size=128,
    lr=1e-3,
    clip_norm=1.0,
    base_sigma=0.02,
    dp_mode="adaptive",
    drift_beta=0.75,
    imbalance_gamma=1.00,
    size_lambda=0.50,
    sigma_min=0.005,
    sigma_max=0.10,
    verbose=True,
)

ecu_fixed_state, ecu_fixed_hist, ecu_fixed_sigma, ecu_features, ecu_clients = run_experiment(
    ecu_df, cfg_fixed, dataset_name="ECU / Fixed DP"
)

ecu_adapt_state, ecu_adapt_hist, ecu_adapt_sigma, _, _ = run_experiment(
    ecu_df, cfg_adapt, dataset_name="ECU / Adaptive DP"
)

wustl_fixed_state, wustl_fixed_hist, wustl_fixed_sigma, wustl_features, wustl_clients = run_experiment(
    wustl_df, cfg_fixed, dataset_name="WUSTL / Fixed DP"
)

wustl_adapt_state, wustl_adapt_hist, wustl_adapt_sigma, _, _ = run_experiment(
    wustl_df, cfg_adapt, dataset_name="WUSTL / Adaptive DP"
)