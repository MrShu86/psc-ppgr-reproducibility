from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import load_events
from .stage2_utils import ensure_dir


def _pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])


def _metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": _pearson(y_true, y_pred),
    }


def _build_covariates(meta: pd.DataFrame, config: dict) -> pd.DataFrame:
    numeric_cols = []
    for c in config.get("nutrition_columns", []):
        if c in meta.columns:
            numeric_cols.append(c)
    for c in ["baseline_glucose", "missing_ratio"]:
        if c in meta.columns:
            numeric_cols.append(c)
    for c in meta.columns:
        if c.endswith("_pre30_mean"):
            numeric_cols.append(c)
    for c in meta.columns:
        if c.startswith("subject_") and c != "subject_id":
            numeric_cols.append(c)

    numeric_cols = list(dict.fromkeys(numeric_cols))
    cat_cols = [c for c in ["device", "meal_type", "setting_baseline_bin", "setting_activity_bin"] if c in meta.columns]

    df = meta[numeric_cols + cat_cols].copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    return df


def run_sequence_baseline(
    config: dict,
    split_name: str,
    model_name: str = "gru",
    epochs: int = 120,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 20,
) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        raise RuntimeError("PyTorch is required. Please install torch before running sequence baselines.") from e

    out = Path(config["output_dir"])
    res_dir = ensure_dir(out / "results")
    meta, seq, rel_grid = load_events(out)

    split_path = out / "splits" / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split = pd.read_csv(split_path)
    split_map = dict(zip(split["event_id"], split["split"]))
    split_label = meta["event_id"].map(split_map).fillna("unused")

    target = config.get("main_target", "iauc_2h")
    y = pd.to_numeric(meta[target], errors="coerce").to_numpy(dtype=np.float32)

    train_mask = (split_label == "train").to_numpy() & np.isfinite(y)
    val_mask = (split_label == "val").to_numpy() & np.isfinite(y)
    test_mask = (split_label == "test").to_numpy() & np.isfinite(y)

    if train_mask.sum() < 20 or test_mask.sum() < 5:
        raise RuntimeError(f"Too few samples for {split_name}: train={train_mask.sum()}, test={test_mask.sum()}")

    if val_mask.sum() < 5:
        train_idx = np.where(train_mask)[0]
        rng = np.random.default_rng(int(config.get("random_seed", 42)))
        rng.shuffle(train_idx)
        n_val = max(5, int(0.15 * len(train_idx)))
        val_idx = train_idx[:n_val]
        new_train_idx = train_idx[n_val:]
        train_mask[:] = False
        val_mask[:] = False
        train_mask[new_train_idx] = True
        val_mask[val_idx] = True

    pre_mask = rel_grid <= 0
    X_seq = seq[:, pre_mask].astype(np.float32)

    X_cov_df = _build_covariates(meta, config)
    X_cov = X_cov_df.to_numpy(dtype=np.float32)

    train_cov = X_cov[train_mask]
    med = np.nanmedian(train_cov, axis=0)
    med[~np.isfinite(med)] = 0
    bad = np.where(~np.isfinite(X_cov))
    X_cov[bad] = np.take(med, bad[1])

    seq_scaler = StandardScaler()
    cov_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_seq_flat = X_seq.reshape(len(X_seq), -1)
    seq_scaler.fit(X_seq_flat[train_mask])
    X_seq_all = seq_scaler.transform(X_seq_flat).astype(np.float32).reshape(len(X_seq), X_seq.shape[1], 1)

    cov_scaler.fit(X_cov[train_mask])
    X_cov_all = cov_scaler.transform(X_cov).astype(np.float32)

    y_scaler.fit(y[train_mask].reshape(-1, 1))
    y_all_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    class MLPRegressor(nn.Module):
        def __init__(self, seq_len, cov_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(seq_len + cov_dim, 128), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(64, 1)
            )
        def forward(self, xs, xc):
            return self.net(torch.cat([xs.squeeze(-1), xc], dim=1)).squeeze(-1)

    class GRURegressor(nn.Module):
        def __init__(self, seq_len, cov_dim):
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(64 + cov_dim, 96), nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, 1)
            )
        def forward(self, xs, xc):
            _, h = self.gru(xs)
            return self.head(torch.cat([h[-1], xc], dim=1)).squeeze(-1)

    class TCNRegressor(nn.Module):
        def __init__(self, seq_len, cov_dim):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(64 + cov_dim, 96), nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, 1)
            )
        def forward(self, xs, xc):
            h = self.conv(xs.transpose(1, 2)).squeeze(-1)
            return self.head(torch.cat([h, xc], dim=1)).squeeze(-1)

    class TransformerRegressor(nn.Module):
        def __init__(self, seq_len, cov_dim):
            super().__init__()
            d_model = 48
            self.embed = nn.Linear(1, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=96, dropout=0.1, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(
                nn.Linear(d_model + cov_dim, 96), nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, 1)
            )
        def forward(self, xs, xc):
            h = self.embed(xs) + self.pos
            h = self.encoder(h).mean(dim=1)
            return self.head(torch.cat([h, xc], dim=1)).squeeze(-1)

    seq_len = X_seq_all.shape[1]
    cov_dim = X_cov_all.shape[1]
    name = model_name.lower()

    if name == "mlp":
        model = MLPRegressor(seq_len, cov_dim)
    elif name == "gru":
        model = GRURegressor(seq_len, cov_dim)
    elif name == "tcn":
        model = TCNRegressor(seq_len, cov_dim)
    elif name == "transformer":
        model = TransformerRegressor(seq_len, cov_dim)
    else:
        raise ValueError("model_name must be one of: mlp, gru, tcn, transformer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def make_loader(mask, shuffle):
        idx = np.where(mask)[0]
        ds = TensorDataset(
            torch.tensor(X_seq_all[idx], dtype=torch.float32),
            torch.tensor(X_cov_all[idx], dtype=torch.float32),
            torch.tensor(y_all_scaled[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_mask, True)
    val_loader = make_loader(val_mask, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, cb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
                val_losses.append(float(loss_fn(model(xb, cb), yb).detach().cpu()))
        val_loss = float(np.mean(val_losses))

        if epoch == 1 or epoch % 10 == 0:
            print(f"[{model_name}][{split_name}] epoch={epoch} train={np.mean(train_losses):.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EARLY STOP] epoch={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    idx_test = np.where(test_mask)[0]
    preds_scaled = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx_test), batch_size):
            idx = idx_test[start:start + batch_size]
            xb = torch.tensor(X_seq_all[idx], dtype=torch.float32).to(device)
            cb = torch.tensor(X_cov_all[idx], dtype=torch.float32).to(device)
            preds_scaled.append(model(xb, cb).detach().cpu().numpy())

    pred = y_scaler.inverse_transform(np.concatenate(preds_scaled).reshape(-1, 1)).ravel()
    y_true = y[idx_test]
    m = _metrics(y_true, pred)

    row = {
        "split_file": split_name,
        "model": model_name,
        "target": target,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "seq_len": int(seq_len),
        "cov_dim": int(cov_dim),
        **m,
    }

    result = pd.DataFrame([row])
    out_path = res_dir / f"sequence_baseline_{model_name}_{Path(split_name).stem}_{target}.csv"
    result.to_csv(out_path, index=False)

    print(f"[RESULT] {row}")
    print(f"[OK] Saved {out_path}")
