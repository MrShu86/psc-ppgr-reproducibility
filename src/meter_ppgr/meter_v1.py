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


def _build_covariates(meta: pd.DataFrame, config: dict, include_device: bool = False) -> pd.DataFrame:
    """
    Prediction-safe covariates.
    By default, device is NOT included as a predictor.
    In METER-v1, device is used only as a domain-adversarial label.
    """
    numeric_cols = []

    # Meal nutrition.
    for c in config.get("nutrition_columns", []):
        if c in meta.columns:
            numeric_cols.append(c)

    # Prediction-safe baseline features.
    for c in ["baseline_glucose", "missing_ratio"]:
        if c in meta.columns:
            numeric_cols.append(c)

    # Prediction-safe pre-meal activity only.
    for c in meta.columns:
        if c.endswith("_pre30_mean"):
            numeric_cols.append(c)

    # Subject-level metadata.
    for c in meta.columns:
        if c.startswith("subject_") and c != "subject_id":
            numeric_cols.append(c)

    numeric_cols = list(dict.fromkeys(numeric_cols))

    categorical_cols = []
    for c in ["meal_type", "setting_baseline_bin", "setting_activity_bin"]:
        if c in meta.columns:
            categorical_cols.append(c)

    if include_device and "device" in meta.columns:
        categorical_cols.append("device")

    df = meta[numeric_cols + categorical_cols].copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)
    return df


def _build_subject_context_covariates(meta: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Covariates for subject/context-adaptive bias head.
    Does not include device.
    """
    cols = []
    for c in meta.columns:
        if c.startswith("subject_") and c != "subject_id":
            cols.append(c)
    for c in ["baseline_glucose"]:
        if c in meta.columns:
            cols.append(c)

    cat_cols = []
    for c in ["meal_type", "setting_baseline_bin", "setting_activity_bin"]:
        if c in meta.columns:
            cat_cols.append(c)

    if not cols and not cat_cols:
        return pd.DataFrame({"const": np.ones(len(meta))})

    df = meta[cols + cat_cols].copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    return df


def _make_device_labels(meta: pd.DataFrame):
    devices = sorted(meta["device"].dropna().unique().tolist())
    lookup = {d: i for i, d in enumerate(devices)}
    labels = meta["device"].map(lookup).fillna(-1).astype(int).to_numpy()
    return labels, lookup


def run_meter_v1(
    config: dict,
    split_name: str,
    encoder: str = "tcn",
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 25,
    lambda_device: float = 0.10,
    use_device_adv: bool = True,
    use_adapter: bool = True,
    include_device_as_predictor: bool = False,
) -> None:
    try:
        import torch
        from torch import nn
        from torch.autograd import Function
        from torch.utils.data import TensorDataset, DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required. Please install torch first.") from e

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
        raise RuntimeError(f"Too few samples: train={train_mask.sum()}, test={test_mask.sum()}")

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

    # Sequence: pre-meal CGM only.
    pre_mask = rel_grid <= 0
    X_seq = seq[:, pre_mask].astype(np.float32)

    # Predictor covariates, with device excluded by default.
    X_cov_df = _build_covariates(meta, config, include_device=include_device_as_predictor)
    X_adapt_df = _build_subject_context_covariates(meta, config)

    X_cov = X_cov_df.to_numpy(dtype=np.float32)
    X_adapt = X_adapt_df.to_numpy(dtype=np.float32)

    # Impute and scale using train only.
    def impute_with_train_median(X):
        X = X.copy()
        med = np.nanmedian(X[train_mask], axis=0)
        med[~np.isfinite(med)] = 0
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(med, bad[1])
        return X

    X_cov = impute_with_train_median(X_cov)
    X_adapt = impute_with_train_median(X_adapt)

    seq_scaler = StandardScaler()
    cov_scaler = StandardScaler()
    adapt_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_seq_flat = X_seq.reshape(len(X_seq), -1)
    seq_scaler.fit(X_seq_flat[train_mask])
    X_seq_all = seq_scaler.transform(X_seq_flat).astype(np.float32).reshape(len(X_seq), X_seq.shape[1], 1)

    cov_scaler.fit(X_cov[train_mask])
    X_cov_all = cov_scaler.transform(X_cov).astype(np.float32)

    adapt_scaler.fit(X_adapt[train_mask])
    X_adapt_all = adapt_scaler.transform(X_adapt).astype(np.float32)

    y_scaler.fit(y[train_mask].reshape(-1, 1))
    y_all_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    device_labels, device_lookup = _make_device_labels(meta)
    train_device_classes = sorted(set(device_labels[train_mask].tolist()))
    adv_enabled = use_device_adv and len(train_device_classes) >= 2 and lambda_device > 0

    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = lambd
            return x.view_as(x)
        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambd * grad_output, None

    def grad_reverse(x, lambd=1.0):
        return GradReverse.apply(x, lambd)

    class TCNEncoder(nn.Module):
        def __init__(self, seq_len, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, hidden, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
        def forward(self, x):
            return self.net(x.transpose(1, 2)).squeeze(-1)

    class GRUEncoder(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        def forward(self, x):
            _, h = self.gru(x)
            return h[-1]

    class METERv1(nn.Module):
        def __init__(self, seq_len, cov_dim, adapt_dim, encoder_name, n_devices):
            super().__init__()
            hidden = 64
            if encoder_name == "gru":
                self.seq_encoder = GRUEncoder(hidden=hidden)
            elif encoder_name == "tcn":
                self.seq_encoder = TCNEncoder(seq_len=seq_len, hidden=hidden)
            else:
                raise ValueError("encoder must be 'gru' or 'tcn'.")

            self.cov_encoder = nn.Sequential(
                nn.Linear(cov_dim, 96), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(96, 64), nn.ReLU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(128, 64), nn.ReLU(),
            )

            self.pred_head = nn.Linear(64, 1)

            self.adapter = nn.Sequential(
                nn.Linear(adapt_dim, 64), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(64, 1),
            )

            self.domain_disc = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, n_devices),
            )

        def forward(self, xs, xc, xa, grl_lambda=1.0, use_adapter=True):
            hs = self.seq_encoder(xs)
            hc = self.cov_encoder(xc)
            z = self.fusion(torch.cat([hs, hc], dim=1))
            y_base = self.pred_head(z).squeeze(-1)
            if use_adapter:
                y_adapt = self.adapter(xa).squeeze(-1)
                y_hat = y_base + y_adapt
            else:
                y_hat = y_base
            d_logits = self.domain_disc(grad_reverse(z, grl_lambda))
            return y_hat, d_logits, z

    seq_len = X_seq_all.shape[1]
    cov_dim = X_cov_all.shape[1]
    adapt_dim = X_adapt_all.shape[1]
    n_devices = len(device_lookup)

    model = METERv1(seq_len, cov_dim, adapt_dim, encoder, n_devices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def make_loader(mask, shuffle):
        idx = np.where(mask)[0]
        ds = TensorDataset(
            torch.tensor(X_seq_all[idx], dtype=torch.float32),
            torch.tensor(X_cov_all[idx], dtype=torch.float32),
            torch.tensor(X_adapt_all[idx], dtype=torch.float32),
            torch.tensor(y_all_scaled[idx], dtype=torch.float32),
            torch.tensor(device_labels[idx], dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_mask, True)
    val_loader = make_loader(val_mask, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    reg_loss_fn = nn.HuberLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pred_losses = []
        dev_losses = []
        for xb, cb, ab, yb, db in train_loader:
            xb, cb, ab, yb, db = xb.to(device), cb.to(device), ab.to(device), yb.to(device), db.to(device)
            optimizer.zero_grad()
            y_hat, d_logits, _ = model(xb, cb, ab, grl_lambda=1.0, use_adapter=use_adapter)
            loss_pred = reg_loss_fn(y_hat, yb)
            loss = loss_pred

            if adv_enabled:
                loss_dev = ce_loss_fn(d_logits, db)
                loss = loss + lambda_device * loss_dev
                dev_losses.append(float(loss_dev.detach().cpu()))
            else:
                loss_dev = None

            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred_losses.append(float(loss_pred.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, cb, ab, yb, db in val_loader:
                xb, cb, ab, yb = xb.to(device), cb.to(device), ab.to(device), yb.to(device)
                y_hat, _, _ = model(xb, cb, ab, grl_lambda=0.0, use_adapter=use_adapter)
                val_losses.append(float(reg_loss_fn(y_hat, yb).detach().cpu()))
        val_loss = float(np.mean(val_losses))

        if epoch == 1 or epoch % 10 == 0:
            msg = f"[METER-v1][{encoder}][{split_name}] epoch={epoch} train={np.mean(losses):.4f} pred={np.mean(pred_losses):.4f} val={val_loss:.4f}"
            if dev_losses:
                msg += f" dev={np.mean(dev_losses):.4f}"
            print(msg)

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
    model.eval()
    preds_scaled = []
    zs = []
    with torch.no_grad():
        for start in range(0, len(idx_test), batch_size):
            idx = idx_test[start:start + batch_size]
            xb = torch.tensor(X_seq_all[idx], dtype=torch.float32).to(device)
            cb = torch.tensor(X_cov_all[idx], dtype=torch.float32).to(device)
            ab = torch.tensor(X_adapt_all[idx], dtype=torch.float32).to(device)
            y_hat, _, z = model(xb, cb, ab, grl_lambda=0.0, use_adapter=use_adapter)
            preds_scaled.append(y_hat.detach().cpu().numpy())
            zs.append(z.detach().cpu().numpy())

    pred = y_scaler.inverse_transform(np.concatenate(preds_scaled).reshape(-1, 1)).ravel()
    y_true = y[idx_test]
    m = _metrics(y_true, pred)

    method_name = "meter_v1"
    if not use_adapter:
        method_name += "_no_adapter"
    if not adv_enabled:
        method_name += "_no_device_adv"

    row = {
        "split_file": split_name,
        "model": method_name,
        "encoder": encoder,
        "target": target,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "seq_len": int(seq_len),
        "cov_dim": int(cov_dim),
        "adapt_dim": int(adapt_dim),
        "device_adv_requested": bool(use_device_adv),
        "device_adv_enabled": bool(adv_enabled),
        "lambda_device": float(lambda_device),
        "use_adapter": bool(use_adapter),
        "include_device_as_predictor": bool(include_device_as_predictor),
        **m,
    }

    res = pd.DataFrame([row])
    out_name = f"meter_v1_{encoder}_{Path(split_name).stem}_{target}.csv"
    out_path = res_dir / out_name
    res.to_csv(out_path, index=False)

    # Save test predictions for case studies.
    pred_df = meta.iloc[idx_test][["event_id", "paired_event_id", "subject_id", "device", "meal_type", "meal_timestamp"]].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = pred
    pred_df["abs_error"] = np.abs(y_true - pred)
    pred_path = res_dir / f"meter_v1_{encoder}_{Path(split_name).stem}_{target}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save representation for later UMAP/t-SNE.
    z_arr = np.concatenate(zs, axis=0)
    np.savez_compressed(
        res_dir / f"meter_v1_{encoder}_{Path(split_name).stem}_{target}_repr.npz",
        z=z_arr,
        event_id=pred_df["event_id"].astype(str).to_numpy(),
        subject_id=pred_df["subject_id"].astype(str).to_numpy(),
        device=pred_df["device"].astype(str).to_numpy(),
        meal_type=pred_df["meal_type"].astype(str).to_numpy(),
        y_true=y_true,
        y_pred=pred,
    )

    print(f"[RESULT] {row}")
    print(f"[OK] Saved result: {out_path}")
    print(f"[OK] Saved predictions: {pred_path}")
