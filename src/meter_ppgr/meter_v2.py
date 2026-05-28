from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Pearson": np.nan}
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": _pearson(y_true, y_pred),
    }


def _build_predictor_covariates(meta: pd.DataFrame, config: dict, include_device: bool = False) -> pd.DataFrame:
    """
    Prediction-safe event covariates. Device is excluded by default.
    """
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


def _build_calibration_covariates(meta: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Domain/setting calibration features.
    No future information. No device by default.
    """
    numeric_cols = []
    for c in ["baseline_glucose"]:
        if c in meta.columns:
            numeric_cols.append(c)
    for c in meta.columns:
        if c.startswith("subject_") and c != "subject_id":
            numeric_cols.append(c)

    categorical_cols = []
    for c in ["meal_type", "setting_baseline_bin", "setting_activity_bin"]:
        if c in meta.columns:
            categorical_cols.append(c)

    if not numeric_cols and not categorical_cols:
        return pd.DataFrame({"const": np.ones(len(meta))})

    df = meta[numeric_cols + categorical_cols].copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)
    return df


def _make_device_labels(meta: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    devices = sorted(meta["device"].dropna().unique().tolist())
    lookup = {d: i for i, d in enumerate(devices)}
    labels = meta["device"].map(lookup).fillna(-1).astype(int).to_numpy()
    return labels, lookup


def _make_pair_index(meta: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    """
    Creates index pairs for paired-device consistency among training samples.
    Each pair contains two device-level events from the same paired_event_id.
    """
    pairs = []
    train_idx_set = set(np.where(train_mask)[0].tolist())
    grouped = meta.reset_index().groupby("paired_event_id")["index"].apply(list)
    for _, idxs in grouped.items():
        idxs = [int(i) for i in idxs if int(i) in train_idx_set]
        if len(idxs) >= 2:
            # Use the first two devices if more than two somehow exist.
            pairs.append((idxs[0], idxs[1]))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _support_calibration_summary(
    pred_df: pd.DataFrame,
    shots=(0, 1, 3, 5, 10),
    split_name: str = "",
    model_name: str = "meter_v2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates support-set personalization on the test set.

    For each subject, sort paired events by meal timestamp.
    First K paired events are support; the rest are query.
    Residual calibration:
        y_pred_query + mean(y_support - y_pred_support)
    Affine calibration:
        a * y_pred_query + b fitted from support when K is sufficient.
    """
    df = pred_df.copy()
    df["meal_timestamp"] = pd.to_datetime(df["meal_timestamp"], errors="coerce")
    detail_rows = []

    for subject_id, sub_all in df.groupby("subject_id"):
        pair_order = (
            sub_all.groupby("paired_event_id")["meal_timestamp"]
            .min()
            .sort_values()
            .index
            .tolist()
        )
        if len(pair_order) < 3:
            continue

        for k in shots:
            if len(pair_order) <= k + 1:
                continue

            support_pairs = set(pair_order[:k])
            query_pairs = set(pair_order[k:])

            support = sub_all[sub_all["paired_event_id"].isin(support_pairs)]
            query = sub_all[sub_all["paired_event_id"].isin(query_pairs)]
            if len(query) < 3:
                continue

            yq = query["y_true"].to_numpy(dtype=float)
            pq = query["y_pred"].to_numpy(dtype=float)

            # Global/no update.
            m = _metrics(yq, pq)
            detail_rows.append({
                "split_file": split_name,
                "model": model_name,
                "fold_subject": subject_id,
                "shot": k,
                "personalization": "global_0shot" if k == 0 else "global_no_update",
                "n_support": len(support),
                "n_query": len(query),
                **m,
            })

            if k > 0 and len(support) >= 1:
                ps = support["y_pred"].to_numpy(dtype=float)
                ys = support["y_true"].to_numpy(dtype=float)

                # Residual calibration.
                delta = float(np.nanmean(ys - ps))
                pq_res = pq + delta
                m = _metrics(yq, pq_res)
                detail_rows.append({
                    "split_file": split_name,
                    "model": model_name,
                    "fold_subject": subject_id,
                    "shot": k,
                    "personalization": "support_residual_calibration",
                    "n_support": len(support),
                    "n_query": len(query),
                    **m,
                })

                # Affine calibration.
                if len(support) >= 3 and np.nanstd(ps) > 1e-8:
                    a, b = np.polyfit(ps, ys, deg=1)
                    pq_aff = a * pq + b
                else:
                    pq_aff = pq_res
                m = _metrics(yq, pq_aff)
                detail_rows.append({
                    "split_file": split_name,
                    "model": model_name,
                    "fold_subject": subject_id,
                    "shot": k,
                    "personalization": "support_affine_calibration",
                    "n_support": len(support),
                    "n_query": len(query),
                    **m,
                })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, detail

    summary_rows = []
    for keys, sub in detail.groupby(["split_file", "model", "personalization", "shot"]):
        split_file, model, personalization, shot = keys
        row = {
            "split_file": split_file,
            "model": model,
            "personalization": personalization,
            "shot": shot,
            "n_folds": sub["fold_subject"].nunique(),
            "n_query_total": sub["n_query"].sum(),
        }
        for metric in ["MAE", "RMSE", "R2", "Pearson"]:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_median"] = sub[metric].median()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["shot", "personalization"])
    return detail, summary


def run_meter_v2(
    config: dict,
    split_name: str,
    encoder: str = "tcn",
    epochs: int = 180,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 30,
    lambda_device: float = 0.10,
    lambda_pair_z: float = 0.05,
    lambda_pair_pred: float = 0.05,
    use_device_adv: bool = True,
    use_pair_consistency: bool = True,
    use_calibration_head: bool = True,
    include_device_as_predictor: bool = False,
    shots=(0, 1, 3, 5, 10),
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

    X_cov_df = _build_predictor_covariates(meta, config, include_device=include_device_as_predictor)
    X_cal_df = _build_calibration_covariates(meta, config)

    X_cov = X_cov_df.to_numpy(dtype=np.float32)
    X_cal = X_cal_df.to_numpy(dtype=np.float32)

    def impute_train_median(X):
        X = X.copy()
        med = np.nanmedian(X[train_mask], axis=0)
        med[~np.isfinite(med)] = 0
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(med, bad[1])
        return X

    X_cov = impute_train_median(X_cov)
    X_cal = impute_train_median(X_cal)

    seq_scaler = StandardScaler()
    cov_scaler = StandardScaler()
    cal_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_seq_flat = X_seq.reshape(len(X_seq), -1)
    seq_scaler.fit(X_seq_flat[train_mask])
    X_seq_all = seq_scaler.transform(X_seq_flat).astype(np.float32).reshape(len(X_seq), X_seq.shape[1], 1)

    cov_scaler.fit(X_cov[train_mask])
    X_cov_all = cov_scaler.transform(X_cov).astype(np.float32)

    cal_scaler.fit(X_cal[train_mask])
    X_cal_all = cal_scaler.transform(X_cal).astype(np.float32)

    y_scaler.fit(y[train_mask].reshape(-1, 1))
    y_all_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    device_labels, device_lookup = _make_device_labels(meta)
    train_devices = sorted(set(device_labels[train_mask].tolist()))
    adv_enabled = use_device_adv and len(train_devices) >= 2 and lambda_device > 0

    pair_index = _make_pair_index(meta, train_mask)
    pair_enabled = use_pair_consistency and len(pair_index) > 0 and (lambda_pair_z > 0 or lambda_pair_pred > 0)

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
        def __init__(self, hidden=64):
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

    class METERv2(nn.Module):
        def __init__(self, seq_len, cov_dim, cal_dim, encoder_name, n_devices):
            super().__init__()
            hidden = 64
            if encoder_name == "tcn":
                self.seq_encoder = TCNEncoder(hidden=hidden)
            elif encoder_name == "gru":
                self.seq_encoder = GRUEncoder(hidden=hidden)
            else:
                raise ValueError("encoder must be tcn or gru")

            self.cov_encoder = nn.Sequential(
                nn.Linear(cov_dim, 96), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(96, 64), nn.ReLU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(128, 64), nn.ReLU(),
            )

            self.pred_head = nn.Linear(64, 1)

            self.calibration_head = nn.Sequential(
                nn.Linear(cal_dim, 64), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(64, 1),
            )

            self.domain_disc = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, n_devices),
            )

        def forward(self, xs, xc, xcal, grl_lambda=1.0, use_calibration=True):
            hs = self.seq_encoder(xs)
            hc = self.cov_encoder(xc)
            z = self.fusion(torch.cat([hs, hc], dim=1))
            y_base = self.pred_head(z).squeeze(-1)
            if use_calibration:
                y_cal = self.calibration_head(xcal).squeeze(-1)
                y_hat = y_base + y_cal
            else:
                y_hat = y_base
            d_logits = self.domain_disc(grad_reverse(z, grl_lambda))
            return y_hat, d_logits, z

    seq_len = X_seq_all.shape[1]
    cov_dim = X_cov_all.shape[1]
    cal_dim = X_cal_all.shape[1]
    n_devices = len(device_lookup)

    model = METERv2(seq_len, cov_dim, cal_dim, encoder, n_devices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def make_loader(mask, shuffle):
        idx = np.where(mask)[0]
        ds = TensorDataset(
            torch.tensor(X_seq_all[idx], dtype=torch.float32),
            torch.tensor(X_cov_all[idx], dtype=torch.float32),
            torch.tensor(X_cal_all[idx], dtype=torch.float32),
            torch.tensor(y_all_scaled[idx], dtype=torch.float32),
            torch.tensor(device_labels[idx], dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def make_pair_loader(pair_index, shuffle=True):
        if len(pair_index) == 0:
            return []
        ds = TensorDataset(
            torch.tensor(pair_index[:, 0], dtype=torch.long),
            torch.tensor(pair_index[:, 1], dtype=torch.long),
        )
        return DataLoader(ds, batch_size=max(16, batch_size // 2), shuffle=shuffle)

    train_loader = make_loader(train_mask, True)
    val_loader = make_loader(val_mask, False)
    pair_loader = make_pair_loader(pair_index, True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    reg_loss_fn = nn.HuberLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    # Full tensors on device for pair consistency mini-batches by original index.
    X_seq_tensor = torch.tensor(X_seq_all, dtype=torch.float32).to(device)
    X_cov_tensor = torch.tensor(X_cov_all, dtype=torch.float32).to(device)
    X_cal_tensor = torch.tensor(X_cal_all, dtype=torch.float32).to(device)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_losses = []
        pred_losses = []
        dev_losses = []
        pair_losses = []

        for xb, cb, calb, yb, db, ib in train_loader:
            xb, cb, calb, yb, db = xb.to(device), cb.to(device), calb.to(device), yb.to(device), db.to(device)
            optimizer.zero_grad()

            y_hat, d_logits, z = model(xb, cb, calb, grl_lambda=1.0, use_calibration=use_calibration_head)
            loss_pred = reg_loss_fn(y_hat, yb)
            loss = loss_pred

            if adv_enabled:
                loss_dev = ce_loss_fn(d_logits, db)
                loss = loss + lambda_device * loss_dev
                dev_losses.append(float(loss_dev.detach().cpu()))

            loss.backward()
            optimizer.step()

            total_losses.append(float(loss.detach().cpu()))
            pred_losses.append(float(loss_pred.detach().cpu()))

        # Pair consistency update.
        if pair_enabled:
            for ia, ib in pair_loader:
                ia = ia.to(device)
                ib = ib.to(device)
                optimizer.zero_grad()

                ya, _, za = model(
                    X_seq_tensor[ia], X_cov_tensor[ia], X_cal_tensor[ia],
                    grl_lambda=0.0, use_calibration=use_calibration_head
                )
                yb, _, zb = model(
                    X_seq_tensor[ib], X_cov_tensor[ib], X_cal_tensor[ib],
                    grl_lambda=0.0, use_calibration=use_calibration_head
                )

                loss_pair = 0
                if lambda_pair_z > 0:
                    loss_pair = loss_pair + lambda_pair_z * mse_loss_fn(za, zb)
                if lambda_pair_pred > 0:
                    loss_pair = loss_pair + lambda_pair_pred * mse_loss_fn(ya, yb)

                loss_pair.backward()
                optimizer.step()
                pair_losses.append(float(loss_pair.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, cb, calb, yb, db, ib in val_loader:
                xb, cb, calb, yb = xb.to(device), cb.to(device), calb.to(device), yb.to(device)
                y_hat, _, _ = model(xb, cb, calb, grl_lambda=0.0, use_calibration=use_calibration_head)
                val_losses.append(float(reg_loss_fn(y_hat, yb).detach().cpu()))
        val_loss = float(np.mean(val_losses))

        if epoch == 1 or epoch % 10 == 0:
            msg = (
                f"[METER-v2][{encoder}][{split_name}] epoch={epoch} "
                f"train={np.mean(total_losses):.4f} pred={np.mean(pred_losses):.4f} val={val_loss:.4f}"
            )
            if dev_losses:
                msg += f" dev={np.mean(dev_losses):.4f}"
            if pair_losses:
                msg += f" pair={np.mean(pair_losses):.4f}"
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

    # Evaluation.
    idx_test = np.where(test_mask)[0]
    model.eval()
    preds_scaled = []
    zs = []

    with torch.no_grad():
        for start in range(0, len(idx_test), batch_size):
            idx = idx_test[start:start + batch_size]
            xb = torch.tensor(X_seq_all[idx], dtype=torch.float32).to(device)
            cb = torch.tensor(X_cov_all[idx], dtype=torch.float32).to(device)
            calb = torch.tensor(X_cal_all[idx], dtype=torch.float32).to(device)
            y_hat, _, z = model(xb, cb, calb, grl_lambda=0.0, use_calibration=use_calibration_head)
            preds_scaled.append(y_hat.detach().cpu().numpy())
            zs.append(z.detach().cpu().numpy())

    pred = y_scaler.inverse_transform(np.concatenate(preds_scaled).reshape(-1, 1)).ravel()
    y_true = y[idx_test]
    m = _metrics(y_true, pred)

    model_name = "meter_v2"
    if not use_calibration_head:
        model_name += "_no_calibration"
    if not adv_enabled:
        model_name += "_no_device_adv"
    if not pair_enabled:
        model_name += "_no_pair"

    row = {
        "split_file": split_name,
        "model": model_name,
        "encoder": encoder,
        "target": target,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "seq_len": int(seq_len),
        "cov_dim": int(cov_dim),
        "cal_dim": int(cal_dim),
        "n_train_pairs": int(len(pair_index)),
        "device_adv_requested": bool(use_device_adv),
        "device_adv_enabled": bool(adv_enabled),
        "pair_consistency_requested": bool(use_pair_consistency),
        "pair_consistency_enabled": bool(pair_enabled),
        "use_calibration_head": bool(use_calibration_head),
        "include_device_as_predictor": bool(include_device_as_predictor),
        "lambda_device": float(lambda_device),
        "lambda_pair_z": float(lambda_pair_z),
        "lambda_pair_pred": float(lambda_pair_pred),
        **m,
    }

    stem = Path(split_name).stem
    result_path = res_dir / f"meter_v2_{encoder}_{stem}_{target}.csv"
    pd.DataFrame([row]).to_csv(result_path, index=False)

    pred_df = meta.iloc[idx_test][["event_id", "paired_event_id", "subject_id", "device", "meal_type", "meal_timestamp"]].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = pred
    pred_df["abs_error"] = np.abs(y_true - pred)
    pred_path = res_dir / f"meter_v2_{encoder}_{stem}_{target}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    z_arr = np.concatenate(zs, axis=0)
    repr_path = res_dir / f"meter_v2_{encoder}_{stem}_{target}_repr.npz"
    np.savez_compressed(
        repr_path,
        z=z_arr,
        event_id=pred_df["event_id"].astype(str).to_numpy(),
        paired_event_id=pred_df["paired_event_id"].astype(str).to_numpy(),
        subject_id=pred_df["subject_id"].astype(str).to_numpy(),
        device=pred_df["device"].astype(str).to_numpy(),
        meal_type=pred_df["meal_type"].astype(str).to_numpy(),
        y_true=y_true,
        y_pred=pred,
    )

    support_detail, support_summary = _support_calibration_summary(
        pred_df,
        shots=shots,
        split_name=split_name,
        model_name=f"meter_v2_{encoder}",
    )
    support_detail_path = res_dir / f"meter_v2_{encoder}_{stem}_{target}_support_detail.csv"
    support_summary_path = res_dir / f"meter_v2_{encoder}_{stem}_{target}_support_summary.csv"
    support_detail.to_csv(support_detail_path, index=False)
    support_summary.to_csv(support_summary_path, index=False)

    print(f"[RESULT] {row}")
    print(f"[OK] Saved result: {result_path}")
    print(f"[OK] Saved predictions: {pred_path}")
    print(f"[OK] Saved representation: {repr_path}")
    print(f"[OK] Saved support summary: {support_summary_path}")
