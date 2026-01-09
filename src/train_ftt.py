from preprocess import MorphologyDataset, Normalize, load_preprocessed_data, load_config
from ft_transformer import TabTransformerClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
)
import pandas as pd
import numpy as np
import copy
import os
from pathlib import Path
import yaml
import argparse

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_predicted = []
    all_y = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        predicted = model(X)
        all_predicted.append(predicted.detach().cpu())
        all_y.append(y.detach().cpu())

    predicted = torch.cat(all_predicted).numpy()
    y_true = torch.cat(all_y).numpy()

    probs = 1.0 / (1.0 + np.exp(-predicted))  # sigmoid
    y_pred = (probs >= 0.5).astype(np.int64)

    # Metrics (guard against edge cases with single-class batches)
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y_true, probs)
    except ValueError:
        out["roc_auc"] = float("nan")

    try:
        out["pr_auc"] = average_precision_score(y_true, probs)
    except ValueError:
        out["pr_auc"] = float("nan")

    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["acc"] = accuracy_score(y_true, y_pred)
    return out

def train_one_run(
    train_ds: MorphologyDataset,
    val_ds: MorphologyDataset,
    test_ds: MorphologyDataset,
    n_features: int,
    *,
    d_model: int=128,
    n_heads: int=8,
    n_layers: int=3,
    dropout: float=0.2,
    token_dropout: float=0.1,
    batch_size: int=64,
    lr: float=2e-4,
    weight_decay: float=1e-3,
    max_epochs: int=200,
    patience: int=20,
    pos_weight=None,   # torch scalar or None
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model = TabTransformerClassifier(
        num_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        token_dropout=token_dropout,
        mlp_hidden=d_model,
    ).to(device)

    # Loss
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Simple cosine schedule (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best = {"model": None, "validation_score": -np.inf, "epoch": -1}
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device).float()

            logits = model(Xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        # early-stop metric roc-auc
        score = metrics["roc_auc"]
        if np.isnan(score):
            # fallback if roc-auc can't be computed
            score = metrics["pr_auc"]

        if score > best["validation_score"]:
            best["validation_score"] = score
            best["epoch"] = epoch
            best["model"] = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    # Load best
    if best["model"] is not None:
        model.load_state_dict(best["model"])

    test_metrics = evaluate(model, test_loader, device)
    return model, test_metrics, best    

def run_normal_training(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    config: dict
):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)

    X_train, X_combine, y_train, y_combine = train_test_split(
        X, y, test_size=config["training"]["test_size"], stratify=y, random_state=config["training"]["seed"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_combine, y_combine, test_size=config["training"]["test_size"], stratify=y_combine, random_state=config["training"]["seed"]
    )

    # Preprocess (fit on train only)
    normalize = Normalize(mode="robust", clip_z=5.0)
    normalize.fit(X_train)
    X_train = normalize.transform(X_train)
    X_val = normalize.transform(X_val)
    X_test = normalize.transform(X_test)

    train_ds = MorphologyDataset(X_train, y_train)
    val_ds = MorphologyDataset(X_val, y_val)
    test_ds = MorphologyDataset(X_test, y_test)

    # Handle class imbalance via pos_weight if needed
    # pos_weight = (#neg / #pos)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    model, test_metrics, best = train_one_run(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        n_features=X_train.shape[1],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
        token_dropout=config["model"]["token_dropout"],
        batch_size=config["training"]["batch_size"],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        pos_weight=pos_weight,
    )

    print("Best epoch:", best["epoch"])
    print("Test metrics:", test_metrics)

    return model, train_ds, val_ds, test_ds

def run_5fold_cv(
    df: pd.DataFrame,
    feature_cols,
    label_col: str,
    config: dict,
    n_splits: int = 5,
    random_state: int = 42,
    save_folds: bool = True,
):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        # Split train fold into train/val
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            stratify=y_train_full,
            random_state=random_state,
        )

        # Preprocess (fit on train only)
        normalize = Normalize(mode="robust", clip_z=5.0)
        normalize.fit(X_train)
        X_train_n = normalize.transform(X_train)
        X_val_n = normalize.transform(X_val)
        X_test_n = normalize.transform(X_test)

        train_ds = MorphologyDataset(X_train_n, y_train)
        val_ds = MorphologyDataset(X_val_n, y_val)
        test_ds = MorphologyDataset(X_test_n, y_test)

        # Handle class imbalance via pos_weight if needed
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        model, test_metrics, best = train_one_run(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            n_features=X_train_n.shape[1],
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["n_heads"],
            n_layers=config["model"]["n_layers"],
            dropout=config["model"]["dropout"],
            token_dropout=config["model"]["token_dropout"],
            batch_size=config["training"]["batch_size"],
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["weight_decay"]),
            max_epochs=config["training"]["max_epochs"],
            patience=config["training"]["patience"],
            pos_weight=pos_weight,
        )

        row = {
            "fold": fold,
            "best_epoch": int(best.get("epoch", -1)),
            "best_val_score": float(best.get("validation_score", float("nan"))),
            **{k: float(v) for k, v in test_metrics.items()},
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "n_test": int(len(test_ds)),
            "pos_weight": float(pos_weight.item()),
        }
        fold_rows.append(row)

        print(f"[CV] Fold {fold}/{n_splits} | best_epoch={row['best_epoch']} | test={test_metrics}")

        if save_folds:
            fold_dir = EXPERIMENT_DIR / f"cv_fold_{fold}"
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(model.state_dict(), fold_dir / "model_wts.pt")
            torch.save(
                {
                    "normalize": {
                        "mode": "robust",
                        "clip_z": 5.0,
                        # Store fitted params if Normalize exposes them
                        "__dict__": getattr(normalize, "__dict__", {}),
                    },
                    "best": best,
                    "metrics": test_metrics,
                    "row": row,
                },
                fold_dir / "fold_summary.pt",
            )
            torch.save({"X": X_train_n, "y": y_train}, fold_dir / "train_ds.pt")
            torch.save({"X": X_val_n, "y": y_val}, fold_dir / "val_ds.pt")
            torch.save({"X": X_test_n, "y": y_test}, fold_dir / "test_ds.pt")

    results = pd.DataFrame(fold_rows)

    # Aggregate
    metric_cols = [c for c in ["roc_auc", "pr_auc", "f1", "acc"] if c in results.columns]
    summary = {
        "n_splits": n_splits,
        "random_state": random_state,
        "val_size": 0.2,
        "mean": results[metric_cols].mean(numeric_only=True).to_dict(),
        "std": results[metric_cols].std(ddof=1, numeric_only=True).to_dict(),
    }

    print("\n[CV] Per-fold results:\n", results)
    print("\n[CV] Summary (mean ± std):")
    for m in metric_cols:
        mu = summary["mean"].get(m, float("nan"))
        sd = summary["std"].get(m, float("nan"))
        print(f"  {m}: {mu:.4f} ± {sd:.4f}")

    if save_folds:
        results.to_csv(EXPERIMENT_DIR / "cv_results.csv", index=False)
        with open(EXPERIMENT_DIR / "cv_summary.yml", "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)

    return {"per_fold": results, "summary": summary}

def save_ds(ds: MorphologyDataset, filename: str):
    torch.save({
        "X": ds.X.detach().cpu().numpy(),
        "y": ds.y.detach().cpu().numpy()
    }, EXPERIMENT_DIR / filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="normal or cross-validation")
    args = parser.parse_args()

    mode = args.mode
    if mode not in ("normal", "cv"):
        print(f"Training mode {args.mode} not recognized. Defaulting to normal training...")
        mode = "normal"

    OUTPUTS_DIR = Path("./outputs")

    # Create experiment directory with unique identifier attached to "fft_" prefix
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    experiments = list(OUTPUTS_DIR.rglob("ftt_*"))
    EXPERIMENT_DIR = OUTPUTS_DIR / f"ftt_{len(experiments)+1}"
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"Created experiment directory {str(EXPERIMENT_DIR)}...")

    df = load_preprocessed_data()

    print("Loaded preprocessed data...")

    # Train
    config = load_config(Path("./configs/config.yml"))
    label_col = "label"  # 0/1
    feature_cols = [c for c in df.columns if c != label_col]

    if mode == "normal":
        print("Normal training started...")
        model, train_ds, val_ds, test_ds = run_normal_training(df, feature_cols, label_col, config)

        # Save model and test split
        torch.save(model.state_dict(), EXPERIMENT_DIR / "model_wts.pt" )
        save_ds(train_ds, "train_ds.pt")
        save_ds(val_ds, "val_ds.pt")
        save_ds(test_ds, "test_ds.pt")
        print(f"Trained model and test split saved in {EXPERIMENT_DIR}...")

    elif mode == "cv":
        print("5-fold cross-validation training started...")
        run_5fold_cv(df, feature_cols, label_col, config)