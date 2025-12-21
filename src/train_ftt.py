from preprocess import MorphologyDataset, Normalize, load_preprocessed_data
from ft_transformer import TabTransformerClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, average_precision_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import pandas as pd
import numpy as np
import copy

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
    X_train, y_train,
    X_val, y_val,
    *,
    d_model=128,
    n_heads=8,
    n_layers=3,
    dropout=0.2,
    token_dropout=0.1,
    batch_size=64,
    lr=2e-4,
    weight_decay=1e-3,
    max_epochs=200,
    patience=20,
    pos_weight=None,   # torch scalar or None
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MorphologyDataset(X_train, y_train)
    val_ds = MorphologyDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    model = TabTransformerClassifier(
        num_features=X_train.shape[1],
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

    best = {"model": None, "score": -np.inf, "epoch": -1}
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

        if score > best["score"]:
            best["score"] = score
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

    final_metrics = evaluate(model, val_loader, device)
    return model, final_metrics, best

def run_train_val_split(
    df: pd.DataFrame,
    feature_cols,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)

    X_train, X_combine, y_train, y_combine = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    X_val, X_test, y_val, y_test = train_test_split(
        X_combine, y_combine, test_size=test_size, stratify=y, random_state=random_state
    )

    # TODO - run separate test on trained model

    # Preprocess (fit on train only)
    normalize = Normalize(mode="robust", clip_z=5.0)
    normalize.fit(X_train)
    X_train = normalize.transform(X_train)
    X_val = normalize.transform(X_val)

    # Handle class imbalance via pos_weight if needed
    # pos_weight = (#neg / #pos)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    model, val_metrics, best = train_one_run(
        X_train, y_train,
        X_val, y_val,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.2,
        token_dropout=0.1,
        batch_size=64,
        lr=2e-4,
        weight_decay=1e-3,
        max_epochs=300,
        patience=30,
        pos_weight=pos_weight,
    )

    print("Best epoch:", best["epoch"])
    print("Validation metrics:", val_metrics)

    return model, normalize, X_train, X_val, y_train, y_val

if __name__ == "__main__":

    df = load_preprocessed_data()

    # Train
    label_col = "label"  # 0/1
    feature_cols = [c for c in df.columns if c != label_col]
    model, prep, X_tr, X_va, y_tr, y_va = run_train_val_split(df, feature_cols, label_col)