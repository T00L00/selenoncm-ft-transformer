from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
)
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
import argparse
import os
from .data import load_config

def measure_performance(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray):

    probs = model.predict_proba(X_test)[:, 1]
    pred_05 = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "acc@0.5": accuracy_score(y_test, pred_05),
        "f1@0.5": f1_score(y_test, pred_05, zero_division=0),
        "prec@0.5": precision_score(y_test, pred_05, zero_division=0),
        "rec@0.5": recall_score(y_test, pred_05, zero_division=0),
    }

    print("XGBoost validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    print("Generating evaluation plots...")

    roc_pr_fig = plt.figure(figsize=(10,5))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_ax = roc_pr_fig.add_subplot(1,2,1)
    roc_ax.plot(fpr, tpr)
    roc_ax.plot([0,1],[0,1], linestyle="--")
    roc_ax.set_xlabel("FPR")
    roc_ax.set_ylabel("TPR")
    roc_ax.set_title(f"ROC (AUC={metrics['roc_auc']:.3f})")

    # PR
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_ax = roc_pr_fig.add_subplot(1,2,2)
    pr_ax.plot(rec, prec)
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title(f"Precision–Recall (AP={metrics['pr_auc']:.3f})")

    roc_pr_fig.tight_layout()
    roc_pr_fig.savefig(EXPERIMENT / "roc-pr.png")

    plt.clf()

    # Confusion Matrix @0.5
    cm = confusion_matrix(y_test, pred_05)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest", cmap="Oranges")
    plt.title("XGBoost Confusion Matrix @0.5")
    plt.xticks([0,1], ["WT(0)", "SELENON(1)"])
    plt.yticks([0,1], ["WT(0)", "SELENON(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(EXPERIMENT / "cm05.png")

    plt.clf()

    # Probability histograms
    plt.figure(figsize=(7,5))
    plt.hist(probs[y_test==0], bins=30, alpha=0.6, label="WT (0)")
    plt.hist(probs[y_test==1], bins=30, alpha=0.6, label="SELENON (1)")
    plt.axvline(0.5, linestyle="--")
    plt.title("XGBoost Predicted Probability Distributions")
    plt.xlabel("P(class=KO)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXPERIMENT / "predicted-prob-dist.png")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str, help="name of experiment in ouptuts directory", required=True)
    parser.add_argument("-c", "--config", type=str, help="config file path", required=True)
    args = parser.parse_args()

    EXPERIMENT = Path(f"./outputs/training/{args.experiment}")
    if not os.path.isdir(EXPERIMENT):
        raise Exception(f"Experiment directory {EXPERIMENT} not found...")
    
    MODEL_PATH = EXPERIMENT / "xgb.json"
    TESTDS_PATH = EXPERIMENT / "test_ds.npz"

    if not os.path.exists(args.config):
        raise Exception(f"Could not find config file: {args.config}")
        
    config = load_config(args.config)

    test_ds = np.load(TESTDS_PATH)
    
    model = xgb.XGBClassifier(
        n_estimators=config["model"]["n_estimators"],
        learning_rate=config["training"]["lr"],
        max_depth=config["model"]["max_depth"],
        subsample=config["model"]["subsample"],
        colsample_bytree=config["model"]["subsample"],
        reg_lambda=config["model"]["reg_lambda"],
        reg_alpha=config["model"]["reg_alpha"],
        min_child_weight=config["model"]["min_child_weight"],
        gamma=config["model"]["gamma"],
        objective=config["model"]["objective"],
        eval_metric=config["model"]["eval_metric"],
        tree_method=config["model"]["tree_method"],
        random_state=config["training"]["seed"]
    )

    model.load_model(MODEL_PATH)
    measure_performance(model, test_ds["X"], test_ds["y"])

