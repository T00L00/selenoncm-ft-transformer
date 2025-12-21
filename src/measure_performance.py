from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def collect_probs_and_labels(model, loader, device):
    model.eval()
    probs_list, y_list = [], []
    for X, y in loader:
        X = X.to(device)
        prediction = model(X)
        probs = torch.sigmoid(prediction).detach().cpu().numpy()
        probs_list.append(probs)
        y_list.append(y.detach().cpu().numpy())
    probs = np.concatenate(probs_list)
    y = np.concatenate(y_list).astype(int)
    return probs, y

def plot_model_performance(y_true, probs, title_prefix="Validation"):
    # Basic headline metrics
    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else np.nan
    pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) == 2 else np.nan

    # Curves
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, pr_thresh = precision_recall_curve(y_true, probs)

    # Thresholds
    #t_best, f1_best = best_f1_threshold(y_true, probs)

    def summarize_at_threshold(t):
        y_pred = (probs >= t).astype(int)
        return {
            "threshold": t,
            "acc": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "cm": confusion_matrix(y_true, y_pred),
        }

    s_05 = summarize_at_threshold(0.5)
    #s_best = summarize_at_threshold(t_best)

    print(f"{title_prefix} ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"@0.50:  acc={s_05['acc']:.4f} f1={s_05['f1']:.4f} prec={s_05['precision']:.4f} rec={s_05['recall']:.4f}")
    #print(f"@bestF1(thr={t_best:.3f}): acc={s_best['acc']:.4f} f1={s_best['f1']:.4f} prec={s_best['precision']:.4f} rec={s_best['recall']:.4f}")

    # Plot ROC curve and PR curve next to each other
    # ROC Curve
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve (AUC={roc_auc:.3f})")

    # Precision-Recall Curve
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.plot(rec, prec)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision–Recall (AP={pr_auc:.3f})")

    fig1.tight_layout()

    # Confusion Matrices
    def plot_cm(ax: plt.Axes, cm, cm_title):
        ax.imshow(cm, interpolation="nearest", cmap="Oranges")
        ax.set_title(cm_title)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks, ["WT(0)", "SELENON(1)"])
        ax.set_yticks(tick_marks, ["WT(0)", "SELENON(1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # annotate
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    # Plot confusion matrices next to each other
    fig2 = plt.figure(figsize=(10, 5))
    ax3 = fig2.add_subplot(1, 2, 1)

    plot_cm(ax3, s_05["cm"], f"{title_prefix} Confusion Matrix @0.50")
    #plot_cm(ax4, s_best["cm"], f"{title_prefix} Confusion Matrix @bestF1 (thr={t_best:.3f})")

    # Probability histograms by class
    probs0 = probs[y_true == 0]
    probs1 = probs[y_true == 1]

    plt.figure(figsize=(7, 5))
    plt.hist(probs0, bins=30, alpha=0.6, label="WT (0)")
    plt.hist(probs1, bins=30, alpha=0.6, label="SELENON (1)")
    plt.axvline(0.5, linestyle="--")
    #plt.axvline(t_best, linestyle="--")
    plt.xlabel("P(class=1)")
    plt.ylabel("Count")
    plt.title(f"FT-Transformer Predicted Probability Distributions")
    plt.legend()
    plt.show()

    # Calibration curve (reliability diagram)
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title_prefix} Calibration Curve")
    plt.show()

    # Threshold sweep (F1/precision/recall vs threshold)
    thresholds = np.linspace(0, 1, 201)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, f1s, label="F1")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.axvline(0.5, linestyle="--", label="0.5")
    #plt.axvline(t_best, linestyle="--", label=f"bestF1={t_best:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{title_prefix} Threshold Sweep")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # Load model and test dataset

    # Compute performance
    probs_val, y_val = collect_probs_and_labels(model, val_loader, device)
    plot_model_performance(y_val, probs_val, title_prefix="Validation")