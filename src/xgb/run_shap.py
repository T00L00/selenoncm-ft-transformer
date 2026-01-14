import numpy as np
import shap
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost
from .data import get_features, load_config
import argparse
import os

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str, help="name of experiment in ouptuts directory", required=True)
    parser.add_argument("-c", "--config", type=str, help="config file path", required=True)
    args = parser.parse_args()

    EXPERIMENT = Path(f"./outputs/{args.experiment}")
    if not os.path.isdir(EXPERIMENT):
        raise Exception(f"Experiment directory {EXPERIMENT} not found...")
    
    MODEL_PATH = EXPERIMENT / "xgb.json"
    TRAINDS_PATH = EXPERIMENT / "train_ds.npz"
    TESTDS_PATH = EXPERIMENT / "test_ds.npz"

    if not os.path.exists(args.config):
        raise Exception(f"Could not find config file: {args.config}")

    config = load_config(args.config)

    train_ds = np.load(TRAINDS_PATH)
    val_ds = np.load(TESTDS_PATH)
    num_features = val_ds["X"].shape[1]
    feature_names = get_features(config["experiment"]["dataset"])

    model = xgboost.XGBClassifier(
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

    # Choose background and explain sets (keep small-ish)
    rng = np.random.default_rng(0)
    background_size = 50
    explain_size = 200

    bg_idx = rng.choice(len(train_ds["X"]), size=min(background_size, len(train_ds["X"])), replace=False)
    ex_idx = rng.choice(len(val_ds["X"]), size=min(explain_size, len(val_ds["X"])), replace=False)

    X_bg = train_ds["X"][bg_idx]
    X_explain = val_ds["X"][ex_idx]

    # Use TreeExplainer for XGBoost models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    print("SHAP:", shap_values.shape, "X:", X_explain.shape)

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        plot_size=(12,8),
        max_display=30,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(EXPERIMENT / "shap-overview.png")
    plt.clf()

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=30,
        plot_size=(12,8),
        show=False
    )
    plt.tight_layout()
    plt.savefig(EXPERIMENT / "shap-beeswarm.png")


