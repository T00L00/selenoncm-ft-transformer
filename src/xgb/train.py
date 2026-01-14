import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import argparse
from data import Normalize, load_dataset, load_config

if __name__ == "__main__":

    OUTPUTS_DIR = Path("./outputs")

    # Set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file path", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise Exception(f"Could not find config file: {args.config}")
    
    config = load_config(args.config)

    # Create experiment directory with unique identifier attached to experiment name prefix
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    experiments = list(OUTPUTS_DIR.rglob(f"{config['experiment']['name']}_*"))
    EXPERIMENT_DIR = OUTPUTS_DIR / f"{config['experiment']['name']}_{len(experiments)+1}"
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"Created experiment directory {str(EXPERIMENT_DIR)}...")

    df = load_dataset(Path(config["experiment"]["dataset"]))
    label_col = "label"
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config["training"]["seed"]
    )

    # Preprocess (fit on train only)
    normalize = Normalize(mode="robust", clip_z=5.0)
    normalize.fit(X_train)
    X_train = normalize.transform(X_train)
    X_test = normalize.transform(X_test)

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

    print("Training started...")

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    model.save_model(EXPERIMENT_DIR / "xgb.json")

    np.savez(EXPERIMENT_DIR / "train_ds.npz", X=X_train, y=y_train)
    np.savez(EXPERIMENT_DIR / "test_ds.npz", X=X_test, y=y_test)

    print(f"Trained model and test split saved in {EXPERIMENT_DIR}")