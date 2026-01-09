import numpy as np
import torch
import shap
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ft_transformer import TabTransformerClassifier
from preprocess import get_features, load_config, MorphologyDataset
import argparse
import os

class LogitModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # return shape [B, 1] for SHAP compatibility
        return self.model(x).unsqueeze(-1)

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str, help="name of experiment in ouptuts directory")
    args = parser.parse_args()

    EXPERIMENT = Path(f"./outputs/{args.experiment}")
    if not os.path.isdir(EXPERIMENT):
        raise Exception(f"Experiment directory {EXPERIMENT} not found...")
    
    CONFIG = Path("./configs/config.yml")
    MODEL_PATH = EXPERIMENT / "model_wts.pt"
    TRAINDS_PATH = EXPERIMENT / "train_ds.pt"
    TESTDS_PATH = EXPERIMENT / "val_ds.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(CONFIG)

    chkpt = torch.load(MODEL_PATH)
    train_ds = torch.load(TRAINDS_PATH, weights_only=False)
    val_ds = torch.load(TESTDS_PATH, weights_only=False)
    num_features = val_ds["X"].shape[1]
    feature_names = get_features()

    model = TabTransformerClassifier(
        num_features=num_features,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
        token_dropout=config["model"]["token_dropout"],
        mlp_hidden=config["model"]["mlp_hidden"],
    ).to(device)

    model.load_state_dict(chkpt)
    model.eval()

    # Choose background and explain sets (keep small-ish)
    rng = np.random.default_rng(0)
    background_size = 50
    explain_size = 200

    bg_idx = rng.choice(len(train_ds["X"]), size=min(background_size, len(train_ds["X"])), replace=False)
    ex_idx = rng.choice(len(val_ds["X"]), size=min(explain_size, len(val_ds["X"])), replace=False)

    X_bg = torch.tensor(train_ds["X"][bg_idx], dtype=torch.float32, device=device)
    X_explain = torch.tensor(val_ds["X"][ex_idx], dtype=torch.float32, device=device)

    wrapped_model = LogitModelWrapper(model).to(device)
    wrapped_model.eval()

    explainer = shap.GradientExplainer(wrapped_model, X_bg)
    shap_values = np.array(explainer.shap_values(X_explain))
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    X_explain = X_explain.detach().cpu().numpy()
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


