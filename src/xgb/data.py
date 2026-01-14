import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from pathlib import Path
import yaml
import os

DATASET = Path("./data")
    
class Normalize:
    """
    Normalize the data and handle missing values
      - median imputation
      - robust scaling (median/IQR) or standard scaling
      - optional clipping
    """
    def __init__(self, mode="robust", clip_z=None, eps=1e-8):
        assert mode in ("robust", "standard")
        self.mode = mode
        self.clip_z = clip_z
        self.eps = eps
        self.median_ = None
        self.scale_ = None  # IQR or std

    def fit(self, X: np.ndarray):
        '''
        Computes the median and scale for normalization
        
        :param X: dataframe containing dataset with shape=(n_samples, n_features)
        :type X: np.ndarray
        '''
        # X: [N, D]
        self.median_ = np.nanmedian(X, axis=0)

        X_imp = np.where(np.isnan(X), self.median_, X)

        # Impute by iqr
        if self.mode == "robust":
            q1 = np.percentile(X_imp, 25, axis=0)
            q3 = np.percentile(X_imp, 75, axis=0)
            iqr = (q3 - q1)
            self.scale_ = np.where(iqr < self.eps, 1.0, iqr)
        else:
            std = X_imp.std(axis=0)
            self.scale_ = np.where(std < self.eps, 1.0, std)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        '''
        Performs normalization
        
        :param X: dataframe containing dataset with shape=(n_samples, n_features)
        :type X: np.ndarray
        :return: normalized dataframe
        :rtype: ndarray
        '''
        assert self.median_ is not None and self.scale_ is not None, "Must run Normalize.fit() first"
        X_imp = np.where(np.isnan(X), self.median_, X)

        if self.mode == "robust":
            X_scaled = (X_imp - self.median_) / self.scale_
        else:
            mean = X_imp.mean(axis=0)  # NOTE: if you want strict train-only mean, store it too.
            X_scaled = (X_imp - mean) / self.scale_

        if self.clip_z is not None:
            X_scaled = np.clip(X_scaled, -self.clip_z, self.clip_z)

        return X_scaled.astype(np.float32)

def load_dataset(path: Path) -> pd.DataFrame:
    if not os.path.exists(path):
        raise Exception(f"{path} dataset does not exist!")
    
    with open(path, "rb") as f:
        df: pd.DataFrame = pickle.load(f)
    return df

def load_config(path: str | Path) -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
    
def get_features(dataset_path: Path) -> list[str]:
    if not os.path.exists(dataset_path):
        raise Exception(f"{dataset_path} dataset does not exist!")
    
    with open(dataset_path, "rb") as f:
        df: pd.DataFrame = pickle.load(f)
    return df.columns
