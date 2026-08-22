"""
Loads the Diabetic Retinopathy Debrecen Data Set (Section IV-A).

Source: UCI Machine Learning Repository, DOI 10.24432/C5XP4P
(Antal & Hajdu, 2014). 1151 instances, 19 numerical attributes, binary
label (1 = signs of DR present, 0 = absent).

No global preprocessing is done here on purpose: normalization must be
fit per-fold inside the nested CV loop (Section IV-B), never on the full
dataset, so this module only loads and validates the raw data.
"""

import numpy as np
import pandas as pd

UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/329/data.csv"


def load_debrecen(csv_path=None):
    """
    Load the dataset either from a local CSV (Attr1..Attr19,Target columns,
    exactly as distributed by UCI) or, if csv_path is None, by downloading
    it directly from the UCI Machine Learning Repository.

    Returns (X, y) as numpy arrays: X shape (1151, 19), y shape (1151,).
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(UCI_CSV_URL)

    target_col = "Target" if "Target" in df.columns else df.columns[-1]
    y = df[target_col].to_numpy().astype(int)
    X = df.drop(columns=[target_col]).to_numpy().astype(float)

    if X.shape[1] != 19:
        raise ValueError(
            f"Expected 19 feature columns, found {X.shape[1]}. "
            "Check that the CSV matches the UCI Debrecen dataset layout."
        )
    return X, y


def describe(X, y):
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "n_instances": len(y),
        "n_features": X.shape[1],
        "n_positive": n_pos,
        "n_negative": n_neg,
    }
