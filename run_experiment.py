"""
Full reproduction pipeline for:

  "On the Effect of Hyperparameter Tuning Fairness in Classical Classifier
  Comparisons: A Case Study on Feature-Based Diabetic Retinopathy
  Screening" (CBEB 2026).

Runs the seven classifier configurations from Table IV under the
leakage-free nested cross-validation protocol described in Sections
III-IV, then reproduces Tables III, IV and V (default/tuned performance,
the full benchmark, and the six pairwise statistical comparisons with
Holm-Bonferroni correction and Cohen's d_z effect size).

Usage:
    python run_experiment.py --csv path/to/debrecen.csv
    python run_experiment.py                     # downloads from UCI

Random seed = 7 everywhere (outer folds, inner folds, K-means), matching
Sections IV-C/IV-D of the paper.
"""

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from data import load_debrecen, describe
from rbf_network import RBFNetworkClassifier
from nested_cv import run_nested_cv
from metrics import pooled_metrics
from stats_utils import corrected_resampled_ttest, holm_bonferroni, cohens_dz, mean_ci_95

RANDOM_STATE = 7


def build_configs():
    """Seven configurations, matching Section IV-C and Table II."""
    configs = {}

    # --- MLP -----------------------------------------------------------
    mlp_base = dict(
        hidden_layer_sizes=(12, 8),
        activation="relu",
        solver="adam",
        max_iter=250,
        batch_size=10,
        random_state=RANDOM_STATE,
    )
    configs["MLP (default)"] = dict(
        estimator=MLPClassifier(**mlp_base), param_grid=None, refit_kwargs=None,
    )
    mlp_grid = {
        "clf__hidden_layer_sizes": [(12, 8), (24, 16), (50, 25)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [0.001, 0.01],
    }
    configs["MLP (tuned)"] = dict(
        estimator=MLPClassifier(
            activation="relu", solver="adam", max_iter=250,
            batch_size=10, random_state=RANDOM_STATE,
        ),
        param_grid=mlp_grid, refit_kwargs=None,
    )

    # --- RBF network -----------------------------------------------------
    configs["RBF network (default)"] = dict(
        estimator=RBFNetworkClassifier(K=30, s=1.0, C=1.0, kmeans_n_init=10, random_state=RANDOM_STATE),
        param_grid=None, refit_kwargs=None,
    )
    rbf_grid = {
        "clf__K": [250, 300],
        "clf__s": [5, 6, 7],
        "clf__C": [200, 500],
    }
    configs["RBF network (tuned)"] = dict(
        estimator=RBFNetworkClassifier(kmeans_n_init=2, random_state=RANDOM_STATE),
        param_grid=rbf_grid,
        refit_kwargs={"kmeans_n_init": 10},  # Section IV-C: n_init=10 on final refit
    )

    # --- Random Forest ---------------------------------------------------
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 3],
    }
    configs["Random Forest (tuned)"] = dict(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=rf_grid, refit_kwargs=None,
    )

    # --- SVM (RBF kernel) --------------------------------------------------
    svm_grid = {
        "clf__C": [100, 500, 1000, 5000],
        "clf__gamma": ["scale", 0.05, 0.1],
    }
    configs["SVM, RBF kernel (tuned)"] = dict(
        estimator=SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        param_grid=svm_grid, refit_kwargs=None,
    )

    # --- Gradient Boosting -------------------------------------------------
    gb_grid = {
        "clf__max_iter": [100, 200],
        "clf__max_depth": [3, 5, None],
        "clf__learning_rate": [0.05, 0.1],
    }
    configs["Gradient Boosting (tuned)"] = dict(
        estimator=HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid=gb_grid, refit_kwargs=None,
    )

    return configs


def main(csv_path):
    X, y = load_debrecen(csv_path)
    print("Dataset:", describe(X, y))

    configs = build_configs()
    results = {}

    for name, cfg in configs.items():
        print(f"\nRunning: {name} ...")
        t0 = time.time()
        out = run_nested_cv(
            cfg["estimator"], X, y,
            param_grid=cfg["param_grid"], refit_kwargs=cfg["refit_kwargs"],
        )
        elapsed = time.time() - t0
        m = pooled_metrics(out["oof_true"], out["oof_pred"], out["oof_proba"])
        mean_acc, ci = mean_ci_95(out["fold_acc"])
        results[name] = {**out, **m, "mean_acc": mean_acc, "ci": ci, "wall_time_s": elapsed}
        print(
            f"  acc={m['accuracy']*100:.2f}%  95% CI=[{ci[0]*100:.1f}, {ci[1]*100:.1f}]  "
            f"sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}  "
            f"MCC={m['mcc']:.3f}  AUC={m['auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
            f"(search time: {out['search_time_s']:.1f}s, wall: {elapsed:.1f}s)"
        )

    # ---- Table IV: comprehensive comparison -----------------------------
    table_iv = pd.DataFrame(
        [
            {
                "Configuration": name,
                "Acc. (%)": round(r["accuracy"] * 100, 2),
                "95% CI low": round(r["ci"][0] * 100, 1),
                "95% CI high": round(r["ci"][1] * 100, 1),
                "Sens.": round(r["sensitivity"], 3),
                "Spec.": round(r["specificity"], 3),
                "MCC": round(r["mcc"], 3),
                "AUC": round(r["auc"], 3),
                "PR-AUC": round(r["pr_auc"], 3),
            }
            for name, r in results.items()
        ]
    )
    print("\n=== Table IV: comprehensive comparison ===")
    print(table_iv.to_string(index=False))
    table_iv.to_csv("table_IV_comprehensive_comparison.csv", index=False)

    # ---- Table V: six pairwise comparisons, Holm-Bonferroni + d_z --------
    pairs = [
        ("MLP (default)", "RBF network (default)"),
        ("RBF network (tuned)", "RBF network (default)"),
        ("MLP (default)", "RBF network (tuned)"),
        ("SVM, RBF kernel (tuned)", "MLP (default)"),
        ("SVM, RBF kernel (tuned)", "RBF network (tuned)"),
        ("SVM, RBF kernel (tuned)", "RBF network (default)"),
    ]
    rows = []
    raw_ps = []
    for a, b in pairs:
        t_stat, p_raw = corrected_resampled_ttest(results[a]["fold_acc"], results[b]["fold_acc"])
        raw_ps.append(p_raw)
        rows.append({"Comparison": f"{a} vs. {b}", "t": t_stat, "Raw p": p_raw})
    adjusted, reject = holm_bonferroni(raw_ps)
    for row, adj, t_stat in zip(rows, adjusted, [r["t"] for r in rows]):
        row["Holm p"] = adj
        row["dz"] = cohens_dz(t_stat, n=10)

    table_v = pd.DataFrame(rows)
    print("\n=== Table V: pairwise statistical comparisons ===")
    print(table_v.to_string(index=False))
    table_v.to_csv("table_V_statistical_comparisons.csv", index=False)

    print(
        "\nDone. Compare the printed Table IV / Table V against the paper's "
        "published Tables IV and V before publishing this code to the "
        "repository -- see README.md."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default=None,
        help="Path to a local copy of the Debrecen CSV (Attr1..Attr19,Target). "
             "If omitted, the dataset is downloaded from the UCI ML Repository.",
    )
    args = parser.parse_args()
    main(args.csv)
