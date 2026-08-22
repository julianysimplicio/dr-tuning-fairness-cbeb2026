"""
Leakage-free nested cross-validation protocol (Sections III, IV-B, IV-D).

Key design point: normalization is fit ONLY on the current training
partition, at BOTH the outer and inner level. This is achieved by putting
MinMaxScaler inside a sklearn Pipeline together with the classifier:
scikit-learn's cross-validation machinery (GridSearchCV, cross_val_predict,
manual loops below) always fits every step of a Pipeline on the training
fold only and applies it (transform-only, no refit) to the held-out fold.
That is exactly the leakage-free behaviour described in the paper -- no
statistic computed from a test fold is ever used upstream of that fold's
prediction.

Outer loop:   StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
Inner loop:   StratifiedKFold(n_splits=3,  shuffle=True, random_state=7)
              (only used for configurations with a non-empty param grid)

For every outer fold, every sample receives exactly one out-of-fold
prediction; these are pooled across all 10 outer folds for reporting
(Section IV-D), and the ten per-fold accuracies are kept separately for
the statistical tests in Section IV-F / V-E.
"""

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 7
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 3


def run_nested_cv(estimator, X, y, param_grid=None, refit_kwargs=None):
    """
    Run the leakage-free nested (or plain, if param_grid is None) CV
    protocol for a single classifier configuration.

    estimator   : an unfitted scikit-learn-compatible classifier instance
                  (e.g. MLPClassifier(...), RBFNetworkClassifier(...)).
    X, y        : full dataset (numpy arrays / pandas objects).
    param_grid  : dict of {"clf__param": [values]} for GridSearchCV, using
                  the Pipeline step name "clf". Pass None for a default
                  (untuned) run -- no inner loop is performed in that case.
    refit_kwargs: optional dict of estimator params to force on the FINAL
                  refit after best-param selection (used for the RBF
                  network's n_init=2 during search / n_init=10 on refit,
                  Section IV-C).

    Returns a dict with:
      fold_acc        : array of shape (10,), per-outer-fold accuracy
      oof_true        : pooled out-of-fold true labels (n_samples,)
      oof_pred        : pooled out-of-fold predicted labels (n_samples,)
      oof_proba       : pooled out-of-fold P(class=1) (n_samples,)
      best_params     : list of the winning grid combo per outer fold
                         (None entries for untuned runs)
      search_time_s   : total wall-clock time spent in the inner grid
                         search across all 10 outer folds (0 for untuned)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_acc = np.zeros(N_OUTER_FOLDS)
    oof_true = np.empty_like(y)
    oof_pred = np.empty_like(y)
    oof_proba = np.zeros(len(y), dtype=float)
    best_params_per_fold = []
    search_time_s = 0.0

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", estimator)])

        if param_grid is None:
            # Default / untuned configuration: fit once on the outer-train
            # fold (normalization is still fit on outer-train only).
            pipe.fit(X_train, y_train)
            best_params_per_fold.append(None)
            fitted = pipe
        else:
            inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            t0 = time.time()
            search = GridSearchCV(
                pipe, param_grid=param_grid, cv=inner_cv,
                scoring="accuracy", refit=False, n_jobs=1,
            )
            search.fit(X_train, y_train)
            search_time_s += time.time() - t0
            best_params_per_fold.append(search.best_params_)

            # Final refit on the FULL outer-training fold, with
            # normalization refit on that same fold (Section IV-D).
            final_params = dict(search.best_params_)
            if refit_kwargs:
                final_params.update({f"clf__{k}": v for k, v in refit_kwargs.items()})
            pipe.set_params(**final_params)
            pipe.fit(X_train, y_train)
            fitted = pipe

        y_pred = fitted.predict(X_test)
        y_proba = fitted.predict_proba(X_test)[:, 1]

        fold_acc[fold_idx] = (y_pred == y_test).mean()
        oof_true[test_idx] = y_test
        oof_pred[test_idx] = y_pred
        oof_proba[test_idx] = y_proba

    return {
        "fold_acc": fold_acc,
        "oof_true": oof_true,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "best_params": best_params_per_fold,
        "search_time_s": search_time_s,
    }
