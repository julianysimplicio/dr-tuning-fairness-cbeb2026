"""
Statistical testing utilities matching Section IV-F of the paper:
  - Corrected resampled paired t-test (Nadeau & Bengio, 2003), which
    adjusts the variance estimate for the non-independence of k-fold
    cross-validation folds.
  - Holm-Bonferroni step-down correction for the family of pairwise
    comparisons.
  - Cohen's d_z paired effect size (d_z = t / sqrt(n)), reported
    alongside every p-value in Table V of the paper.
"""

import numpy as np
from scipy import stats


def corrected_resampled_ttest(fold_scores_a, fold_scores_b, n_train_folds=9, n_test_folds=1):
    """
    Corrected resampled paired t-test for k-fold cross-validation
    (Nadeau & Bengio, 2003; see also Dietterich, 1998, cited as [12],[13]
    in the paper).

    fold_scores_a, fold_scores_b : array-like, length n (per-fold accuracy,
        or any other per-fold metric, for classifiers A and B on the SAME
        folds).
    n_train_folds, n_test_folds : number of folds used for training/testing
        in each split (10-fold CV -> 9 train folds, 1 test fold).

    Returns (t_statistic, raw_p_value).
    """
    a = np.asarray(fold_scores_a, dtype=float)
    b = np.asarray(fold_scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("fold_scores_a and fold_scores_b must have the same shape")
    n = a.shape[0]
    d = a - b
    mean_d = d.mean()
    var_d = d.var(ddof=1)

    ratio = n_test_folds / n_train_folds
    corrected_var = (1.0 / n + ratio) * var_d
    if corrected_var <= 0:
        # all differences identical (e.g. smoke test with tiny/degenerate data)
        return 0.0, 1.0

    t_stat = mean_d / np.sqrt(corrected_var)
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=n - 1)
    return t_stat, p_value


def holm_bonferroni(p_values, alpha=0.05):
    """
    Holm step-down correction. Returns adjusted p-values in the ORIGINAL
    order of the input list, plus a boolean reject array at the given alpha.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * p[idx]
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    reject = adjusted < alpha
    return adjusted, reject


def cohens_dz(t_statistic, n):
    """Paired effect size from a paired t-statistic: d_z = t / sqrt(n)."""
    return t_statistic / np.sqrt(n)


def mean_ci_95(fold_values):
    """
    95% CI for the mean of n fold-level values, from the t-distribution
    (Section IV-F: "computed from the t-distribution of the ten fold-level
    accuracies for each classifier").
    """
    x = np.asarray(fold_values, dtype=float)
    n = x.shape[0]
    mean = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, (mean - t_crit * sem, mean + t_crit * sem)
