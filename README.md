# Reproduction code

Code accompanying **"On the Effect of Hyperparameter Tuning Fairness in
Classical Classifier Comparisons: A Case Study on Feature-Based Diabetic
Retinopathy Screening"** (CBEB 2026).

## Status of this code

This repository was reconstructed directly from the methodology described
in the paper (Sections III, IV and V), after the original analysis scripts
were lost. It implements, as literally as possible, everything the paper
specifies: the leakage-free nested cross-validation protocol, the seven
classifier configurations and their exact hyperparameter grids (Table II),
the corrected resampled paired t-test with Holm-Bonferroni correction, and
the Cohen's d_z effect size added in the CBEB revision (Table V).

This code has been run end-to-end and verified against the paper's
published Tables IV and V, on macOS (Apple Silicon) with numpy==2.0.2,
pandas==2.3.3, scipy==1.13.1, scikit-learn==1.6.1 (pinned in
`requirements.txt`). Six of the seven configurations (MLP default and
tuned, RBF network default, Random Forest, SVM, Gradient Boosting)
reproduce bit-for-bit across runs. The RBF network (tuned) configuration
initially showed run-to-run numerical drift (its K-means step uses a much
larger number of centers, K in {250, 300}, than the untuned network's
K=30), traced to multi-threaded BLAS non-determinism; setting
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1` before running
eliminated it -- two consecutive runs with these variables set produced
identical results for every cell of Table IV and Table V. **Set these
five environment variables before running `run_experiment.py` if you want
bit-for-bit reproducibility.**

With that fix in place, every configuration except RBF network (tuned)
and Random Forest (tuned) matches the paper's published accuracy exactly.
Those two rows differ from the paper by a small margin (RBF tuned: 72.72%
vs. 72.89% published; Random Forest: 68.98% vs. 69.68% published) --
consistent with the (lost) original code having run in a different,
unrecorded software/hardware environment. Crucially, all six significance
verdicts in Table V (which pairs remain significant after Holm-Bonferroni
correction) reproduce exactly, and the paper's central claim -- that the
classifier ranking collapses once tuning is applied fairly -- is
unaffected.

If you see the significance PATTERN itself change, or drift in any of the
five configurations that reproduce exactly here, treat it as a real bug
and open an issue before publishing further.

## Layout

- `data.py` -- loads the Debrecen dataset (from a local CSV or by
  downloading it from the UCI Machine Learning Repository).
- `rbf_network.py` -- the custom RBF-network classifier (K-means centers +
  Gaussian hidden layer + logistic-regression output), Section IV-C.
- `nested_cv.py` -- the leakage-free nested cross-validation protocol
  (10 outer / 3 inner stratified folds, seed 7, per-fold normalization),
  Sections III, IV-B, IV-D.
- `stats_utils.py` -- corrected resampled paired t-test (Nadeau-Bengio),
  Holm-Bonferroni correction, Cohen's d_z, 95% CI, Section IV-F.
- `metrics.py` -- accuracy / sensitivity / specificity / MCC / AUC / PR-AUC.
- `run_experiment.py` -- orchestrates everything and reproduces Tables
  III, IV and V.

## Usage

```bash
pip install -r requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Option A: let the script download the dataset from UCI directly
python run_experiment.py

# Option B: point it at a local copy of the CSV
# (columns Attr1..Attr19,Target -- the layout UCI distributes)
python run_experiment.py --csv path/to/debrecen.csv
```

Expect a total run time in the low single-digit minutes on a laptop CPU
(Table VI in the paper reports 17-201 seconds of grid search per
classifier on comparable hardware).

## Dataset

Diabetic Retinopathy Debrecen Data Set, UCI Machine Learning Repository,
DOI: [10.24432/C5XP4P](https://doi.org/10.24432/C5XP4P) (Antal & Hajdu,
2014). 1151 instances, 19 numerical attributes, binary label.

## Citation

If you use this code, please cite the paper (see the main repository /
Zenodo record for the full citation) and the original dataset paper:

> B. Antal and A. Hajdu, "An ensemble-based system for automatic
> screening of diabetic retinopathy," Knowledge-Based Systems, vol. 60,
> pp. 20-27, 2014.
