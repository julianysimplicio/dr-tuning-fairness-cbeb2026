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

**Before pushing this to Zenodo/GitHub as the paper's official code
release, run it once end-to-end and confirm the printed Table IV / Table V
values are close to the ones published in the paper.** Small numerical
differences are expected and acceptable (MLPClassifier's default weight
initialization is not seeded in scikit-learn, so exact bit-for-bit
reproduction of the MLP rows is not guaranteed run-to-run; everything
else -- data split, K-means, RBF network, RF/SVM/GB -- is fully seeded and
should reproduce closely). If a value is off by more than what you'd
expect from that MLP-seeding caveat, treat it as a real bug and open an
issue / fix before publishing.

This was verified for syntax correctness (`python -m py_compile` on every
file) and the statistics module (Holm-Bonferroni + Cohen's d_z) was
independently checked against the paper's own published t/p/d_z values in
Table V and reproduces them exactly. The full pipeline could not be
executed end-to-end in the environment used to write this code, because
that sandbox has no internet access to install scikit-learn/scipy or to
download the dataset -- it has only been checked for correctness by
careful reading and by the statistics unit-check above. **Running it
yourself once, locally, is the necessary last step.**

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
