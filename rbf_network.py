"""
Custom RBF-network classifier used in:
"On the Effect of Hyperparameter Tuning Fairness in Classical Classifier
Comparisons: A Case Study on Feature-Based Diabetic Retinopathy Screening"

Architecture (Section IV-C of the paper):
  - Hidden layer: K centers found by K-means on the (already per-fold
    normalized) training features.
  - Gaussian spread: sigma = s * d_max / sqrt(2K), where d_max is the
    maximum pairwise Euclidean distance between the K centers.
  - Output layer: logistic regression (regularization C) trained on the
    K Gaussian-basis-function activations.

The class follows the scikit-learn Estimator API (fit / predict /
predict_proba / get_params / set_params) so it can be dropped directly
into a Pipeline and searched with GridSearchCV, exactly like the other
four classifiers in the study.
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RBFNetworkClassifier(BaseEstimator, ClassifierMixin):
    """RBF network: K-means centers + Gaussian hidden layer + logistic output."""

    def __init__(self, K=30, s=1.0, C=1.0, kmeans_n_init=10, random_state=7):
        self.K = K
        self.s = s
        self.C = C
        self.kmeans_n_init = kmeans_n_init
        self.random_state = random_state

    def _rbf_features(self, X):
        # Gaussian basis function activations for every sample against
        # every center: phi_ij = exp(-||x_i - c_j||^2 / (2 * sigma^2))
        d2 = cdist(X, self.centers_, metric="sqeuclidean")
        return np.exp(-d2 / (2.0 * self.sigma_ ** 2))

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]
        k = min(self.K, n_samples)  # guard for tiny folds / smoke tests

        kmeans = KMeans(
            n_clusters=k,
            n_init=self.kmeans_n_init,
            random_state=self.random_state,
        )
        kmeans.fit(X)
        self.centers_ = kmeans.cluster_centers_

        if k > 1:
            d_max = pdist(self.centers_, metric="euclidean").max()
        else:
            d_max = 1.0  # degenerate case, avoids division by zero
        self.sigma_ = self.s * d_max / np.sqrt(2.0 * k)
        if self.sigma_ <= 0:
            self.sigma_ = 1e-6

        phi = self._rbf_features(X)
        self.output_ = LogisticRegression(
            C=self.C, max_iter=2000, random_state=self.random_state
        )
        self.output_.fit(phi, y)
        self.classes_ = self.output_.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "output_")
        X = check_array(X)
        phi = self._rbf_features(X)
        return self.output_.predict_proba(phi)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
