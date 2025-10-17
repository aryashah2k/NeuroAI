from typing import Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def train_readout(X_train: np.ndarray, y_train: np.ndarray,
                  model_type: str = "logreg",
                  random_state: int = 42,
                  max_iter: int = 200,
                  solver: str = "lbfgs",
                  C: float = 1.0,
                  penalty: Optional[str] = None) -> object:
    if model_type == "logreg":
        kwargs = dict(max_iter=max_iter, n_jobs=-1, random_state=random_state, solver=solver)
        if penalty is not None:
            kwargs["penalty"] = penalty
        if C is not None:
            kwargs["C"] = C
        clf = LogisticRegression(**kwargs)
    elif model_type == "ridge":
        clf = RidgeClassifier()
    else:
        raise ValueError("model_type must be 'logreg' or 'ridge'")
    clf.fit(X_train, y_train)
    return clf


def evaluate_readout(clf: object, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)


def scale_features(X_train: np.ndarray, X_test: Optional[np.ndarray] = None):
    """Standardize features with zero mean and unit variance using training stats.
    Returns (X_train_scaled, X_test_scaled, scaler). If X_test is None, returns (X_train_scaled, None, scaler).
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if X_test is not None else None
    return X_train_s, X_test_s, scaler
