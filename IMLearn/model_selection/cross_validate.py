from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[
    float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_scores = []
    val_scores = []
    permutation = np.random.permutation(len(X))
    shuffled_X = X#[permutation]
    shuffled_y = y#[permutation]

    for i in range(cv):
        fold_size = int(np.floor(shuffled_X.shape[0] / cv))
        test = shuffled_X[i * fold_size:(i + 1) * fold_size]
        train = np.concatenate((shuffled_X[:i * fold_size], shuffled_X[(i + 1) * fold_size:]))
        train_y = np.concatenate((shuffled_y[:i * fold_size], shuffled_y[(i + 1) * fold_size:]))
        estimator.fit(train, train_y)
        train_response = estimator.predict(train)
        train_scores.append(scoring(train_y, train_response))
        test_response = estimator.predict(test)
        test_y = shuffled_y[i * fold_size:(i + 1) * fold_size]
        val_scores.append(scoring(test_y, test_response))
    return float(np.mean(train_scores)), float(np.mean(val_scores))
