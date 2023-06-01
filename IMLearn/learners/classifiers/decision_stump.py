from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        risks = []
        for j in range(X.shape[1]):
            threshold_sign_pos, error_sign_pos = self._find_threshold(X[:, j], y, 1)
            threshold_sign_neg, error_sign_neg = self._find_threshold(X[:, j], y, -1)
            if error_sign_pos > error_sign_neg:
                risks.append([threshold_sign_neg, error_sign_neg, -1])
            else:
                risks.append([threshold_sign_pos, error_sign_pos, 1])
        risks = np.array(risks)
        index = np.argmin(risks[:, 1])
        self.j_ = index
        self.threshold_ = risks[index][0]
        self.sign_ = risks[index][2]
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        mask = X[:, self.j_] >= self.threshold_
        response = np.zeros((X.shape[0],))
        response[mask] = self.sign_
        response[~mask] = -self.sign_
        return response

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_ids = np.argsort(values)
        sorted_labels = labels[sorted_ids]
        sorted_values = values[sorted_ids]
        risks = [np.sum(np.abs(np.ones_like(sorted_labels) * sign - sorted_labels))]
        for i, threshold in enumerate(sorted_values):
            risks.append(risks[-1] - np.abs(sign - sorted_labels[i]) + np.abs(-sign - sorted_labels[i]))
        index = np.argmin(np.array(risks))
        print(index)
        if index == 0:
            return -np.inf, risks[index]
        if index == len(risks) - 1:
            return np.inf, risks[index]
        return sorted_values[index-1], risks[index]






        ids = np.argsort(values)
        values, labels = values[ids], labels[ids]

        # Loss for classifying all as `sign` - namely, if threshold is smaller than values[0]
        loss = np.sum(np.abs(labels)[np.sign(labels) == sign])

        # Loss of classifying threshold being each of the values given
        loss = np.append(loss, loss - np.cumsum(labels * sign))

        id = np.argmin(loss)
        return np.concatenate([[-np.inf], values[1:], [np.inf]])[id], loss[id]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        response = self._predict(X)
        return misclassification_error(y, response)
