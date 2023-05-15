from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _get_misclassified_sample(self, X: np.ndarray, y: np.ndarray):
        if self.include_intercept_:
            X_to_pred = X[:,1:]
        else:
            X_to_pred = X
        response = self._predict(X_to_pred)
        for i, label in enumerate(response):
            if label != y[i]:
                return y[i],X[i]
        return None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        self.coefs_ = np.zeros((X.shape[1],))
        self.fitted_ = True
        if self.include_intercept_:
            ones = np.ones((X.shape[0],1))
            X = np.hstack([ones, X]) #TODO order?
            self.coefs_ = np.zeros((X.shape[1],))
        misclassified = self._get_misclassified_sample(X, y)
        iterations = 0
        while misclassified is not None and iterations < self.max_iter_: #TODO < or <=?
            self.callback_(self, misclassified[1], misclassified[0])  # TODO what?
            self.coefs_ = self.coefs_ + misclassified[0]*misclassified[1]
            misclassified = self._get_misclassified_sample(X, y)
            iterations += 1
        self.callback_(self, None, None)



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            ones = np.ones((X.shape[0],1))
            X = np.hstack([ones, X]) #TODO order?
        distance = self.coefs_ @ X.T
        return np.sign(distance).astype(int) + (distance == 0)


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
        return misclassification_error(y, response) #TODO should normalize?
