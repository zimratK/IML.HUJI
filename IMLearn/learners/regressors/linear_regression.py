from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
from ...metrics.loss_functions import mean_square_error
import numpy as np
from numpy.linalg import pinv


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            ones = np.ones((X.shape[0],1))
            X = np.hstack([X, ones])
        p_inv = np.linalg.pinv(X.T @ X)
        # u, s, vh = np.linalg.svd(X)
        # sigma_dagger = np.zeros((u.shape[0], vh.shape[0]))
        # sigma_dagger[:s.shape[0], :s.shape[0]] = np.diag(1./s)
        # x_dagger = vh.T @ sigma_dagger @ u.T
        # self.coefs_ = x_dagger @ y
        self.coefs_ = p_inv @ X.T @ y
        print("sdfsdgsdg", y[np.isnan(y)])

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
            X = np.hstack([X, ones])
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        pred = self._predict(X)
        return mean_square_error(y, pred)
