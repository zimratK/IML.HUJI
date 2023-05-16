from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        mu = []
        pi_mle = []
        for i, k in enumerate(self.classes_):
            class_arr = np.where(y == k, 1, 0)
            n_k = np.sum(class_arr)
            mu_k = X.T @ class_arr / n_k
            mu.append(mu_k)
            pi_mle.append(n_k / X.shape[0])
        self.mu_ = np.array(mu)
        self.pi_ = np.array(pi_mle)

        cov = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            vec = X[i] - mu[y[i]]
            vec = vec.reshape((vec.shape[0], 1))
            cov += vec @ vec.T
        self.cov_ = cov / (X.shape[0] - len(self.classes_))
        self._cov_inv = inv(self.cov_)

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
        likelihoods = self.likelihood(X)
        max_likelihoods = np.argmax(likelihoods, axis=1)
        return self.classes_[max_likelihoods]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = []
        z = 1 / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.cov_))
        for k in range(len(self.classes_)):
            matrix_to_transpose = X - self.mu_[k]
            exp_factor = np.sum(matrix_to_transpose @ self._cov_inv * matrix_to_transpose, axis=1) / -2
            likelihood_matrix = np.exp(exp_factor)
            likelihoods.append(z * likelihood_matrix * self.pi_[k])
        return np.array(likelihoods).T

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
