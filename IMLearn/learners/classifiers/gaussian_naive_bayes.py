from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        mu = []
        sigma = []
        pi_mle = []
        for i, k in enumerate(self.classes_):
            class_arr = np.where(y == k, 1, 0)
            n_k = np.sum(class_arr)
            mu_k = X.T @ class_arr / n_k
            mu.append(mu_k)
            x_minus_mu_sqr = (X - mu_k) ** 2
            sigma_k = x_minus_mu_sqr.T @ class_arr / n_k
            sigma.append(sigma_k)
            pi_mle.append(n_k / X.shape[0])
        self.mu_ = np.array(mu)
        self.vars_ = np.array(sigma)
        self.pi_ = np.array(pi_mle)

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
        for k in range(len(self.classes_)):
            z = 1 / np.sqrt(2 * np.pi * self.vars_[k])
            exp_factor = (X - self.mu_[k]) ** 2 / (-2 * self.vars_[k])
            likelihood_matrix = z * np.exp(exp_factor)
            result_for_class = np.prod(likelihood_matrix, axis=1) * self.pi_[k]
            likelihoods.append(result_for_class)

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
