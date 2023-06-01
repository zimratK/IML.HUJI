from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500 #TODO!!
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True
                                  # , as_frame=True
                                  )
    train_X = X[:n_samples, :]
    test_X = X[n_samples:, :]
    train_y = y[:n_samples]
    test_y = y[n_samples:]

    # TODO random
    # train_proportion = n_samples / X.shape[0]
    # train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion)
    # train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(
    #     test_y)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_range_for_lasso = np.linspace(pow(10, -2), 2, n_evaluations)
    lambda_range_for_ridge = np.linspace(pow(10, -5), pow(10, -2) * 5, n_evaluations)

    train_errors_lasso = []
    val_errors_lasso = []
    train_errors_ridge = []
    val_errors_ridge = []

    for param in lambda_range_for_lasso:
        train_err, val_err = cross_validate(Lasso(param), train_X, train_y, mean_square_error)
        train_errors_lasso.append(train_err)
        val_errors_lasso.append(val_err)

    for param in lambda_range_for_ridge:
        train_err, val_err = cross_validate(RidgeRegression(param), train_X, train_y, mean_square_error)
        train_errors_ridge.append(train_err)
        val_errors_ridge.append(val_err)

    plot = go.Figure(
        [
            go.Scatter(
                x=lambda_range_for_lasso,
                y=train_errors_lasso,
                mode='lines+markers',
                name='train error'
            ),
            go.Scatter(
                x=lambda_range_for_lasso,
                y=val_errors_lasso,
                mode='lines+markers',
                name='validation error'
            )
        ]
    )
    plot.update_layout(
        title='Train and Validation Errors for Different Lambda Values - Lasso Regression',
        xaxis_title='Lambda',
        yaxis_title='MSE',
        width=700,
        height=500
    )
    plot.update_xaxes(range=[np.min(lambda_range_for_lasso), np.max(lambda_range_for_lasso)])
    plot.show()

    plot = go.Figure(
        [
            go.Scatter(
                x=lambda_range_for_ridge,
                y=train_errors_ridge,
                mode='lines+markers',
                name='train error'
            ),
            go.Scatter(
                x=lambda_range_for_ridge,
                y=val_errors_ridge,
                mode='lines+markers',
                name='validation error'
            )
        ]
    )
    plot.update_layout(
        title='Train and Validation Errors for Different Lambda Values - Ridge Regression',
        xaxis_title='Lambda',
        yaxis_title='MSE',
        width=700,
        height=500
    )
    plot.update_xaxes(range=[np.min(lambda_range_for_ridge), np.max(lambda_range_for_ridge)])
    plot.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lasso_index = np.argmin(val_errors_lasso)
    best_lasso_lambda = lambda_range_for_lasso[best_lasso_index]
    print("best lambda for lasso:", best_lasso_lambda)
    best_ridge_index = np.argmin(val_errors_ridge)
    best_ridge_lambda = lambda_range_for_ridge[best_ridge_index]
    print("best lambda for ridge:", best_ridge_lambda)
    ridge = RidgeRegression(best_ridge_lambda)
    ridge.fit(train_X, train_y)
    print("Ridge Loss:", ridge.loss(test_X, test_y))
    lasso = Lasso(best_lasso_lambda)
    lasso.fit(train_X, train_y)
    lasso_result = lasso.predict(test_X)
    print("Lasso Loss:", mean_square_error(test_y, lasso_result))
    least_squares = LinearRegression()
    least_squares.fit(train_X, train_y)
    print("Least Squares Loss:", least_squares.loss(test_X, test_y))


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
