from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


# zipcodes = pd.Series()


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # if y is not None:  # train set
    #     zipcodes = X["zipcode"]
    # zipcodes_df = pd.DataFrame({'zipcode': zipcodes.astype(str)})
    # # print(pd.get_dummies(zipcodes_df))
    # # TODO

    # delete the id column
    X = X.drop('id', axis=1)
    # before calling this function, get all zipcodes of train. create dummies using these zipcodes.
    # yr_renovated - if 0, replace by value of yr_built
    X.loc[df['yr_renovated'] == 0, 'yr_renovated'] = df.loc[df['yr_renovated'] == 0, 'yr_built']
    dates = X['date'].copy(deep=True)
    # date - convert to unix time
    X['date'] = pd.to_datetime(X['date'], errors='coerce')
    X['date'].fillna(method='ffill')  # TODO explain errors='coerce'
    X['date'] = (X['date'].astype('int64') // 10 ** 9).astype('int32')
    if y is not None:
        X, y = drop_invalid_prices(X, y)
    nan_rows = X[X.isna().any(axis=1)].index
    if y is not None:
        X = X.drop(index=nan_rows)
        y = y.drop(index=nan_rows)
    X = X.fillna(0)  # handle test set
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        covariance = np.cov(X[feature], y)[0][1]  # the covariance between x and y in the cov matrix
        x_std = np.std(X[feature])
        y_std = np.std(y)
        pearson = covariance / (x_std * y_std)
        print(feature, covariance)
        plot = go.Figure(
            go.Scatter(
                x=X[feature],
                y=y,
                mode='markers'
            )
        )
        plot.update_layout(
            title='Feature: ' + feature + ', Pearson Correlation: ' + str(pearson),
            xaxis_title=feature,
            yaxis_title='price'
        )
        plot.write_image(output_path + '/' + feature + '.png')


def drop_invalid_prices(X, y):
    X = X.drop(index=y[y <= 0].index)
    y = y.drop(index=y[y <= 0].index)
    X = X.drop(index=y[y.isna()].index)
    y = y.drop(index=y[y.isna()].index)
    return X, y


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    # raise NotImplementedError()
    data = df.drop('price', axis=1)
    labels = df['price']
    train_X, train_y, test_X, test_y = split_train_test(data, labels)
    # Question 2 - Preprocessing of housing prices dataset
    prep_train, prep_labels = preprocess_data(train_X, train_y)
    prep_test, _ = preprocess_data(test_X)
    #######try######TODO delete
    regressor = LinearRegression()
    regressor.fit(prep_train.to_numpy(), prep_labels.to_numpy())
    prep_test, test_y = drop_invalid_prices(prep_test, test_y)
    print(regressor.loss(prep_test.to_numpy(), test_y.to_numpy()))

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(prep_train, prep_labels, '../../Ex2/pearson_results')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    regressor = LinearRegression()
    prep_test, test_y = drop_invalid_prices(prep_test, test_y)
    means = []
    stds = []
    for p in range(10, 101):
        fraction = p / 100
        concat_df = pd.concat([prep_train, prep_labels], axis=1)
        loss = []

        for i in range(10):
            sampled_df = concat_df.sample(frac=fraction)
            sampled_features = sampled_df.drop('price', axis=1)
            sampled_labels = sampled_df['price']
            regressor.fit(sampled_features.to_numpy(), sampled_labels.to_numpy())
            loss.append(regressor.loss(prep_test.to_numpy(), test_y.to_numpy()))

        means.append(np.array(loss).mean())
        stds.append(np.array(loss).std())
    means = np.array(means)
    stds = np.array(stds)
    print(stds)

    plot = go.Figure(
        [go.Scatter(
            x=np.arange(10, 101),
            y=means,
            mode='markers'
        ),
            go.Scatter(x=np.arange(10, 101), y=means - 2 * stds, fill=None, mode="lines",
                       line=dict(color="lightgrey"),
                       showlegend=False),
            go.Scatter(x=np.arange(10, 101), y=means + 2 * stds, fill='tonexty', mode="lines",
                       line=dict(color="lightgrey"),
                       showlegend=False)]
    )
    plot.update_layout(
        title='mean loss as function of the percentage of samples from the training set',
        xaxis_title='percentage from training set',
        yaxis_title='mean of loss'
    )

    plot.show()
