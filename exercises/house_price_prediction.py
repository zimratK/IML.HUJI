from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime

pio.templates.default = "simple_white"
MIN_ROOM_SIZE = 50
avg_dict = {}

train_columns = []


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
    X = X.drop('date', axis=1)
    X = X.drop('id', axis=1)
    X = X.replace('nan', np.nan)

    X, y = handle_zipcode(X, y)
    X = X.apply(pd.to_numeric)

    # yr_renovated - if 0, replace by value of yr_built
    X.loc[X['yr_renovated'] == 0, 'yr_renovated'] = X.loc[X['yr_renovated'] == 0, 'yr_built']

    if y is not None:
        X, y = drop_invalid_prices(X, y)

    handle_missing_data(X)
    X['dist_from_center'] = X.apply(distance_from_center, axis=1)
    X = X.drop('long', axis=1)
    X = X.drop('lat', axis=1)

    if y is not None:
        concat_df = pd.concat([X, y], axis=1)
        concat_df = concat_df.drop_duplicates()
        X = concat_df.drop("price", axis=1)
        y = concat_df["price"]
        return X, y
    return X


def handle_missing_data(X):
    for col in X.columns:
        if col.startswith('zipcode'):
            continue
        X.loc[X[col] < 0, col] = 0
        if col not in ['sqft_basement', 'view', 'waterfront']:
            if (X[col] <= 0).any():
                X.loc[X[col] == 0, col] = max((avg_dict[col], 0))
            if col.startswith('sqft'):
                X.loc[X[col] < MIN_ROOM_SIZE, col] = MIN_ROOM_SIZE
        if X[col].isnull().any():
            X[col].fillna(max((avg_dict[col], 0)), inplace=True)


def handle_zipcode(X, y):
    if y is not None:
        y = y.replace('nan', np.nan)
        global avg_dict
        avg_dict = {col: pd.to_numeric(X[col]).mean() for col in X.columns}
        X = pd.get_dummies(X, columns=['zipcode'])
        global train_columns
        train_columns = X.columns
        y = y.apply(pd.to_numeric)
    else:
        X = pd.get_dummies(X, columns=['zipcode'])
        X = X.reindex(columns=train_columns, fill_value=0)
    return X, y


def distance_from_center(row):
    long_sqr = (row["long"] - avg_dict["long"]) ** 2
    lat_sqr = (row["lat"] - avg_dict["lat"]) ** 2
    return np.sqrt(long_sqr + lat_sqr)


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
        pearson = 0
        if not x_std * y_std == 0:
            pearson = covariance / (x_std * y_std)
        plot = go.Figure(
            go.Scatter(
                x=X[feature],
                y=y,
                mode='markers'
            )
        )
        plot.update_layout(
            title='Price As Function Of ' + feature + ', Pearson Correlation: ' + str(pearson),
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


def plot_distance_price(X, y):
    global plot
    plot = go.Figure(
        go.Scatter(
            x=X['dist_from_center'],
            y=y,
            mode='markers'
        )
    )
    plot.update_layout(
        title='Price As Function Of Distance From The Center',
        xaxis_title="distance",
        yaxis_title='price'
    )
    plot.write_image('./price_distance_correlation.png')


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
    prep_test = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(prep_train, prep_labels, '.Ex2/pearson_results')

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
        loss_arr = []
        i = 0
        while i < 10:
            regressor = LinearRegression()
            sampled_df = concat_df.sample(frac=fraction)
            sampled_features = sampled_df.drop('price', axis=1)
            sampled_labels = sampled_df['price']
            regressor.fit(sampled_features.to_numpy(), sampled_labels.to_numpy())
            current_loss = regressor.loss(prep_test.to_numpy(), test_y.to_numpy())
            if current_loss < 1e11:
                i = i + 1
                loss_arr.append(current_loss)
        means.append(np.array(loss_arr).mean())
        stds.append(np.array(loss_arr).std())
    means = np.array(means)
    stds = np.array(stds)

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

    plot.write_image('.loss_by_percentage.png')

    # TODO replace all paths
