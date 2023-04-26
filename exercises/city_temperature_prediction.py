import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df[df['Temp'] > -30].dropna().drop_duplicates()
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df['Country'] == 'Israel']

    color_map = dict(zip(pd.Categorical(israel_df['Year']).categories, px.colors.qualitative.Plotly))
    fig = px.scatter(israel_df, x="DayOfYear", y="Temp", color=israel_df['Year'].astype(str),
                     color_discrete_map=color_map)
    fig.update_layout(
        title='Temperature in israel as function of DayOfYear',
        xaxis_title="Day Of Year",
        yaxis_title='Temperature',
        legend={
            'title': 'Year'
        }
    )

    fig.write_image('../../Ex2/temp_by_day.png')

    month_std = israel_df.groupby('Month').agg({'Temp': ['std']})
    month_std = month_std.reset_index()
    fig = px.bar(x=np.arange(1,13), y=month_std[('Temp', 'std')], text=np.array(month_std[('Temp', 'std')]).round(2))
    fig.update_layout(
        title='Standard Deviation As Function Of Month',
        xaxis_title="Month",
        yaxis_title='Temperature Standard Deviation'
    )
    fig.write_image('../../Ex2/std_by_month.png')

    # Question 3 - Exploring differences between countries
    month_stats = df.groupby(['Month', 'Country'], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std"))
    color_map = dict(zip(pd.Categorical(month_stats["Country"]).categories, px.colors.qualitative.Plotly))

    fig = px.line(month_stats, title='Mean Of Daily Temperature By Month And Country', x='Month', y='mean',
                  error_y='std', color='Country')
    fig.update_layout(
        # title='Mean Of Daily Temperature By Month And Country ',
        xaxis_title='Month',
        yaxis_title='Mean Of Daily Temperature'
    )
    fig.write_image('../../Ex2/by_country.png')


    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df.drop('Temp', axis=1), israel_df["Temp"])
    loss = []
    for k in range(1, 11):
        poly_regressor = PolynomialFitting(k)
        poly_regressor.fit(train_X["DayOfYear"].to_numpy(), train_y.to_numpy())
        loss.append(poly_regressor.loss(test_X["DayOfYear"].to_numpy(), test_y.to_numpy()))
    loss = np.array(loss).round(2)

    fig = px.bar(x=np.arange(1, 11), y=loss, text=loss)
    fig.update_layout(
        title='Loss As Function Of K Value (Polynom Degree)',
        xaxis_title="K",
        yaxis_title='Loss'
    )
    fig.write_image('../../Ex2/loss_by_k.png')

    # Question 5 - Evaluating fitted model on different countries
    poly_regressor = PolynomialFitting(5)
    poly_regressor.fit(israel_df["DayOfYear"].to_numpy(), israel_df["Temp"].to_numpy())
    countries = df.loc[df['Country'] != 'Israel', 'Country'].unique()
    loss = []
    for country_name in countries:
        country_df = df[df['Country'] == country_name]
        country_loss = poly_regressor.loss(country_df["DayOfYear"].to_numpy(), country_df["Temp"].to_numpy())
        loss.append(country_loss)
    fig = px.bar(x=countries, y=loss, text=np.array(loss).round(2))
    fig.update_layout(
        title='Loss For Each country, k=5',
        xaxis_title="Country Name",
        yaxis_title='Loss'
    )
    fig.write_image('../../Ex2/loss_by_country.png')



