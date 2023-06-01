import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy  # TODO is ok??


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):  # TODO 250 learners, 5000 train
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size,
                                                                                           noise)  # TODO uncomment

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)
    # response = adaboost.predict(test_X)
    # print("response", response)
    # print("real", test_y)
    # print(np.sum(np.abs(response-test_y))/2)
    train_errors = [adaboost.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
    test_errors = [adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]
    plot = go.Figure(
        [
            go.Scatter(
                x=np.arange(n_learners),
                y=train_errors,
                mode='lines+markers',
                name='train error'
            ),
            go.Scatter(
                x=np.arange(n_learners),
                y=test_errors,
                mode='lines+markers',
                name='test error'
            )
        ]
    )
    plot.update_layout(
        title='Adaboost Train and Test Errors By Number Of Learners',
        xaxis_title='Number Of Learners',
        yaxis_title='Misclassification Error'
    )
    plot.update_xaxes(range=[0, n_learners])
    plot.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    symbols = np.array(["circle", "x"])
    fig = make_subplots(rows=1, cols=len(T), subplot_titles=[str(t) + " Iterations" for t in T],
                        horizontal_spacing=0.01)
    for i, iterations in enumerate(T):
        def partial(X):
            return adaboost.partial_predict(X, iterations)

        fig.add_traces([decision_surface(partial, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=(test_y > 0).astype(int),
                                               symbol=[symbols[j] for j in (test_y > 0).astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=1, cols=i + 1)
    fig.update_layout(
        title="Decision Surfaces For Different Number Of Iterations",
        title_font=dict(size=20),
        width=1200,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    # fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_index = np.argmin(np.array(test_errors)) + 1
    acc = accuracy(test_y, adaboost.partial_predict(test_X, best_index))
    print(best_index)

    def partial(X):
        return adaboost.partial_predict(X, best_index)

    fig = go.Figure()
    fig.add_traces([decision_surface(partial, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=(test_y > 0).astype(int),
                                           symbol=[symbols[j] for j in (test_y > 0).astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig.update_layout(
        title="Decision Surfaces For Ensemble size " + str(best_index) + ". Accuracy: " + str(acc),
        title_font=dict(size=20),
    )
    # fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    fig.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(size=25 * adaboost.D_ / np.max(adaboost.D_),  # TODO 5
                                           color=(train_y > 0).astype(int),
                                           symbol=[symbols[j] for j in (train_y > 0).astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig.update_layout(
        title="Decision Surfaces For Last Iteration, With Weights",
        width=1000,
        height=1000,
    )
    # fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)  # TODO
    # fit_and_evaluate_adaboost(0.4)
