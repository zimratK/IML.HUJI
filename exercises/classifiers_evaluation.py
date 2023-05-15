from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def perceptron_callback(model, dummy_x, dummy_y):
            losses.append(model.loss(X, y))

        perceptron = Perceptron(callback=perceptron_callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plot = go.Figure(
            go.Scatter(
                x=np.arange(len(losses)),
                y=losses,
                mode='lines+markers'
            )
        )
        plot.update_layout(
            title='Loss For Each Iteration',
            xaxis_title='iteration',
            yaxis_title='loss'
        )
        plot.show()
        plot.write_image('../../Ex3/loss_by_iterations_' + n + '.png')


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_res_y = lda.predict(X)

        gna = GaussianNaiveBayes()
        gna.fit(X, y)
        gna_res_y = gna.predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        shapes = ['circle', 'square', 'triangle-up']  # circle and square
        colors = ['blue', 'red', 'yellow']
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Plot 1", "Plot 2"))

        # Update the layout
        fig.update_layout(width=800, height=400, title='Two plots side by side')
        # create plot
        for i in range(len(lda.classes_)):
            for j in range(len(lda.classes_)):
                mask = (lda_res_y == i) & (y == j)
                print(mask)
                fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers',
                                         marker=dict(symbol=shapes[j], color=colors[i])), row=1, col=1)

        for i in range(len(lda.classes_)):
            marker = go.scatter.Marker(symbol='x', color='black', size=12)
            fig.add_trace(go.Scatter(x=[lda.mu_[i, 0]], y=[lda.mu_[i, 1]], marker=marker), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=1)

        for i in range(len(gna.classes_)):
            for j in range(len(gna.classes_)):
                mask = (gna_res_y == i) & (y == j)
                print(mask)
                fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers',
                                         marker=dict(symbol=shapes[j], color=colors[i])), row=1, col=2)

        for i in range(len(gna.classes_)):
            marker = go.scatter.Marker(symbol='x', color='black', size=12)
            fig.add_trace(go.Scatter(x=[lda.mu_[i, 0]], y=[lda.mu_[i, 1]], marker=marker), row=1, col=2)
            fig.add_trace(get_ellipse(gna.mu_[i], np.diag(gna.vars_[i])), row=1, col=2)
        fig.update_layout(showlegend=False)
        fig.write_image("../../Ex3/scatter_plot.png")

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()

# TODO unbiased!!!
if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
