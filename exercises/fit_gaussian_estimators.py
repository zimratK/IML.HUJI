from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    univariate_gaussian = UnivariateGaussian()
    q1_fitted = univariate_gaussian.fit(samples)
    print((q1_fitted.mu_.round(3), q1_fitted.var_.round(3)))
    # Question 2 - Empirically showing sample mean is consistent
    results = []
    for i in range(10, 1010, 10):
        fitted = univariate_gaussian.fit(samples[:i])
        results.append(np.abs(fitted.mu_ - 10))
    results = np.array(results)
    plot = go.Figure(
        go.Scatter(
            x=np.arange(10, 1010, 10),
            y=results,
            mode='markers'
        )
    )
    plot.update_layout(
        title='Distance Between Estimated And True Value Of Expectation, As Function Of Sample Size',
        xaxis_title='Sample Size',
        yaxis_title='Deviation of Sample Mean Estimation'
    )
    plot.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = q1_fitted.pdf(samples)
    plot_pdf = go.Figure(
        go.Scatter(
            x=samples,
            y=pdf,
            mode='markers'
        )
    )
    plot_pdf.update_layout(
        title='Empirical PDF Function Under The Fitted Model',
        xaxis_title='Samples',
        yaxis_title='PDF According To The Fitted Model'
    )
    plot_pdf.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean, cov, 1000)
    multivariate_gaussian = MultivariateGaussian()
    q4_fitted = multivariate_gaussian.fit(samples)
    print(q4_fitted.mu_.round(3))
    print(q4_fitted.cov_.round(3))

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    results = []
    for i in range(len(f1)):
        results.append([])
        for j in range(len(f3)):
            log_likelihood = multivariate_gaussian.log_likelihood(np.array([f1[i], 0, f3[j], 0]),
                                                                  cov, samples)
            results[i].append(log_likelihood)
    results = np.array(results)

    plot = go.Figure(
        go.Heatmap(
            x=f3,
            y=f1,
            z=results
        )
    )
    plot.update_layout(
        title='Multivariate Gaussian - Log-Likelihood As Function Of f1 And f3 Values in Expectation Vector',
        xaxis_title='f3 values',
        yaxis_title='f1 values'
    )
    plot.show()

    # Question 6 - Maximum likelihood
    ind = np.unravel_index(np.argmax(results, axis=None), results.shape)
    print(f1[ind[0]].round(3))
    print(f3[ind[1]].round(3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
