import os
import pathlib

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def image_dir() -> pathlib.Path:
    return pathlib.Path("images") / "problem2"

def eval_hw_gaussian(XX: np.array, YY: np.array):
    N = 2 * np.pi * np.sqrt(3)
    fac = 1/3 * (2 * np.square(XX) + 2 * np.square(YY) - 2 * XX * YY + 4 * XX - 20 * YY + 56)
    return np.exp(fac / -2) / N


def bivariate_gaussian_fast(X: np.array, Y: np.array, mu: np.array, sigma: np.array):
    # https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # Pre-calculate constants
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi)**2 * sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)
    return np.exp(fac / -2) / N


if __name__ == "__main__":
    os.makedirs(image_dir(), exist_ok=True)

    n = 100
    X = np.linspace(-1, 5, n)
    Y = np.linspace(0, 10, n)
    XX, YY = np.meshgrid(X, Y)
    mu = np.array([2, 6]).T
    sigma = np.array([[2, 1], [1, 2]])
    # result = bivariate_gaussian_fast(XX, YY, mu, sigma)
    result = eval_hw_gaussian(XX, YY)

    fig = plt.figure()

    plt.contour(XX, YY, result)
    plt.title(
        rf"2D Gaussian contour plot")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.grid(visible=True)
    fig.savefig(image_dir() / "hw1_p2_a_ii.svg")

    # Plot the surface for fun
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(XX, YY, result, linewidth=1,  cmap=cm.viridis)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    # Eigen decomposition of Sigma_y
    sigma_y = np.array([[2, 1], [1, 2]])
    eig_val, eig_vec = np.linalg.eig(sigma_y)
    # Calculate A = Q * sqrt(Lambda), where Q is the eigenvector matrix and Lambda is the diagonal eigenvalue matrix
    Lambda = np.identity(2) * eig_val.T
    A = np.matmul(eig_vec, np.sqrt(Lambda))
    print(f"Q*sqrt(Delta) = {A}")

    # Verify that we get what is expected
    print(np.matmul(A, A.T))

    # Can also use cholesky decomposition to get A; however, this A will be lower triangular
    # http://www.sefidian.com/2021/12/04/steps-to-sample-from-a-multivariate-gaussian-normal-distribution-with-python-code/
    # A = np.linalg.cholesky(sigma_y)
    # print(A)

    ####################### PART C #######################
    # Generate 2-D guassian samples of zero mean and unit variance
    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    X = np.random.multivariate_normal(mu, cov, 10000).T

    b = np.array([2, 6])
    res = (np.matmul(A, X).T + b).T

    fig = plt.figure()
    plt.scatter(x=X[0], y=X[1], c="k")
    plt.scatter(x=res[0], y=res[1], c="c")
    plt.title(
        rf"Scatter plot of transformed Gaussian data")
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')
    plt.grid(visible=True)
    fig.savefig(image_dir() / "hw1_p2_c_ii.svg")
    plt.show()
