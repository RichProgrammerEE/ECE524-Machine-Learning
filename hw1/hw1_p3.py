import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
from scipy.optimize import linprog


def image_dir() -> pathlib.Path:
    return pathlib.Path("images") / "problem3"


if __name__ == "__main__":
    os.makedirs(image_dir(), exist_ok=True)

    ######################### PART A #########################
    n = 50
    x = np.linspace(-1, 1, n)  # 50 points in the interval [-1, 1]
    beta = [-0.001, 0.01, 0.55, 1.5, 1.2]
    epsilon = np.random.default_rng().normal(loc=0, scale=0.2**2, size=n)

    y = epsilon

    for i in range(len(beta)):
        y += beta[i] * eval_legendre(i, x)

    fig = plt.figure()
    plt.scatter(x=x, y=y, c="b")
    plt.title(
        rf"Scatter plot Legendre model data")
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    plt.grid(visible=True)
    plt.show()

    ######################### PART C #########################
    # legendre polynomial data matrix
    A_leg = np.column_stack([eval_legendre(i, x) for i in range(5)])
    # print(A_leg.shape)

    # Let numpy do the heavy lifting and calculate the least-squares solution
    theta = np.linalg.lstsq(A_leg, y, rcond=None)[0]
    # print(theta)

    # Generate the predicted curve (use a more dense x so that plot looks better)
    x_p = np.linspace(-1, 1, n*5)
    yhat = np.zeros(x_p.size)
    for i in range(len(theta)):
        yhat += theta[i] * eval_legendre(i, x_p)

    fig = plt.figure()
    plt.plot(x, y, 'o', c="b", markersize=8)
    plt.plot(x_p, yhat, c="k", linewidth=4)
    plt.title('Legendre Polynomial fitted model (L2 Norm)')
    plt.grid(visible=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    fig.savefig(image_dir() / "hw1_p3_a+c.svg")
    plt.show()

    ######################### PART D #########################
    # Add outliers to y
    idx = [10, 16, 23, 37, 45]
    y[idx] = 5

    # Let numpy do the heavy lifting and calculate the least-squares solution
    theta = np.linalg.lstsq(A_leg, y, rcond=None)[0]
    # print(theta)

    # Generate the predicted curve (use a more dense x so that plot looks better)
    x_p = np.linspace(-1, 1, n*5)
    yhat = np.zeros(x_p.size)
    for i in range(len(theta)):
        yhat += theta[i] * eval_legendre(i, x_p)

    fig = plt.figure()
    plt.plot(x, y, 'o', c="b", markersize=8)
    plt.plot(x_p, yhat, c="k", linewidth=4)
    plt.title('Legendre Polynomial fitted model (L2 Norm) with outliers')
    plt.grid(visible=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    fig.savefig(image_dir() / "hw1_p3_d.svg")
    plt.show()

    ######################### PART F #########################
    # scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
    #                        method='highs', callback=None, options=None, x0=None, integrality=None)

    # A is Nxd
    # C is (d+n)x1 ==> [00000 111111].T
    N = A_leg.shape[0]
    d = A_leg.shape[1]
    c = np.concatenate([np.zeros(d), np.ones(N)])
    # print(c.shape)

    # A_ub should be 2N x (d + N) ==> 2*50 x 5 + 50 = 100 x 55
    A_ub = np.concatenate([
        np.concatenate([A_leg, -A_leg]),
        np.concatenate([-np.identity(N), -np.identity(N)])], axis=1)
    # print(A_ub.shape)

    # b_ub = [y -y].T, should be 2N x 1
    b_ub = np.concatenate([y, -y])
    # print(b_ub.shape)

    # Call the optimization routine
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))

    # The first d results from the linear programming solver is our theta
    theta = result.x[0:5]
    # print(theta)

    # Generate the predicted curve (use a more dense x so that plot looks better)
    x_p = np.linspace(-1, 1, n*5)
    yhat = np.zeros(x_p.size)
    for i in range(len(theta)):
        yhat += theta[i] * eval_legendre(i, x_p)

    fig = plt.figure()
    plt.plot(x, y, 'o', c="b", markersize=8)
    plt.plot(x_p, yhat, c="k", linewidth=4)
    plt.title('Legendre Polynomial fitted model (L1 Norm) with outliers')
    plt.grid(visible=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    fig.savefig(image_dir() / "hw1_p3_f.svg")
    plt.show()
