from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

h = 1


def kernel(x1, x2):
    return np.exp(-np.sum((x1 - x2)**2) / h)


if __name__ == "__main__":
    base_dir = Path().cwd() / "hw4" / "data" / "q4_2"
    class0 = np.loadtxt(base_dir / "quiz4_class0.txt")
    class1 = np.loadtxt(base_dir / "quiz4_class1.txt")

    A = np.vstack([class0, class1])
    X = np.hstack([A, np.ones((A.shape[0], 1))])
    y = np.vstack([np.zeros([class0.shape[0], 1]),
                  np.ones([class1.shape[0], 1])])
    y = y.reshape((100, 1))
    N = X.shape[0]

    # Construct kernel matrix K
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j])

    print(K[47:52, 47:52])

    # Use CVXPY to minimize regularized logistic loss function + kernel trick
    lambd = 0.01
    alpha_cp = cp.Variable((N, 1))
    # y.T @ K @ alpha_cvx
    loss = cp.sum(cp.multiply(y, cp.psd_wrap(K) @ alpha_cp)) \
        - cp.sum(cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), K @ alpha_cp]), axis=1))
    reg = cp.quad_form(alpha_cp, cp.psd_wrap(K))
    prob = cp.Problem(cp.Minimize((-1/N) * loss + lambd * reg))

    prob.solve()
    alpha = alpha_cp.value
    print(alpha[0:2])

    def make_decision_kernel(x0, x1) -> int:
        '''Returns 0/1 depending on classification'''
        x = np.array([x0, x1, 1]).reshape((3, 1))
        sum = 0.0
        for i in range(N):
            data = X[i].reshape((3, 1))
            sum += alpha[i][0] * kernel(data, x)
        return 1 / (1 + np.exp(-sum))

    xset = np.linspace(-5, 10, 100)
    yset = np.linspace(-5, 10, 100)
    XA, YA = np.meshgrid(xset, yset)
    vfunc = np.vectorize(make_decision_kernel)
    Z = vfunc(XA, YA)

    fig = plt.figure()
    plt.scatter(class0[:, 0], class0[:, 1], color="b")
    plt.scatter(class1[:, 0], class1[:, 1], color="r")
    plt.contour(XA, YA, Z > 0.5)
    plt.title("Contour Plot of Logistic Regression + Kernel Trick")
    plt.xlabel("class 0")
    plt.ylabel("class 1")
    plt.show()
