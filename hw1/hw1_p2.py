import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


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


def bivariate_gaussian(x_vec: np.array, y_vec: np.array, mu: np.array, sigma: np.array):
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi)**2 * sigma_det)

    data = []
    for x in x_vec:
        for y in y_vec:
            X = np.array((x, y)).T
            fac = np.matmul((X-mu).T, np.matmul(sigma_inv, (X-mu)))
            res = np.exp(fac / -2) / N
            data.append(res)
    return np.reshape(data, [x_vec.size, y_vec.size])


n = 100
X = np.linspace(-1, 5, n)
Y = np.linspace(0, 10, n)
XX, YY = np.meshgrid(X, Y)
mu = np.array([2, 6]).T
sigma = np.array([[2, 1], [1, 2]])
# result = bivariate_gaussian(X, Y, mu, sigma)

result = bivariate_gaussian_fast(XX, YY, mu, sigma)


fig = plt.figure()

plt.contour(XX, YY, result)
# plt.title(
#     rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: {optimal_m} bins + fitted")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.grid(visible=True)
# if args.show:
plt.show()
fig.savefig(f"hw1_p2_aii.png")
plt.close()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(XX, YY, result, linewidth=1,  cmap=cm.viridis)
# ax.plot_surface(XX, YY, result, rstride=3, cstride=3,
#                 linewidth=1, antialiased=True, cmap=cm.viridis)
# ax.view_init(55, -70)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.show()
plt.close()
