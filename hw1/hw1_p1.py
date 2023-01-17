import sys
import math
import argparse

import numpy as np
from numba import njit
# mpl colors: https://matplotlib.org/stable/tutorials/colors/colors.html
# mpl text: https://matplotlib.org/stable/tutorials/text/mathtext.html#symbols
import matplotlib.pyplot as plt
import scipy.stats as ss


@njit
def gaussian(x: np.array, mu: float, sigma: float):
    two_sigma2 = 2*sigma*sigma
    y = np.square(x - mu)
    y = np.exp(y / (-two_sigma2))
    return 1 / math.sqrt(np.pi * two_sigma2) * y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ECE 524 HW1 Problem 1')
    parser.add_argument('-s', '--show', action='store_true', help="Show plots")
    args = parser.parse_args()

    try:
        ######################### PART A #########################
        x = np.linspace(-3, 3, 1000)
        mu = 0
        sigma = 1
        y = gaussian(x, mu, sigma)

        # Plot the data
        fig = plt.figure()
        plt.plot(x, y, "k", linewidth=6)
        plt.title(rf"$\mathcal{{N}}$({mu}, {sigma})")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(f"hw1_p1_a.png")

        ######################### PART B #########################
        mu = 0
        sigma = 1
        data = np.random.default_rng().normal(loc=mu, scale=sigma, size=1000)

        fig = plt.figure()
        plt.hist(data, bins=4, density=True)
        plt.title(rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: 4 bins")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(f"hw1_p1_b1.png")

        fig = plt.figure()
        plt.hist(data, bins=1000, density=True)
        plt.title(rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: 1000 bins")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(f"hw1_p1_b2.png")

        ######################### PART C #########################
        fitted_mu, fitted_sigma = ss.norm.fit(data)
        print(f"Fitted mu: {round(fitted_mu, 3)}, sigma: {round(sigma, 3)}")

        ######################### PART D #########################
        # Ploting fitted curve on top of histograms
        rv = ss.norm(loc=fitted_mu, scale=fitted_sigma)
        x = np.linspace(-3, 3, 1000)
        y = rv.pdf(x)

        fig = plt.figure()
        plt.hist(data, bins=4, density=True)
        plt.plot(x, y, 'k', linewidth=3)
        plt.title(
            rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: 4 bins + fitted")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(f"hw1_p1_d1.png")

        fig = plt.figure()
        plt.hist(data, bins=1000, density=True)
        plt.plot(x, y, 'k', linewidth=3)
        plt.title(
            rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: 1000 bins + fitted")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(f"hw1_p1_d2.png")

    except KeyboardInterrupt as keyerr:
        sys.exit()
