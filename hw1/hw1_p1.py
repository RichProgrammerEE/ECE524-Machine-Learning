import sys
import os
import math
import pathlib
import argparse

import numpy as np
# mpl colors: https://matplotlib.org/stable/tutorials/colors/colors.html
# mpl text: https://matplotlib.org/stable/tutorials/text/mathtext.html#symbols
import matplotlib.pyplot as plt
import scipy.stats as ss


def image_dir() -> pathlib.Path:
    return pathlib.Path("images") / "problem1"


def gaussian(x: np.array, mu: float, sigma: float):
    two_sigma2 = 2*sigma*sigma
    y = np.square(x - mu)
    y = np.exp(y / (-two_sigma2))
    return 1 / math.sqrt(np.pi * two_sigma2) * y


def cross_validation_estimator_of_risk(data: np.array, m: int):
    '''m = number of histogram bins'''
    h = (np.max(data) - np.min(data)) / m  # Bin width
    n = data.size  # Number of data points
    # Calculate the histogram so we can get the empirical probabilities
    hist, _ = np.histogram(data, bins=m)
    # Empirical probability of value falling into specific bin
    p_hat = np.divide(hist, n)
    denom = h * (n-1)
    j_hat = 2 / denom - (n+1) / denom * np.sum(np.square(p_hat))
    return j_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ECE 524 HW1 Problem 1')
    parser.add_argument('-s', '--show', action='store_true', help="Show plots")
    args = parser.parse_args()

    os.makedirs(image_dir(), exist_ok=True)

    try:
        ######################### PART A #########################
        x = np.linspace(-3, 3, 1000)
        mu = 0
        sigma = 1
        y = gaussian(x, mu, sigma)

        # Plot the data
        fig = plt.figure()
        plt.plot(x, y, "k", linewidth=3)
        plt.title(rf"$\mathcal{{N}}$({mu}, {sigma})")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(image_dir() / "hw1_p1_a.svg")
        plt.close()

        ######################### PART B #########################
        # i
        mu = 0
        sigma = 1
        data = np.random.default_rng().normal(loc=mu, scale=sigma, size=1000)

        # ii
        # Figures plotted here are same as in part iv

        # iii
        fitted_mu, fitted_sigma = ss.norm.fit(data)
        print(f"Fitted mu: {round(fitted_mu, 3)}, sigma: {round(sigma, 3)}")

        # iv
        # Ploting fitted curve on top of histograms
        rv = ss.norm(loc=fitted_mu, scale=fitted_sigma)
        x = np.linspace(-3, 3, 1000)
        y = rv.pdf(x)

        fig = plt.figure()
        plt.hist(data, bins=15, density=True)
        plt.plot(x, y, 'k', linewidth=3)
        plt.title(
            rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: 15 bins + fitted")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(image_dir() / "hw1_p1_b_iv_1.svg")
        plt.close()

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
        fig.savefig(image_dir() / "hw1_p1_b_iv_2.svg")
        plt.close()

        ######################### PART C #########################
        m_vals = []
        for i in range(50):
            mu = 0
            sigma = 1
            data = np.random.default_rng().normal(loc=mu, scale=sigma, size=1000)
            j_hats = []
            ms = range(1, 201)
            for m in ms:
                j_hats.append(cross_validation_estimator_of_risk(data, m))

            optimal_m = np.argmin(j_hats) + 1  # Indicies are zero-based
            m_vals.append(optimal_m)
            print(f"Optimal number of bins: {optimal_m}")

        # print(m_vals)
        print(sum(m_vals) / len(m_vals))

        fig = plt.figure()
        plt.plot(ms, j_hats, 'k', linewidth=3)
        plt.title(
            rf"$\mathcal{{\hat J(h)}}$ vs. # of bins")
        plt.ylabel("$\mathcal{{\hat J(h)}}$")
        plt.xlabel("m")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(image_dir() / "hw1_p1_c_i.svg")
        plt.close()

        fig = plt.figure()
        plt.hist(data, bins=optimal_m, density=True)
        plt.plot(x, y, 'k', linewidth=3)
        plt.title(
            rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: {optimal_m} bins + fitted")
        plt.ylabel("f(x)")
        plt.xlabel("x")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(image_dir() / "hw1_p1_c_ii.svg")
        plt.close()

    except KeyboardInterrupt as keyerr:
        sys.exit()
