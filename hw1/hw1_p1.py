import sys
import math
import argparse

import numpy as np
# mpl colors: https://matplotlib.org/stable/tutorials/colors/colors.html
# mpl text: https://matplotlib.org/stable/tutorials/text/mathtext.html#symbols
import matplotlib.pyplot as plt
import scipy.stats as ss


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
        fig.savefig(f"hw1_p1_a.png")
        plt.close()

        ######################### PART B #########################
        # i
        mu = 0
        sigma = 1
        data = np.random.default_rng().normal(loc=mu, scale=sigma, size=1000)

        # ii
        # fig = plt.figure()
        # plt.hist(data, bins=4, density=True)
        # plt.title(rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: 4 bins")
        # plt.ylabel("f(x)")
        # plt.xlabel("x")
        # plt.grid(visible=True)
        # if args.show:
        #     plt.show()
        # fig.savefig(f"hw1_p1_bii_1.png")

        # fig = plt.figure()
        # plt.hist(data, bins=1000, density=True)
        # plt.title(rf"$\mathcal{{N}}$({mu}, {sigma}) histogram: 1000 bins")
        # plt.ylabel("f(x)")
        # plt.xlabel("x")
        # plt.grid(visible=True)
        # if args.show:
        #     plt.show()
        # fig.savefig(f"hw1_p1_bii_2.png")

        # iii
        fitted_mu, fitted_sigma = ss.norm.fit(data)
        print(f"Fitted mu: {round(fitted_mu, 3)}, sigma: {round(sigma, 3)}")

        # iv
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
        fig.savefig(f"hw1_p1_biv_1.png")
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
        fig.savefig(f"hw1_p1_biv_2.png")
        plt.close()

        ######################### PART C #########################
        j_hats = []
        ms = range(1, 201)
        for m in ms:
            j_hats.append(cross_validation_estimator_of_risk(data, m))

        optimal_m = np.argmin(j_hats) + 1  # Indicies are zero-based
        print(f"Optimal number of bins: {optimal_m}")

        fig = plt.figure()
        plt.plot(ms, j_hats, 'k', linewidth=3)
        plt.title(
            rf"$\mathcal{{\hat J(h)}}$ vs. # of bins")
        plt.ylabel("$\mathcal{{\hat J(h)}}$")
        plt.xlabel("m")
        plt.grid(visible=True)
        if args.show:
            plt.show()
        fig.savefig(f"hw1_p1_ci.png")
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
        fig.savefig(f"hw1_p1_cii.png")
        plt.close()

    except KeyboardInterrupt as keyerr:
        sys.exit()
