import sys
import math

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x: np.array, mu: float, sigma: float):
    two_pi_sigma2 = 2*np.pi*sigma*sigma
    y = np.square(x - mu)
    y = np.exp(y / (-two_pi_sigma2))
    return 1 / math.sqrt(two_pi_sigma2) * y


if __name__ == "__main__":

    x = np.linspace(-5, 5, 1000)
    y = gaussian(x, mu=0, sigma=1)

    try:
        # Plot the data
        fig = plt.figure(figsize=[12, 6])
        # colors: https://matplotlib.org/stable/tutorials/colors/colors.html
        plt.plot(x, y, "k", linewidth=6)
        # https://matplotlib.org/stable/tutorials/text/mathtext.html#symbols
        plt.title(r"$\mathcal{N}")
        # plt.legend()
        plt.ylabel("f(x)")
        plt.xlabel("x")
        # plt.grid(visible=True)
        plt.show()
        fig.savefig(f"hw1_p1_a.png")
    except KeyboardInterrupt as keyerr:
        sys.exit()
