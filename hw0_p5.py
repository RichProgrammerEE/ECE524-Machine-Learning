import sys

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    x = np.linspace(-5, 5, 1000)
    y = 1 / (1 + np.exp(-5 * (x - 1)))


    try:
        # Plot the data
        plt.figure(figsize=[12, 6])
        # colors: https://matplotlib.org/stable/tutorials/colors/colors.html
        plt.plot(x, y, "k", linewidth=6)

        plt.title(f"my plot")
        # plt.legend()
        plt.ylabel("f(x)")
        plt.xlabel("x")
        # plt.grid(visible=True)
        plt.show()
    except KeyboardInterrupt as keyerr:
        sys.exit()
