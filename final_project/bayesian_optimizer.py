import warnings
from enum import Enum
from typing import Callable, List, Tuple

from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor


class AcquisitionFunc(Enum):
    EI = 0,  # Expected improvement
    PI = 1,  # Probability of improvement
    UCB = 2  # Upper confidence bound


class OptimizerMode(Enum):
    MCMC = 0,  # Markov Chain Monte-Carlo
    OPT = 1,   # Maximum-a-posterior


class BayesianOptimizer:
    def __init__(self,
                 objective_function: Callable,
                 n_iter,
                 n_init,
                 bounds: List[Tuple[float, float]],
                 acq_func=AcquisitionFunc.EI,
                 mode=OptimizerMode.OPT,
                 debug=False,
                 verbose=True):
        self.x = None
        self.y = None
        # The x-values that we actually evaluate the function at
        self.x_eval = None
        # The x-values that we use for initialization
        self.x_init = None
        self.objective_function = objective_function
        # Number of iterations used for optimizing
        self.n_iter = n_iter
        # Number of objective functions calls that will be made to initialize the model
        self.n_init = n_init
        # Bounds on the parameter domain
        self.bounds = np.array(
            bounds,
            dtype=float
        )
        # Dimension/# of parameters we are optimizing over
        self.dims = len(bounds)
        # The acquisition function that will be used during the optimization process
        self.acq_func = acq_func
        # MCMC or OPT mode
        self.mode = mode
        # Debug mode
        self.debug = debug
        # Verbose logging mode
        self.verbose = verbose

    def _generate_init_input(self, num_samples):
        params = []
        for bound in self.bounds:
            param_init = np.linspace(
                bound[0], bound[1], num_samples).reshape(num_samples, 1)
            params.append(param_init)
        return np.concatenate(params, axis=1)

    def _init_prior(self):
        # First, we need to generate some random samples for each parameter
        self.x = self._generate_init_input(self.n_init)
        # Copy these to the init variable in case we want to reference them later
        self.x_init = self.x
        # Now for each row of x, we evaluate the objective function
        costs = []
        for i, row in enumerate(self.x):
            cost = self._eval_objective_function(f"Init {i + 1}", row)
            costs.append(cost)
        self.y = np.array(costs)

    def _eval_objective_function(self, i_iter, x):
        y = self.objective_function(*x)
        if self.verbose:
            print(f"Iter: {i_iter}, Value: {y}, x: {x}")
        return y

    def _expected_improvement(self, mean, std, f_best, xi=0.0):
        num = (mean - f_best - xi)
        gamma = num / std
        return num * norm.cdf(gamma) + std * norm.pdf(gamma)

    def _probability_of_improvement(self, mean, std, f_best, xi=0.0):
        return norm.cdf((mean - f_best - xi) / std)

    def _upper_confidence_bound(self, mean, std, kappa=2.5):
        return mean + kappa * std

    def _acquisition_fn(self, model: GaussianProcessRegressor, x: np.array):
        # Get the mean, var for the chosen x value
        x = np.array(x).reshape(-1, self.dims)
        mean_new, std_new = model.predict(x, return_std=True)
        mean_new = mean_new.reshape(x.shape[0], -1)
        std_new = std_new.reshape(x.shape[0], -1)

        # Negative sign because scipy.optimize.minimize needs to maximize this function
        f_best = self.y.max()
        if self.acq_func == AcquisitionFunc.EI:
            return self._expected_improvement(mean_new, std_new, f_best)
        elif self.acq_func == AcquisitionFunc.PI:
            return self._probability_of_improvement(mean_new, std_new, f_best)
        elif self.acq_func == AcquisitionFunc.UCB:
            return self._upper_confidence_bound(mean_new, std_new)
        else:
            raise NotImplementedError()

    def _maximize_acquisition_fn(self, model, bounds, n_warmup, n_iter):
        '''Special handling to maximize the acquisition function'''
        # Just do a warmup with some random sampling
        x_tries = np.random.uniform(self.bounds[:, 0], bounds[:, 1],
                                    size=(n_warmup, bounds.shape[0]))
        ys = self._acquisition_fn(model, x_tries)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()

        # Seed the minimizer with more throughly spaced points
        x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_iter, bounds.shape[0]))

        # Define the function we want to minimize
        def min_acq_fn(x): return -1 * self._acquisition_fn(model, x)

        # Invoke the minimizer
        minimizer_x_tries = []
        minimizer_vals = []
        for x_try in x_seeds:
            res = minimize(fun=min_acq_fn, x0=x_try,
                           bounds=self.bounds, method="L-BFGS-B")

            if not res.success:
                continue

            max_val = -np.squeeze(res.fun)
            if max_val > max_acq:
                x_max = res.x
                max_acq = max_val

            minimizer_x_tries.append(res.x)
            minimizer_vals.append(-res.fun)

        # Clip the value to make sure we are still in bounds
        return np.clip(x_max, bounds[:, 0], bounds[:, 1]), max_acq, \
            x_tries, ys, minimizer_x_tries, minimizer_vals

    def _get_next_probable_point(self, model: GaussianProcessRegressor):
        # Define the acquisition function, there is a -1 because we wan't
        # to maximize the function, not minimize
        def acq_func(x): return self._acquisition_fn(model, x)

        if self.debug:
            # Debug code (only tested on 1-D case)
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            # Plot the samples
            ax1.plot(self.x, self.y, "ok", ms=4, label="Data")
            # Plot the posterior prediction means
            x_vals = np.linspace(self.bounds[0][0], self.bounds[0][1], 500).reshape(-1, self.dims)
            mu, sd = model.predict(x_vals, return_std=True)
            mu = mu.reshape(x_vals.shape).flatten()
            sd = sd.reshape(x_vals.shape).flatten()

            ax1.plot(x_vals, mu, "r", lw=2, label="Posterior Samples")
            ax1.plot(x_vals, mu + 2 * sd, "r", lw=1)
            ax1.plot(x_vals, mu - 2 * sd, "r", lw=1)
            ax1.fill_between(x_vals.flatten(), mu - 2 * sd, mu + 2 * sd, color="r", alpha=0.5)
            ax1.legend()
            ax1.title.set_text("Model Fitting")
            # Plot the acquisition function and its min
            acq_vals = acq_func(x_vals)
            ax2.plot(x_vals, acq_vals, label="Acquisition Function")
            mim_ind = np.argmax(acq_vals)
            ax2.plot(x_vals[mim_ind], acq_vals[mim_ind], "bo", label="True Max")
            ax2.title.set_text(f"Acquisition Function: {self.acq_func}")

        res = self._maximize_acquisition_fn(model, self.bounds, 3000, 4)

        if self.debug:
            # Plot the point the minimizer found
            ax2.plot(res[2], res[3], "go", ms=3, alpha=0.3, label="Maximize Warmup Tries")
            ax2.plot(res[4], res[5], "ro", ms=3, alpha=0.6, label="Optimizer Tries")
            ax2.plot(res[0], res[1], "ko", label="Calculated Max")
            ax2.legend()
            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return res[0], res[1]

    def _sample(self) -> np.array:
        '''Sample the surrogate function to get the next x value'''
        # Ignore sklearn's GP warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianProcessRegressor(
                kernel=gp.kernels.Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )
            model.fit(self.x, self.y)
            return self._get_next_probable_point(model)

    def maximize(self):
        # First call the initialization routine for the specified number of times
        self._init_prior()

        y_max_ind = np.argmax(self.y)
        optimal_y = self.y[y_max_ind]
        optimal_x = self.x[y_max_ind, :]

        for i in range(self.n_iter):
            x_next, _ = self._sample()
            y_next = self._eval_objective_function(i + 1, x_next)

            # Extend our set of observations
            self.x = np.vstack([self.x, x_next])
            self.y = np.hstack([self.y, y_next])
            if self.x_eval is not None:
                self.x_eval = np.vstack([self.x_eval, x_next])
            else:
                self.x_eval = np.array(x_next)

            # Is this a new minimum?
            if y_next > optimal_y:
                optimal_y = y_next
                optimal_x = x_next

        if self.verbose:
            print(f"Optimal VALUE: {optimal_y}, PARAMS: {optimal_x}")
        return optimal_y, optimal_x
