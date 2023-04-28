from enum import Enum
from typing import Callable, List, Tuple

import pymc as pm
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


class AcquisitionFunc(Enum):
    EI = 0,  # Expected improvement
    PI = 1,  # Probability of improvement
    UCP = 2  # Upper confidence bound


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
                 debug=False):
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
        self.bounds = bounds
        # Dimension/# of parameters we are optimizing over
        self.n_param = len(bounds)
        # The acquisition function that will be used during the optimization process
        self.acq_func = acq_func
        # MCMC or OPT mode
        self.mode = mode
        # Debug mode
        self.debug = debug

        if self.n_init < 2:
            raise ValueError("n_init must be greater than 2")

    def _generate_random_x(self):
        params = []
        for bound in self.bounds:
            param_init = np.random.uniform(
                bound[0], bound[1], 1)
            params.append(param_init)
        return np.concatenate(params)

    def _generate_init_input(self, num_samples):
        params = []
        for bound in self.bounds:
            param_init = np.linspace(
                bound[0], bound[1], num_samples).reshape(-1, 1)
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
        print(f"Iter: {i_iter}, Value: {y}, x: {x}")
        return y

    def _expected_improvement(self, mean, std):
        f_best = self.y.min()
        gamma = (f_best - mean) / std
        return std * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

    def _probability_of_improvement(self, mean, std):
        f_best = self.y.min()
        gamma = (f_best - mean) / std
        return norm.cdf(gamma)

    def _acquisition_fn(self, gp: pm.gp.Marginal, point, x: np.array):
        # Get the mean, var for the chosen x value
        x = np.array(x).reshape(-1, 1)
        mean_new, var_new = gp.predict(x, point=point, diag=True)
        # Since we only predicted on point, there should only be one mean/std
        # mean_new = mean_new[0]
        # # Extract the standard deviation from the variance matrix
        # sigma_new = np.sqrt(var_new[0])
        sigma_new = np.sqrt(var_new)

        # Negative sign because scipy.optimize.minimize needs to maximize this function
        if self.acq_func == AcquisitionFunc.EI:
            result = self._expected_improvement(mean_new, sigma_new)
        elif self.acq_func == AcquisitionFunc.PI:
            result = self._probability_of_improvement(mean_new, sigma_new)
        else:
            raise NotImplementedError()

        result = -1 * result
        return result

    def _get_next_probable_point(self, gp: pm.gp.Marginal, point):
        # Define the acquisition function, there is a -1 because we wan't
        # to maximize the function, not minimize
        def acq_func(x): return self._acquisition_fn(gp, point, x)

        if self.debug:
            # Debug code (only tested on 1-D case)
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            # Plot the samples
            ax1.plot(self.x, self.y, "ok", ms=4, label="Data")
            # Plot the posterior prediction means
            x_vals = np.linspace(0, 1, 250).reshape(-1, 1)
            mu, _ = gp.predict(x_vals, point=point, diag=True)
            ax1.plot(x_vals, mu, "r", lw=2, label="Posterior Samples")
            ax1.legend()
            ax1.title.set_text("Model Fitting")
            # Plot the acquisition function and its min
            acq_vals = acq_func(x_vals)
            ax2.plot(x_vals, acq_vals, label="Acquisition Function (*-1)")
            mim_ind = np.argmin(acq_vals)
            ax2.plot(x_vals[mim_ind], acq_vals[mim_ind], "go", label="True Min")
            ax2.title.set_text(f"Acquisition Function: {self.acq_func}")

        # Select a random data point to begin our minimization on
        n_rows = self.x.shape[0]
        row = np.random.randint(0, n_rows-1)
        x0 = self.x[row, :]
        x0 = self._generate_random_x()
        res = minimize(fun=acq_func, x0=x0, bounds=self.bounds)

        if self.debug:
            # Plot the point the minimizer found
            ax2.plot(res.x, res.fun, "ko", label="Minimizer Min")
            ax2.legend()
            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return res.x, res.fun

    def _sample(self) -> np.array:
        '''Sample the surrogate function to get the next x value'''
        model = pm.Model()
        with model:
            # We use the ARD Matern 5/2 kernel here
            ℓ = pm.Gamma("ℓ", alpha=1, beta=0.5)
            # η = pm.HalfCauchy("η", beta=5)
            η = pm.Gamma("η", 1, 1)
            cov = pm.gp.cov.ExpQuad(1, ls=ℓ)
            gp = pm.gp.Marginal(cov_func=cov)
            σ = pm.HalfCauchy("σ", beta=5)
            y_ = gp.marginal_likelihood("y", X=self.x, y=self.y, sigma=σ)

            if self.mode == OptimizerMode.MCMC:
                raise NotImplementedError()
            elif self.mode == OptimizerMode.OPT:
                mp = pm.find_MAP(progressbar=False)
                return self._get_next_probable_point(gp, mp)

    def minimize(self):
        # First call the initialization routine for the specified number of times
        self._init_prior()

        y_min_ind = np.argmin(self.y)
        optimal_y = self.y[y_min_ind]
        optimal_x = self.x[y_min_ind, :]

        for i in range(self.n_iter):
            x_next, _ = self._sample()
            y_next = self._eval_objective_function(i + 1, x_next)

            if x_next in self.x:
                raise RuntimeError("Evaluating same point")
                continue

            # Extend our set of observations
            self.x = np.vstack([self.x, x_next])
            self.y = np.hstack([self.y, y_next])
            if self.x_eval is not None:
                self.x_eval = np.vstack([self.x_eval, x_next])
            else:
                self.x_eval = np.array(x_next)

            # Is this a new minimum?
            if y_next < optimal_y:
                optimal_y = y_next
                optimal_x = x_next

        print(f"Optimal COST: {optimal_y}, PARAMS: {optimal_x}")
        return optimal_y, optimal_x
