import os
import sys
from typing import Callable, List, Tuple

from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np
from prettytable import PrettyTable


class BayesianOptimizer:
    def __init__(self,
                 objective_function: Callable,
                 n_iter,
                 n_init,
                 bounds: List[Tuple[float, float]],
                 scale,
                 batch_size):
        self.x_init = None
        self.y_init = None
        self.objective_function = objective_function
        self.n_iter = n_iter
        self.n_init = n_init
        self.scale = scale
        self.bounds = bounds
        self.n_param = len(bounds)
        self.batch_size = batch_size
        self.gauss_pr = GaussianProcessRegressor()
        self.best_samples_ = pd.DataFrame(columns=['x', 'y', 'ei'])
        self.distances_ = []

        # Setup the table we are going to print out
        self.table = PrettyTable()
        self.table.float_format = ".3"

    def _generate_random_input(self, num_samples):
        params = []
        for bound in self.bounds:
            param_init = np.random.uniform(
                bound[0], bound[1], num_samples).reshape(num_samples, 1)
            params.append(param_init)
        return np.concatenate(params, axis=1)

    def _init_prior(self):
        # First, we need to generate some random samples for each parameter
        self.x_init = self._generate_random_input(self.n_init)
        # Now for each row of x, we evaluate the objective function
        costs = []
        for row in self.x_init:
            cost = self.objective_function(*row)
            costs.append(cost)
        self.y_init = np.array(costs).reshape(self.n_init, 1)

    def _init_table(self):
        # Print the header
        iter_col = [f"init {x + 1}" for x in range(self.n_init)]
        self.table.add_column("N", iter_col)
        self.table.add_column("COST", self.y_init[:, 0].tolist())
        for i in range(self.n_param):
            self.table.add_column(f"x{i}", self.x_init[:, i].tolist())

    def _expected_improvement(self, x_new):
        # Using estimate from Gaussian surrogate instead of actual function for
        # a new trial data point to avoid cost
        mean_y_new, sigma_y_new = self.gauss_pr.predict(np.array([x_new]), return_std=True)
        sigma_y_new = sigma_y_new.reshape(-1, 1)
        if sigma_y_new == 0.0:
            return 0.0

        # Using estimates from Gaussian surrogate instead of actual function for
        # entire prior distribution to avoid cost
        surrogate_mean = self.gauss_pr.predict(self.x_init)
        max_surrogate_mean = np.max(surrogate_mean)
        gamma = (mean_y_new - max_surrogate_mean) / sigma_y_new
        eip = sigma_y_new * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        # exp_imp = (mean_y_new - max_surrogate_mean) * norm.cdf(z) + sigma_y_new * norm.pdf(z)

        return eip

    def _acquisition_fn(self, x):
        # Negative sign because scipy.optimize.minimize needs to maximize this function
        return -1 * self._expected_improvement(x)

    def _get_next_probable_point(self):
        min_ei = float(sys.maxsize)
        x_optimal = None

        # Trial with an array of random data points using BFGS algorithm
        batch = self._generate_random_input(self.batch_size)
        for x_start in batch:
            response = minimize(fun=self._acquisition_fn, x0=x_start,
                                bounds=self.bounds, method='L-BFGS-B')
            if response.fun < min_ei:
                min_ei = response.fun
                x_optimal = response.x
        return x_optimal, min_ei

    def _extend_prior_with_posterior_data(self, x, y):
        self.x_init = np.append(self.x_init,
                                np.array(x).reshape(1, len(self.bounds)), axis=0)
        self.y_init = np.append(self.y_init,
                                np.array([y]).reshape(1, 1), axis=0)

    def _add_iteration_to_table(self, iter, x: np.array, y):
        # Evaluate objective function, return the results and pretty print a summary
        row = [iter]
        row.append(y)
        row.extend(x.tolist())
        self.table.add_row(row)

    def maximize(self):
        # First call the initialization routine for the specified number of times
        self._init_prior()
        # Initialize the pretty formatted table
        self._init_table()
        y_max_ind = np.argmax(self.y_init)
        y_max = self.y_init[y_max_ind]
        optimal_x = self.x_init[y_max_ind]

        optimal_ei = None
        for i in range(self.n_iter):

            self.gauss_pr.fit(self.x_init, self.y_init)

            x_next, ei = self._get_next_probable_point()

            y_next = self.objective_function(*x_next)

            self._add_iteration_to_table(i + 1, x_next, y_next)

            self._extend_prior_with_posterior_data(x_next, y_next)

            if y_next > y_max:
                y_max = y_next
                optimal_x = x_next
                optimal_ei = ei

            if i == 0:
                prev_x = x_next
            else:
                self.distances_.append(np.linalg.norm(prev_x - x_next))
                prev_x = x_next

            df = pd.DataFrame({"y": y_max, "ei": optimal_ei}, index=[0])
            self.best_samples_ = pd.concat([self.best_samples_, df], ignore_index=True)

        self.optimal = (y_max, optimal_x)
        return optimal_x, y_max

    def print_results(self):
        # Print the table we have constructed
        print(self.table)
        # Print the optimal values
        print(f"Optimal COST: {self.optimal[0]}, PARAMS: {self.optimal[1]}")
