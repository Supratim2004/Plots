import numpy as np
import math
import random
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
def required_alpha(K, beta, delta):
        """
        Compute the minimum alpha such that
        P(||X - pi||_∞ ≤ beta) ≥ 1 - delta for Dirichlet(alpha * pi)
        """
        return max((K / (4 * beta**2 * delta)) - 1, 1)
class Expert_Team_Selection:
    def __init__(self,
                N_experts: int,
                Means_Lognormal: List[float],
                Std_devs_Lognormal: List[float],
                correlation: float,
                N_players: int,
                tau: float,
                Analytic_error_bound: float,
                Entry_fee: float,
                Platform_fee_fraction: float):
        
        self.N_experts = N_experts
        self.Means_Lognormal = np.array(Means_Lognormal)
        self.Std_devs_Lognormal = np.array(Std_devs_Lognormal)
        self.Vars_Lognormal = self.Std_devs_Lognormal ** 2

        self.N_players = N_players
        self.tau = tau
        self.Analytic_error_bound = Analytic_error_bound
        self.N_analytical_players = int(N_players * tau)
        self.N_random_players = N_players - self.N_analytical_players

        self.Entry_fee = Entry_fee
        self.Platform_fee_fraction = Platform_fee_fraction
        self.Prize_pool = Entry_fee * N_players * (1 - Platform_fee_fraction)

        self.correlation = correlation
        
        # Compute Normal parameters and covariance matrix
        self.Mean_normal, self.Sigma = self.get_multivariate_normal_from_lognormal()
        self.win_prob_team = self.estimate_max_probabilities(self.Mean_normal,self.Sigma,n_samples=1000000)
        
    def lognormal_params_from_mean_var(self, mean, var):
        sigma2 = np.log(1 + var / mean**2)
        mu = np.log(mean) - 0.5 * sigma2
        sigma = np.sqrt(sigma2)
        return mu, sigma

    def equicorrelation_matrix(self, k, rho):
        return np.full((k, k), rho) + np.diag(np.ones(k) * (1 - rho))

    def get_multivariate_normal_from_lognormal(self):
        mu_normal = np.zeros(self.N_experts)
        sigma_normal = np.zeros(self.N_experts)

        for i in range(self.N_experts):
            mu_i, sigma_i = self.lognormal_params_from_mean_var(
                self.Means_Lognormal[i], self.Vars_Lognormal[i]
            )
            mu_normal[i] = mu_i
            sigma_normal[i] = sigma_i

        R = self.equicorrelation_matrix(self.N_experts, self.correlation)
        D = np.diag(sigma_normal)
        Sigma = D @ R @ D

        return mu_normal, Sigma
    def generate_perturbed_probabilities(self,pi, delta=0.05):
        """
        Generate a perturbed probability vector p = pi + epsilon,
        where epsilon ~ Unif(-0.1, 0.1), sum(epsilon) = 0, p_i >= 0, sum(p) = 1.
        
        Parameters:
        - pi: base probability vector of length len(pi), summing to 1
        - max_tries: max number of attempts before giving up

        Returns:
        - p: perturbed probability vector, or None if failed
        """
        """pi = np.array(pi)
        assert np.isclose(np.sum(pi), 1), "pi must sum to 1"
        assert np.all(pi >= 0), "pi must be non-negative"

        for _ in range(max_tries):
            # Sample (N-1) epsilons
            eps = np.random.uniform(-self.Analytic_error_bound, self.Analytic_error_bound, size=len(pi) - 1)    
            # Compute Nth to enforce sum(eps) = 0
            epsN = -np.sum(eps)
            if -self.Analytic_error_bound <= epsN <= self.Analytic_error_bound:
                full_eps = np.append(eps, epsN)
                p = pi + full_eps
                if np.all(p >= 0):
                    return p
        return None  # Failed to find a valid sample"""
        pi = np.asarray(pi, dtype=np.float64)
        bound = self.Analytic_error_bound  # target max deviation β
        K = len(pi)
        alpha = required_alpha(K, bound, delta)
        alpha_vector = alpha * pi
        p = np.random.dirichlet(alpha_vector)
        return p
    def count_choices(self, arr, N_choices):
        """Count how many times each category is chosen (like np.bincount but with fixed k)."""
        counts = np.zeros(N_choices, dtype=int)
        for i in arr:
            counts[i] += 1
        return counts
    def estimate_max_probabilities(self,mean, cov, n_samples=50000, seed=None):
        """
        Estimate P(X_i is the maximum) for i = 1 to k
        where X ~ N(mean, cov)

        Parameters:
        - mean: list or np.array of length k
        - cov: np.array (k x k), covariance matrix
        - n_samples: number of Monte Carlo samples
        - seed: random seed for reproducibility

        Returns:
        - probs: estimated probabilities (array of length k)
        """
        if seed is not None:
            np.random.seed(seed)

        k = len(mean)
        counts = np.zeros(k)

        samples = np.random.multivariate_normal(mean, cov, size=n_samples)

        # Find argmax in each sample
        max_indices = np.argmax(samples, axis=1)

        # Count frequency of each index being max
        for i in range(k):
            counts[i] = np.sum(max_indices == i)

        probs = counts / n_samples
        return probs
    
    def run_simulation(self, Total_iterations=10000):
        Winnings = [[0] * Total_iterations for _ in range(self.N_experts)]
        Total= {i: np.zeros(Total_iterations) for i in range(1, self.N_experts + 1)}
        Analytical={i: np.zeros(Total_iterations) for i in range(1, self.N_experts + 1)}
        Random={i: np.zeros(Total_iterations) for i in range(1, self.N_experts + 1)}
        Total["Name"]="Total"
        Analytical["Name"]="Analytical"
        Random["Name"]="Random"
        for n_iter in tqdm(range(Total_iterations)):
            # Analytical player choices
            Analytical_choices = np.array([
                np.random.choice(self.N_experts, p=self.generate_perturbed_probabilities(self.win_prob_team))
                for _ in range(self.N_analytical_players)
            ])

            # Random player choices (uniform)
            random_choices = np.random.choice(
                self.N_experts,
                size=self.N_random_players,
                p=[1 / self.N_experts] * self.N_experts
            )

            analytical_counts = self.count_choices(Analytical_choices, self.N_experts)
            random_counts = self.count_choices(random_choices, self.N_experts)
            total_counts = np.array([a + r for a, r in zip(analytical_counts, random_counts)])
            for t in range(1,self.N_experts+1):
                Total[t][n_iter] = total_counts[t-1]
                Analytical[t][n_iter] = analytical_counts[t-1]
                Random[t][n_iter] = random_counts[t-1]
            win_team = np.random.choice(self.N_experts, p=self.win_prob_team)
            for l in range(self.N_experts):
                if total_counts[l] > 0:
                    Winnings[l][n_iter] = (win_team == l) * self.Prize_pool / total_counts[l]
        Data= pd.DataFrame()
        for u in range(1,self.N_experts+1):
            Data["Winnings_"+str(u)] = pd.Series(Winnings[u-1])
            for t in [Analytical,Random,Total]:
                Data[f"{t['Name']}_{u}"] = pd.Series(t[u])
        return Data