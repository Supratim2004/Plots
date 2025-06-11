import numpy as np
import copy
import math
import random
from tqdm import tqdm
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
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
                Platform_fee_fraction: float,
                Dirichlet_Parameters):
        self.Dirichlet_Parameters=Dirichlet_Parameters
        self.N_experts = N_experts
        self.Means_Lognormal = np.array(Means_Lognormal)
        self.Std_devs_Lognormal = np.array(Std_devs_Lognormal)
        self.Vars_Lognormal = self.Std_devs_Lognormal ** 2
        self.correlation = correlation
        self.N_players = N_players
        self.tau = tau
        self.Analytic_error_bound = Analytic_error_bound
        self.N_analytical_players = int(N_players * tau)
        self.N_random_players = N_players - self.N_analytical_players
        self.Map= self.Generate_Map()
        self.Entry_fee = Entry_fee
        self.Platform_fee_fraction = Platform_fee_fraction
        self.Prize_pool = Entry_fee * N_players * (1 - Platform_fee_fraction)
        self.Captaincy_Multiplier = 2
        self.Vice_Captaincy_Multiplier = 1.5
        # Compute Normal parameters and covariance matrix
        self.Mean_normal, self.Sigma = self.get_multivariate_normal_from_lognormal()
        self.win_prob_team = self.estimate_max_probabilities(n_samples=10000)
    def estimate_max_probabilities(self, n_samples, seed=None):
        # Start with an empty Series with an object index
        d=dict()
        Index=self.Map
        d= {index: 0 for index in Index}
        d=pd.Series(d)
        New=pd.DataFrame(copy.deepcopy(d))
        for n_iter in range(n_samples):
            
            sample = np.random.multivariate_normal(self.Mean_normal, self.Sigma)

            lognormal = np.exp(sample)
            Share=self.True_Fluctuation_of_Share()
            New["Multiplier"]=0
            New["Multiplier"] = New["Multiplier"].astype(float)
            for (i,j,k) in Index:
                share=Share[i-1]
                New.loc[(i,j,k),"Multiplier"] = (self.Captaincy_Multiplier-1)*share[j-1]+(self.Vice_Captaincy_Multiplier-1)*share[k-1]+1
            
            
            for (i,j,k) in Index:
                New.loc[(i,j,k),"Total"] = New.loc[(i,j,k),"Multiplier"] * lognormal[i-1]
            new_max_index= New["Total"].idxmax()
            d[new_max_index] += 1
        d=d/ n_samples
        return d
    def Generate_Map(self):
        Index=[]
        for i in range(1,self.N_experts+1):
            Index.append((i,1,2))
        return Index
    def True_Fluctuation_of_Share(self):
        
        x=list(np.zeros(self.N_experts))
        for i in range(self.N_experts):
            Parameters = self.Dirichlet_Parameters[i]
            Sample=np.random.dirichlet(Parameters, size=1)
            while np.sum(Sample[0][:2] < 1/11)!=0:
                Sample=np.random.dirichlet(Parameters, size=1)
            x[i]=Sample[0][:2]
        return x
    @staticmethod
    def Calculate_Multiplier(Array,tuple):
        Multiplier=2*Array[tuple[0]]+1.5*Array[tuple[1]]+1*(1-Array[tuple[0]]-Array[tuple[1]])
        return Multiplier
            
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
    
    def count_choices(self, arr, N_choices):
        """Count how many times each category is chosen (like np.bincount but with fixed k)."""
        counts = np.zeros(N_choices, dtype=int)
        for i in arr:
            counts[i] += 1
        return pd.Series(counts)
    


    def generate_perturbed_probabilities(self, pi, delta=0.05):
        """
        Generate a perturbed probability vector p ~ Dirichlet(alpha * pi),
        such that with probability ≥ 1 - delta, ||p - pi||_∞ ≤ bound.

        Parameters:
        - pi: base probability vector (must sum to 1)
        - delta: allowable probability of deviation beyond bound (default 0.05)
        - debug: print alpha and deviation info if True

        Returns:
        - p: perturbed probability vector
        """
        pi = np.asarray(pi, dtype=np.float64)
        bound = self.Analytic_error_bound  # target max deviation β
        K = len(pi)

        alpha = required_alpha(K, bound, delta)
        alpha_vector = alpha * pi
        p = np.random.dirichlet(alpha_vector)
        
        return p
                
    def run_simulation(self, Total_iterations=10000):
        Index=self.Map
        Winnings = {i: np.zeros(Total_iterations) for i in Index}
        Total= {i: np.zeros(Total_iterations) for i in Index}
        Analytical={i: np.zeros(Total_iterations) for i in Index}
        Random={i: np.zeros(Total_iterations) for i in Index}
        Total["Name"]="Total"
        Analytical["Name"]="Analytical"
        Random["Name"]="Random"
        for n_iter in tqdm(range(Total_iterations)):
            # Analytical player choices
            Analytical_choices = np.array([
                np.random.choice(len(Index), p=self.generate_perturbed_probabilities(self.win_prob_team))
                for _ in range(self.N_analytical_players)
            ])

            # Random player choices (uniform)
            random_choices = np.random.choice(
                len(Index),
                size=self.N_random_players,
                p=[1 / len(Index)] * len(Index)
            )

            analytical_counts = self.count_choices(Analytical_choices, len(Index))
            analytical_counts.index=Index
            random_counts = self.count_choices(random_choices, len(Index))
            random_counts.index=Index
            total_counts = pd.Series([a + r for a, r in zip(analytical_counts, random_counts)])
            total_counts.index = Index
            for t in Index:
                Total[t][n_iter] = total_counts[t]
                Analytical[t][n_iter] = analytical_counts[t]
                Random[t][n_iter] = random_counts[t]
            win_team = np.random.choice(len(Index), p=self.win_prob_team)
            win_team = Index[win_team]
            for l in Index:
                if total_counts[l] > 0:
                    Winnings[l][n_iter] = (win_team == l) * self.Prize_pool / total_counts[l]
        Data= pd.DataFrame()
        for u in Index:
            Data["Winnings_"+str(u)] = pd.Series(Winnings[u])
            for t in [Analytical,Random,Total]:
                Data[f"{t['Name']}_{u}"] = pd.Series(t[u])
        return Data
