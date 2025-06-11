import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import time
import numpy as np
from Team_Win_Analysis_Runner_withcaptain_vice import *
from pathlib import Path
from Team_Win_Analysis_Runner_withcaptain_vice import *
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import os
from plotly.colors import qualitative

# Alternative version with even more granular parallelization
def process_single_simulation(args):
    """
    Process a single simulation run (even more granular parallelization)
    """
    (correlation, Tau, Error_Bound, Dirichlet_Parameters_Name, Config,
     Dirichlet_Parameters_Configurations, N_experts, N_players,
     Entry_fee, Platform_fee_fraction, N_iterations, folder) = args
    
    tau = Tau
    Analytic_Error_Bound = Error_Bound
    
    # Define the specific configuration
    config_templates = {
        "Equi-mean_Equivariance": {
            "Means_Lognormal": [500, 500, 500,500,500],
            "Std_devs_Lognormal": [60, 60, 60,60,60]
        },
        "Unequal-mean_Equivariance": {
            "Means_Lognormal": [440, 473,517,550,583],
            "Std_devs_Lognormal": [60, 60,60, 60,60]
        },
        "Equi-mean_Unequal-std": {
            "Means_Lognormal": [500, 500,500, 500,500],
            "Std_devs_Lognormal": [60, 30, 60,30,90]
        },
        "Unequal-mean_Unequal-std": {
            "Means_Lognormal": [440, 473,517, 550,583],
            "Std_devs_Lognormal": [60, 30, 60,30,90]
        }
    }
    
    config_params = {
        "N_experts": N_experts,
        "Means_Lognormal": config_templates[Config]["Means_Lognormal"],
        "Std_devs_Lognormal": config_templates[Config]["Std_devs_Lognormal"],
        "N_players": N_players,
        "tau": tau,
        "Entry_fee": Entry_fee,
        "Platform_fee_fraction": Platform_fee_fraction,
        "correlation": correlation,
        "Analytic_Error_Bound": Analytic_Error_Bound
    }
    
    # Run Expert_Team_Selection
    team_selector = Expert_Team_Selection(
        Dirichlet_Parameters=Dirichlet_Parameters_Configurations[Dirichlet_Parameters_Name],
        N_experts=config_params["N_experts"],
        N_players=config_params["N_players"],
        tau=config_params["tau"],
        Entry_fee=config_params["Entry_fee"],
        Platform_fee_fraction=config_params["Platform_fee_fraction"],
        Means_Lognormal=config_params["Means_Lognormal"],
        Std_devs_Lognormal=config_params["Std_devs_Lognormal"],
        correlation=config_params["correlation"],
        Analytic_error_bound=config_params["Analytic_Error_Bound"]
    )
    
    # Run simulation
    winnings = team_selector.run_simulation(Total_iterations=N_iterations)
    
    return {
        "correlation": correlation,
        "tau": tau,
        "error_bound": Analytic_Error_Bound,
        "dirichlet_params": Dirichlet_Parameters_Name,
        "config": Config,
        "config_params": config_params,
        "team_selector": team_selector,
        "winnings": winnings,
        "highest_prob_team": int(np.argmax(np.array(team_selector.win_prob_team))) + 1
    }

def run_ultra_parallel_simulation(Dirichlet_Parameters_Configurations, N_experts, N_players,
                                 Entry_fee, Platform_fee_fraction, N_iterations, folder,
                                 n_processes=None):
    """
    Ultra-parallel version that parallelizes each individual simulation
    """
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    print(f"Starting ultra-parallel simulation with {n_processes} processes...")
    
    # Create all parameter combinations (most granular level)
    correlations = [1/3, 1/2, 2/3]
    tau_error_combinations = [(0.1, 0.1), (0.2, 0.06), (0.2, 0.02), (0.2, 0.1),
                             (0.2, 0.14), (0.3, 0.1), (0.4, 0.1)]
    configs = ["Equi-mean_Equivariance", "Unequal-mean_Equivariance", 
              "Equi-mean_Unequal-std", "Unequal-mean_Unequal-std"]
    
    all_args = []
    for correlation in correlations:
        for (Tau, Error_Bound) in tau_error_combinations:
            for Dirichlet_Parameters_Name in Dirichlet_Parameters_Configurations.keys():
                for Config in configs:
                    args = (correlation, Tau, Error_Bound, Dirichlet_Parameters_Name, Config,
                           Dirichlet_Parameters_Configurations, N_experts, N_players,
                           Entry_fee, Platform_fee_fraction, N_iterations, folder)
                    all_args.append(args)
    
    print(f"Total number of individual simulations: {len(all_args)}")
    
    start_time = time.time()
    
    # Run all simulations in parallel
    with Pool(processes=n_processes) as pool:
        results = []
        for i, result in enumerate(pool.imap(process_single_simulation, all_args)):
            results.append(result)
            if (i + 1) % 10 == 0 or i == len(all_args) - 1:
                print(f"Completed {i+1}/{len(all_args)} simulations "
                      f"({((i+1)/len(all_args)*100):.1f}%)")
    
    # Group results and save
    grouped_results = {}
    for result in results:
        key = (result["correlation"], result["tau"], result["error_bound"], 
               result["dirichlet_params"])
        if key not in grouped_results:
            grouped_results[key] = {}
        grouped_results[key][result["config"]] = result
    
    # Save grouped results
    for key, config_results in grouped_results.items():
        correlation, tau, error_bound, dirichlet_params = key
        
        output_dir = Path(f"{folder}/{round(correlation,2)}/Tau={tau}_ErrorBound={error_bound}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"Simulation Output.xlsx"
        
        with pd.ExcelWriter(f"{output_dir}/{filename}", engine="openpyxl") as writer:
            for config_name, result in config_results.items():
                result["winnings"].to_excel(writer, sheet_name=config_name, index=False)
            # Add Best_Teams (now Highest_Prob_Team) sheet
            best_teams_df = pd.DataFrame([
                {
                    "Config": config_name,
                    "Highest_Prob_Team": result["highest_prob_team"]
                }
                for config_name, result in config_results.items()
            ])
            best_teams_df.to_excel(writer, sheet_name="Best_Teams", index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nAll simulations completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per simulation: {total_time/len(all_args):.2f} seconds")
    
    return results

# Usage example:
if __name__ == "__main__":
    Experts=5
    results = run_ultra_parallel_simulation(
         Dirichlet_Parameters_Configurations={"IID":np.array([[2,1.5,7.5] for i in range(Experts)])
                                        },N_iterations=10000,
         N_experts=Experts,
        N_players=1000,
         Entry_fee=25,
         Platform_fee_fraction=0.2,
       folder=f"Base_{Experts}",
        n_processes=11)  # Use more processes for maximum parallelization