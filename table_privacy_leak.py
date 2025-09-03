import json
import numpy as np
import scipy.stats as stats
from decimal import Decimal

datasets = ["adult", "compas", "default_credit"]
depth_values = [3, 5, 7]
epsilon_values = [0.1, 1, 5, 10, 20, 30]
tree_values = [1, 5, 10, 20, 30]

# Random baseline
random_reconstruction_error = {"compas": 0.2081, "adult": 0.2539, "default_credit": 0.2642}

table_data = {dataset: {n_trees: {epsilon: {depth: "-" for depth in depth_values} for epsilon in epsilon_values} for n_trees in tree_values} for dataset in datasets}

for dataset in datasets:
    with open(f'experiments_results/Results_main_paper/N_fixed_{dataset}_results.json', 'r') as f:
        results = json.load(f)

    for depth in depth_values:
        for n_trees in tree_values:
            for epsilon in epsilon_values:
                error = []
                for seed in range(5):
                    for result in results:
                        if (result['depth'] == depth and result['epsilon'] == epsilon 
                                and result['N_trees'] == n_trees and result['solver_status'] != 'UNKNOWN')\
                                    and result['seed'] == seed:
                            random_errs_sorted = sorted(result['reconstruction_error_matching_distrib_list']) # list of values of reconstr error for datasets of the same distrib
                            actual_error = result['reconstruction_error'] # reconstr error for the actual RF training set
                            
                            # Fit a normal distribution with the reconstr errors from datasets randomly drawn from the data distrib
                            # and compute the CDF for all points (including the actual RF training data)
                            # Compute params of the normal distribution for datasets randomly drawn from the distrib
                            df_mean = np.mean(random_errs_sorted)
                            df_std = np.std(random_errs_sorted)
                            
                            # Points for which we will measure the CDF (actual RF training set + datasets randomly drawn from the data distrib)
                            all_errors = [actual_error] + random_errs_sorted 
                            all_errors = sorted(all_errors)

                            cdf = stats.norm.cdf(all_errors, df_mean, df_std)
                            
                            actual_cdf_val = cdf[all_errors.index(actual_error)]

                            error.append(actual_cdf_val)
                
                if error:
                    assert(len(error) == 5)
                mean_error = np.mean(error) if error else "-"
                if mean_error != "-" and mean_error < 0.001:
                    table_data[dataset][n_trees][epsilon][depth] = "\\textbf{%.2E}" %(Decimal(mean_error))
                elif mean_error != "-" and mean_error <= 0.05:
                    table_data[dataset][n_trees][epsilon][depth] = "\\textbf{{%.3f}}" %mean_error if mean_error != "-" else "-"
                else:
                    table_data[dataset][n_trees][epsilon][depth] = f"{mean_error:.3f}" if mean_error != "-" else "-"
# LaTeX table
latex_table = r"""
\begin{tabular}{llccccccccc}
\toprule
                                &                     & \multicolumn{3}{c}{Adult} & \multicolumn{3}{c}{COMPAS} & \multicolumn{3}{c}{Default Credit} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
                                &                     & $d=3$   & $d=5$  & $d=7$  & $d=3$   & $d=5$   & $d=7$  & $d=3$      & $d=5$      & $d=7$     \\
\midrule
"""

for i, n_trees in enumerate(tree_values):
    latex_table += f"\\multirow{{6}}{{*}}{{$\\lvert \\forest \\rvert = {n_trees}$}}  "
    for epsilon in epsilon_values:
        latex_table += f"& $\\varepsilon = {epsilon}$ "
        for dataset in datasets:
            for depth in depth_values:
                latex_table += f"& {table_data[dataset][n_trees][epsilon][depth]} "
        latex_table += "\\\\\n"
    if i < len(tree_values) - 1:
        latex_table += "\midrule\n"
    else:
        latex_table += r"\bottomrule\end{tabular}"


print(latex_table)
