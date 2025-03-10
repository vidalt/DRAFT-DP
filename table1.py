import json
import numpy as np

datasets = ["adult", "compas", "default_credit"]
depth_values = [3, 5, 7]
epsilon_values = [0.1, 1, 5, 10, 20, 30]
tree_values = [1, 5, 10, 20, 30]

# Random baseline
random_reconstruction_error = {"compas": 0.2081, "adult": 0.2539, "default_credit": 0.2642}

table_data = {dataset: {n_trees: {epsilon: {depth: "-" for depth in depth_values} for epsilon in epsilon_values} for n_trees in tree_values} for dataset in datasets}

for dataset in datasets:
    with open(f'experiments_results/N_fixed_{dataset}_results.json', 'r') as f:
        results = json.load(f)
    
    for depth in depth_values:
        for n_trees in tree_values:
            for epsilon in epsilon_values:
                error = []
                for seed in range(5):
                    for result in results:
                        if (result['depth'] == depth and result['epsilon'] == epsilon 
                                and result['N_trees'] == n_trees and result['solver_status'] != 'UNKNOWN'):
                            error.append(result['reconstruction_error'])
                
                mean_error = np.mean(error) if error else "-"
                if mean_error != "-" and mean_error > random_reconstruction_error[dataset]:
                    table_data[dataset][n_trees][epsilon][depth] = f"\\textbf{{{mean_error:.2f}}}"
                else:
                    table_data[dataset][n_trees][epsilon][depth] = f"{mean_error:.2f}" if mean_error != "-" else "-"

# LaTeX table
latex_table = r"""
\begin{tabular}{llccccccccc}
\toprule
                                &                     & \multicolumn{3}{c}{Adult} & \multicolumn{3}{c}{COMPAS} & \multicolumn{3}{c}{Default Credit} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
                                &                     & $d=3$   & $d=5$  & $d=7$  & $d=3$   & $d=5$   & $d=7$  & $d=3$      & $d=5$      & $d=7$     \\
\midrule
"""

for n_trees in tree_values:
    latex_table += f"\\multirow{{6}}{{*}}{{$\\lvert \\forest \\rvert = {n_trees}$}}  "
    for epsilon in epsilon_values:
        latex_table += f"& $\\varepsilon = {epsilon}$ "
        for dataset in datasets:
            for depth in depth_values:
                latex_table += f"& {table_data[dataset][n_trees][epsilon][depth]} "
        latex_table += "\\\\\n"
    latex_table += "\midrule\n"

# Add random baselines
latex_table += "\\multicolumn{2}{c}{Random baseline} "
for dataset in datasets:
    latex_table += f"& \multicolumn{{3}}{{c}}{{{random_reconstruction_error[dataset]:.2f}}} "
latex_table += r"\\\bottomrule\end{tabular}"

print(latex_table)
