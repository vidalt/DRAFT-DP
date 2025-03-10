from DP_RF import DP_RF
from DP_RF_solver import DP_RF_solver
import pandas as pd
from utils import * 
from DRAFT import DRAFT
import matplotlib.pyplot as plt
import json
import numpy as np
from datasets_infos import datasets_ohe_vectors, predictions, datasets_ordinal_attrs, datasets_numerical_attrs, random_predictions_accuracy, random_reconstruction_error, random_reconstruction_error_std

path = "data/compas.csv"
prediction_target = "recidivate-within-two-years:1"
dataset = "compas"
N_samples = 100
N_fixed = 100  # If N is known, set N_fixed = N_samples, else set N_fixed = None
N = "N_fixed"
N_trees_values = [1, 5, 10, 20, 30]
seeds = [0, 1, 2, 3, 4]
depth = 5

time_out = 100
n_threads = 16
verbosity = 0
obj_active = 1

data = pd.read_csv(path)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
sample_size = len(X)

# Store results
baseline_means = {}
baseline_stds = {}

# Main loop
for N_trees in N_trees_values:
    baseline_errors = []
    for seed in seeds:
        np.random.seed(seed)

        # Split dataset
        X_train, X_test, y_train, y_test = data_splitting(data, prediction_target, sample_size - N_samples, seed)

        # Train DP-RF model
        clf = DP_RF(path, dataset, N_samples, N_trees, depth, seed, verbosity)
        clf.fit(X_train, y_train)

        # Baseline (DRAFT without noise)
        extractor = DRAFT(clf)
        dict_DRAFT = extractor.fit(bagging=False, method="cp-sat", timeout=60, verbosity=False, n_jobs=-1, seed=seed)
        x_DRAFT = dict_DRAFT['reconstructed_data']
        e_baseline, _ = average_error(x_DRAFT, X_train.to_numpy(), seed)
        baseline_errors.append(e_baseline)

    # Store results (mean and std) per N_trees
    baseline_means[N_trees] = np.mean(baseline_errors)
    baseline_stds[N_trees] = np.std(baseline_errors)

print("Means:", baseline_means)
print("Stds:", baseline_stds)

# Predefined colors for each N_trees
predefined_colors = {
    1: "tab:blue",
    5: "tab:orange",
    10: "tab:green",
    20: "tab:red",
    30: "tab:purple",
}

# Load JSON results file
with open(f'{N}_{dataset}_results.json', 'r') as f:
    results = json.load(f)

# Store results
data_error = {}

# Extract results
for result in results:
    if result['solver_status'] == 'UNKNOWN':
        continue

    N_trees = result['N_trees']
    epsilon = result['epsilon']
    depth_value = result['depth']
    reconstruction_error = result['reconstruction_error']

    if (N_trees, epsilon, depth_value) not in data_error:
        data_error[(N_trees, epsilon, depth_value)] = []

    data_error[(N_trees, epsilon, depth_value)].append(reconstruction_error)

# Plot graphs
plt.figure()

# Plot reconstruction error curves
for N_trees in sorted(predefined_colors.keys()):
    epsilons = []
    errors_mean = []
    errors_std = []

    for key in sorted(data_error.keys()):
        if key[0] == N_trees and key[2] == depth:
            epsilon = key[1]
            errors = data_error[key]
            mean_err = np.mean(errors)
            std_err = np.std(errors)

            epsilons.append(epsilon)
            errors_mean.append(mean_err)
            errors_std.append(std_err)

    epsilons = np.array(epsilons)
    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)

    # Define color
    color = predefined_colors[N_trees]

    # Plot curves and uncertainty zones
    plt.plot(epsilons, errors_mean, linestyle='-', label=f'#trees={N_trees}', color=color, marker='+', markersize=5)
    plt.fill_between(epsilons, errors_mean - errors_std, errors_mean + errors_std, alpha=0.2, color=color)

    # Use corresponding baseline values for each N_trees
    plt.plot(
        epsilons,
        [baseline_means[N_trees]] * len(epsilons),
        linestyle='--',
        label=f'Baseline #trees={N_trees}',
        color=color,
    )
    plt.fill_between(
        epsilons,
        [baseline_means[N_trees] - baseline_stds[N_trees]] * len(epsilons),
        [baseline_means[N_trees] + baseline_stds[N_trees]] * len(epsilons),
        alpha=0.2,
        color=color,
    )

# Configure plot
plt.xlim(0, 30.5)
plt.xlabel(r'$\varepsilon$')
plt.ylabel('Reconstruction Error')
plt.grid(False)
plt.tight_layout()

# Save figure
plt.savefig(f'reconstruction_error_vs_epsilon_{dataset}.pdf', bbox_inches='tight')
plt.close()