import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import * 
import numpy as np 
from matplotlib.lines import Line2D
from datasets_infos import datasets_ohe_vectors, predictions, datasets_ordinal_attrs, datasets_numerical_attrs, random_predictions_accuracy, random_reconstruction_error, random_reconstruction_error_std
import matplotlib.gridspec as gridspec


# Plot reconstruction error for each dataset

for N in {'N_fixed'}:
    for dataset in ["compas", "adult", "default_credit"]:
        
        with open(f'experiments_results/{N}_{dataset}_results.json', 'r') as f:
            results = json.load(f)


        script_seed = 42
        use_bootstrap = False
        sample_size = 125
        train_size = 100

        size = (7,5)

        ohe_vector = datasets_ohe_vectors[dataset] # list of sublists indicating sets of binary attributes one-hot-encoding the same original one
        ordinal_attrs = datasets_ordinal_attrs[dataset] # list of ordinal attributes
        numerical_attrs = datasets_numerical_attrs[dataset] # list of numerical attributes
        prediction = predictions[dataset] # the attribute we want to predict

        df = pd.read_csv("data/%s.csv" %dataset)

        df = df.sample(n=125, random_state = 0, ignore_index= True)
        X_train, X_test, y_train, y_test = data_splitting(df, prediction, test_size=sample_size-train_size, seed=script_seed)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()


        rand_avg = random_reconstruction_error[dataset]
        rand_std = random_reconstruction_error_std[dataset]

        # Initialisation of data structures to store the results
        data_error = {}
        data_reconstructed = {}
        data_accuracy_train = {}
        data_accuracy_test = {}

        for result in results:
            if result['solver_status'] == 'UNKNOWN':
                continue

            N_trees = result['N_trees']
            epsilon = result['epsilon']
            depth = result['depth']
            reconstruction_error = result['reconstruction_error']
            if N == 'N_fixed':
                accuracy_train = result['accuracy_train']
                accuracy_test = result['accuracy_test']
            N_reconstruit = result['N_reconstruit']

            if (N_trees, epsilon, depth) not in data_error:
                data_error[(N_trees, epsilon, depth)] = []
            if (N_trees, epsilon, depth) not in data_reconstructed:
                data_reconstructed[(N_trees, epsilon, depth)] = []
            if N == 'N_fixed':
                if (N_trees, epsilon, depth) not in data_accuracy_train:
                    data_accuracy_train[(N_trees, epsilon, depth)] = []
                if (N_trees, epsilon, depth) not in data_accuracy_test:
                    data_accuracy_test[(N_trees, epsilon, depth)] = []

            data_error[(N_trees, epsilon, depth)].append(reconstruction_error)
            data_reconstructed[(N_trees, epsilon, depth)].append(N_reconstruit)
            if N == 'N_fixed':
                data_accuracy_train[(N_trees, epsilon, depth)].append(accuracy_train)
                data_accuracy_test[(N_trees, epsilon, depth)].append(accuracy_test)

        depth_values = sorted(set([key[2] for key in data_error.keys()]))

        for depth in depth_values:
            # Reconstruction error plot for each depth value
            plt.figure(figsize=size, dpi=300)

            for N_trees in sorted(set([key[0] for key in data_error.keys()])):
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

                plt.plot(epsilons, errors_mean, linestyle='-')#, label=f'N_trees={'#trees'}')
                plt.fill_between(epsilons, errors_mean - errors_std, errors_mean + errors_std, alpha=0.2)

            plt.plot(np.arange(0.1, 30.1, 0.1), [rand_avg] * len(np.arange(0.1, 30.1, 0.1)), linestyle='--', color='gray')#, label=f'Random Baseline')
            plt.fill_between(np.arange(0.1, 30.1, 0.1), rand_avg - rand_std, rand_avg + rand_std, alpha=0.2, color='gray')

            plt.xlim(0, 30.5)
            plt.xlabel(r'$\varepsilon$')
            plt.ylabel('Reconstruction Error')
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(f'figures/{N}_{dataset}_error_epsilon_depth_{depth}.pdf',bbox_inches='tight')
            plt.close()

            if N == 'N_fixed':
                #Train accuracy plot for each depth value
                plt.figure(figsize=size, dpi=300)

                for N_trees in sorted(set([key[0] for key in data_accuracy_train.keys()])):
                    epsilons = []
                    accuracies_train_mean = []
                    accuracies_train_std = []

                    for key in sorted(data_accuracy_train.keys()):
                        if key[0] == N_trees and key[2] == depth:
                            epsilon = key[1]
                            accuracies_train = data_accuracy_train[key]
                            mean_acc_train = np.mean(accuracies_train)
                            std_acc_train = np.std(accuracies_train)

                            epsilons.append(epsilon)
                            accuracies_train_mean.append(mean_acc_train)
                            accuracies_train_std.append(std_acc_train)

                    epsilons = np.array(epsilons)
                    accuracies_train_mean = np.array(accuracies_train_mean)
                    accuracies_train_std = np.array(accuracies_train_std)

                    plt.plot(epsilons, accuracies_train_mean, linestyle='-')#, label=f'N_trees={'#trees'}')
                    plt.fill_between(epsilons, accuracies_train_mean - accuracies_train_std, accuracies_train_mean + accuracies_train_std, alpha=0.2)

                plt.xlim(0, 30.5)
                plt.xlabel(r'$\varepsilon$')
                plt.ylabel('Train Accuracy')
                plt.legend()
                plt.grid(False)
                plt.tight_layout()
                plt.savefig(f'figures/{N}_{dataset}_train_accuracy_epsilon_depth_{depth}.pdf',bbox_inches='tight')
                plt.close()

                #Test accuracy plot for each depth value
                plt.figure(figsize=size, dpi=300)

                for N_trees in sorted(set([key[0] for key in data_accuracy_test.keys()])):
                    epsilons = []
                    accuracies_test_mean = []
                    accuracies_test_std = []

                    for key in sorted(data_accuracy_test.keys()):
                        if key[0] == N_trees and key[2] == depth:
                            epsilon = key[1]
                            accuracies_test = data_accuracy_test[key]
                            mean_acc_test = np.mean(accuracies_test)
                            std_acc_test = np.std(accuracies_test)

                            epsilons.append(epsilon)
                            accuracies_test_mean.append(mean_acc_test)
                            accuracies_test_std.append(std_acc_test)

                    epsilons = np.array(epsilons)
                    accuracies_test_mean = np.array(accuracies_test_mean)
                    accuracies_test_std = np.array(accuracies_test_std)

                    plt.plot(epsilons, accuracies_test_mean, linestyle='-')#, label=f'N_trees={'#trees'}')
                    plt.fill_between(epsilons, accuracies_test_mean - accuracies_test_std, accuracies_test_mean + accuracies_test_std, alpha=0.2)

                plt.xlim(0, 30.5)
                plt.xlabel(r'$\varepsilon$')
                plt.ylabel('Test Accuracy')
                plt.legend()
                plt.grid(False)
                plt.tight_layout()
                plt.savefig(f'figures/{N}_{dataset}_test_accuracy_epsilon_depth_{depth}.pdf',bbox_inches='tight')
                plt.close()
                
# Separated legend plot
legend_elements = []

# Separated N_trees legend

for N_trees in [1,5,10,20,30]:
    color = plt.plot([], [])[0].get_color()
    legend_elements.append(Line2D([0], [0], marker=None, color=color, lw=5, label="#trees = %d" %N_trees))
plt.clf()
legendFig = plt.figure("Legend plot")

#legend_elements.append(Line2D([0], [0], marker=None, linestyle='dotted', color='black', lw=1, label='Random Baseline'))
legend_elements.append(Line2D([0], [0], marker=None, markersize=5, linestyle='--', color='black', lw=1, label='DRAFT Baseline'))
legend_elements.append(Line2D([0], [0], marker='+', markersize=5, linestyle='solid', color='black', lw=1, label='Ours (DP RFs)'))

legend_elements_reordered = [legend_elements[0], legend_elements[5], legend_elements[1], legend_elements[6], legend_elements[2], legend_elements[3], legend_elements[4]]
legendFig.legend(handles=legend_elements_reordered, loc='center', ncol=5)
plt.axis('off')
legendFig.savefig('figures/legend.pdf', bbox_inches='tight')
plt.close()

# Plot accuracy vs error for each dataset

base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def adjust_marker_size(epsilon, min_epsilon, max_epsilon):
    min_size = 10
    max_size = 250
    return min_size + (max_size - min_size) * (epsilon - min_epsilon) / (max_epsilon - min_epsilon)

for N in {'N_fixed'}:
    for dataset in ["compas", "adult", "default_credit"]:
        with open(f'experiments_results/{N}_{dataset}_results.json', 'r') as f:
            results = json.load(f)

        script_seed = 42
        use_bootstrap = False
        sample_size = 125
        train_size = 100

        size = (7, 5)

        ohe_vector = datasets_ohe_vectors[dataset]
        ordinal_attrs = datasets_ordinal_attrs[dataset]
        numerical_attrs = datasets_numerical_attrs[dataset]
        prediction = predictions[dataset]

        df = pd.read_csv(f"data/{dataset}.csv")

        df = df.sample(n=125, random_state=0, ignore_index=True)
        X_train, X_test, y_train, y_test = data_splitting(df, prediction, test_size=sample_size - train_size, seed=script_seed)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()


        rand_avg = random_reconstruction_error[dataset]
        rand_std = random_reconstruction_error_std[dataset]
        rand_acc = random_predictions_accuracy[dataset]
        
        data_error = {}
        data_accuracy_train = {}

        for result in results:
            if result['solver_status'] == 'UNKNOWN':
                continue

            N_trees = result['N_trees']
            epsilon = result['epsilon']
            depth = result['depth']
            
            reconstruction_error = result['reconstruction_error']
            accuracy_train = result['accuracy_train']

            if dataset == "adult" and depth == 5 and epsilon == 30 and N_trees == 1:
                print(accuracy_train, reconstruction_error)

            if (N_trees, epsilon, depth) not in data_error:
                data_error[(N_trees, epsilon, depth)] = []
            if (N_trees, epsilon, depth) not in data_accuracy_train:
                data_accuracy_train[(N_trees, epsilon, depth)] = []

            data_error[(N_trees, epsilon, depth)].append(reconstruction_error)
            data_accuracy_train[(N_trees, epsilon, depth)].append(accuracy_train)

        depth_values = sorted(set([key[2] for key in data_error.keys()]))
        epsilon_values = sorted(set([key[1] for key in data_error.keys()]))
        n_trees_values = sorted(set([key[0] for key in data_error.keys()])) 

        min_epsilon = min(epsilon_values)
        max_epsilon = max(epsilon_values)
        plt.rcParams.update({'font.size': 14})

        for depth in depth_values:
            fig = plt.figure(figsize=(9,5))
            #gs = gridspec.GridSpec(2, 1, height_ratios=[6, 0.1])
            #ax_main = plt.subplot(gs[0])
            #ax_legend = plt.subplot(gs[1])

            for i, N_trees in enumerate(n_trees_values):
                base_color = base_colors[i % len(base_colors)]
                for j, epsilon in enumerate(epsilon_values):
                    key = (N_trees, epsilon, depth)
                    if key in data_error:
                        errors_mean = np.mean(data_error[key])
                        accuracies_train_mean = np.mean(data_accuracy_train[key])
                        
                        marker_size = adjust_marker_size(epsilon, min_epsilon, max_epsilon)

                        plt.scatter(errors_mean, accuracies_train_mean, marker='+', s=marker_size, color=base_color, label=f'#trees={N_trees}' if j == 0 else "")

            plt.xlabel('Reconstruction Error')
            plt.ylabel('Accuracy (train)')
            #plt.grid(False)
            plt.axvline(x=rand_avg, linestyle='--', color='gray')
            plt.axhline(y=rand_acc, linestyle='--', color='black')

            legend_elements = []
            for i, N_trees in enumerate(n_trees_values):
                color = base_colors[i % len(base_colors)]
                legend_elements.append(Line2D([0], [0], marker='+', color=color, linestyle='None', markersize=10, label=f'#trees = {N_trees}'))
            #ax_main.legend(handles=legend_elements, loc='upper right')

            #legend_elements.append(Line2D([0], [0], marker=None, color='none', linestyle='None', label=None))
            # Epsilon legend
            #ax_legend.axis('off')
            legend_elements_epsilon = []
            offset=0
            for epsilon in epsilon_values:
                marker_size = adjust_marker_size(epsilon, min_epsilon, max_epsilon)
                legend_elements_epsilon.append(Line2D([0], [0], marker='+', color='black', linestyle='None', markersize=np.sqrt(marker_size), label=rf'$\varepsilon =$ {epsilon}'))
                offset += 1
                if offset == 3:
                    legend_elements_epsilon.append(Line2D([0], [0], marker=None, color='none', linestyle='None', label=None))

            legend_elements.extend(legend_elements_epsilon)
            legend_elements.append(Line2D([0], [0], marker=None, color='none', linestyle='None', label=None))

             # Vertical line for random error
            legend_elements.append(Line2D([0], [0], linestyle='--', color='gray', label=f'Random Reconstruction'))

            # Horizontal line for random accuracy
            legend_elements.append(Line2D([0], [0], linestyle='--', color='black', label=f'Majority Classifier'))
            legend_elements.append(Line2D([0], [0], marker=None, color='none', linestyle='None', label=None))
            legend_elements.append(Line2D([0], [0], marker=None, color='none', linestyle='None', label=None))
            # Add baseline legend
            #ax_main.legend(handles=legend_elements, loc='best')

            if dataset == 'compas':
                plt.xlim((0.02, 0.41))
                plt.ylim((0.44, 0.8))
            if dataset == 'adult':
                plt.xlim((0.08, 0.45))
                plt.ylim((0.39, 0.90))
            if dataset == 'default_credit':
                plt.xlim((0.1, 0.44))
                plt.ylim((0.39, 0.95))
            #plt.tight_layout()
            plt.savefig(f'figures/{N}_{dataset}_train_accuracy_vs_error_depth_{depth}.pdf', bbox_inches='tight')
            plt.close()
            
            # Plot legend separately
            legendFig = plt.figure("Legend plot")
            legendFig.legend(handles=legend_elements, loc='center', ncol=4)
            plt.axis('off')
            legendFig.savefig('figures/legend_tradeoffs.pdf', bbox_inches='tight')
            plt.close()

# Plot N_reconstructed for each dataset

"""for depth in depth_values:
    plt.figure(figsize=(20, 10))

    for N_trees in sorted(set([key[0] for key in data_reconstructed.keys()])):
        epsilons = []
        reconstructed_mean = []
        reconstructed_std = []

        for key in sorted(data_reconstructed.keys()):
            if key[0] == N_trees and key[2] == depth:
                epsilon = key[1]
                reconstructed = data_reconstructed[key]
                mean_reconstructed = np.mean(reconstructed)
                std_reconstructed = np.std(reconstructed)

                epsilons.append(epsilon)
                reconstructed_mean.append(mean_reconstructed)
                reconstructed_std.append(std_reconstructed)

        epsilons = np.array(epsilons)
        reconstructed_mean = np.array(reconstructed_mean)
        reconstructed_std = np.array(reconstructed_std)

        plt.plot(epsilons, reconstructed_mean, linestyle='-', label=f'N_trees={N_trees}')
        plt.fill_between(epsilons, reconstructed_mean - reconstructed_std, reconstructed_mean + reconstructed_std, alpha=0.2)
    plt.xlim(0, 31)
    plt.title(f'N_reconstruction en fonction de epsilon (profondeur={depth})')
    plt.xlabel('Epsilon')
    plt.ylabel('N_reconstruction')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'adult_N_reconstructed_epsilon_depth_{depth}.pdf')
"""