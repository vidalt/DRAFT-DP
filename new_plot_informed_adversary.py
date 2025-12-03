import os 
import json
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

list_N_trees = [10]
list_epsilon = [0.1, 1, 5, 10, 20, 30, 1000]
list_obj_active = [1]
list_depth = [5]
list_seed = [0,1,2,3,4,5,6,7,8,9]
list_datasets = ['compas' ,'default_credit', 'adult']
target_ratio_divisors = ["0.050"]

for N_samples in [100, 500, 1000, 2000, 5000, 10000, 20000]:
    threshold_low = 3
    reg_val_1 = float(target_ratio_divisors[0]) # 1 # for epsilon <= than threshold, will use the regularized variant with this parameter
    reg_val_2 = reg_val_1 # 0.001 # for epsilon > than threshold, will use the regularized variant with this parameter
    i = 0
    results_per_dataset = {}
    dataset_medoid_baseline = {'adult': 0.19382631578947365, 'compas': 0.19321428571428573, 'default_credit': 0.3165333333333333}
    for N_trees in list_N_trees:
        for epsilon in list_epsilon: 
            for obj_active_bool in list_obj_active:
                for depth in list_depth:
                    for dataset in list_datasets:
                        if dataset == 'compas' and N_samples != 2000:
                            continue
                        if dataset == 'adult' and N_samples != 20000:
                            continue
                        if dataset == 'default_credit' and N_samples != 10000:
                            continue
                        for seed in list_seed:
                            # Also load the corresponding simple informed adversary results
                            if N_samples == 100:
                                filename = "N_fixed%d_%.2f_%d_%d_%s_results.json" %(N_trees, epsilon, seed, depth, dataset) 
                            else:
                                filename = "N_fixed%d_%.2f_%d_%d_%d_%s_results.json" %(N_trees, epsilon, seed, depth, N_samples, dataset)
                            with open(os.path.join("experiments_results/Results_informed_adversary_balle", filename), 'r') as f:
                                result_simple = json.load(f)[0]
                                #print("Loaded simple informed adversary results file:", filename)
                            
                            if dataset not in results_per_dataset:
                                    results_per_dataset[dataset] = {
                                        'epsilon': [],
                                        'indiv_recon_error': {},
                                        'all_recon_error': [],
                                        'indiv_recon_error_simple': [],
                                        'duration_avg': {}
                                    }
                            train_accuracy_simple = result_simple['accuracy_train']
                            test_accuracy_simple = result_simple['accuracy_test']
                            #all_recon_error = result_main['reconstruction_error']
                            indiv_recon_error_simple = result_simple['example_reconstruction_error_avg']
                            results_per_dataset[dataset]['epsilon'].append(epsilon)
                            results_per_dataset[dataset]['indiv_recon_error_simple'].append(indiv_recon_error_simple)
                            i += 1
                            
                            '''if epsilon > threshold_low: # load results without regularization
                                with open(os.path.join("experiments_results/Results_informed_adversary_save", filename), 'r') as f:
                                    result = json.load(f)[0]
                                indiv_recon_error = result['example_reconstruction_error_avg']
                                if -1 not in results_per_dataset[dataset]['indiv_recon_error']:
                                    results_per_dataset[dataset]['indiv_recon_error'][-1] = []
                                results_per_dataset[dataset]['indiv_recon_error'][-1].append(indiv_recon_error)'''

                            for target_ratio_divisor in target_ratio_divisors:
                                if N_samples == 100:
                                    filename = "N_fixed%d_%.2f_%d_%d_%s_%s_results.json" %(N_trees, epsilon, seed, depth, str(target_ratio_divisor), dataset) 
                                else:
                                    filename = "N_fixed%d_%.2f_%d_%d_%s_%d_%s_results.json" %(N_trees, epsilon, seed, depth, str(target_ratio_divisor), N_samples, dataset) 
                                with open(os.path.join("experiments_results/Results_informed_adversary", filename), 'r') as f:
                                    result = json.load(f)[0]
                                    #print("Loaded informed adversary results file:", filename)
                                
                                target_ratio_divisor = result['target_ratio_divisor']
                                epsilon = result['epsilon']
                                train_accuracy = result['accuracy_train']
                                test_accuracy = result['accuracy_test']
                                indiv_recon_error = result['example_reconstruction_error_avg']
                                duration_avg = result['duration_avg']
                                dataset = os.path.splitext(os.path.basename(result['dataset']))[0]
                                    # Make sure the results correspond to the same DP RFs
                                #assert result_main['epsilon'] == epsilon, "Epsilon values do not match!"
                                assert result_simple['epsilon'] == epsilon, "Epsilon values do not match!"
                                # Also load the corresponding all-reconstr results
                                #with open(os.path.join("experiments_results/Results_main_paper", filename), 'r') as f:
                                #    result_main = json.load(f)[0]
                                #    print("Corresponding main paper results file found for:", filename)

                                #train_accuracy_main = result_main['accuracy_train']
                                #test_accuracy_main = result_main['accuracy_test']
                                

                                #assert(train_accuracy_main == train_accuracy)
                                #assert(test_accuracy_main == test_accuracy)
                                if train_accuracy_simple != train_accuracy or test_accuracy_simple != test_accuracy:
                                    print("Warning: Train/Test accuracies do not match between informed adversary and simple informed adversary for file:", filename)

                                if (epsilon <= threshold_low) and target_ratio_divisor == reg_val_1:
                                    if -1 not in results_per_dataset[dataset]['indiv_recon_error']:
                                        results_per_dataset[dataset]['indiv_recon_error'][-1] = []
                                        results_per_dataset[dataset]['duration_avg'][-1] = []
                                    results_per_dataset[dataset]['indiv_recon_error'][-1].append(indiv_recon_error)
                                    results_per_dataset[dataset]['duration_avg'][-1].append(duration_avg)
                                
                                if epsilon > threshold_low and target_ratio_divisor == reg_val_2:
                                    if -1 not in results_per_dataset[dataset]['indiv_recon_error']:
                                        results_per_dataset[dataset]['indiv_recon_error'][-1] = []
                                        results_per_dataset[dataset]['duration_avg'][-1] = []
                                    results_per_dataset[dataset]['indiv_recon_error'][-1].append(indiv_recon_error)
                                    results_per_dataset[dataset]['duration_avg'][-1].append(duration_avg)

                                if target_ratio_divisor not in results_per_dataset[dataset]['indiv_recon_error']:
                                    results_per_dataset[dataset]['indiv_recon_error'][target_ratio_divisor] = []

                                results_per_dataset[dataset]['indiv_recon_error'][target_ratio_divisor].append(indiv_recon_error)
                                #results_per_dataset[dataset]['all_recon_error'].append(all_recon_error)
                                

                                i += 1

    print("[N_samples %d] Total number of result files processed: %d" % (N_samples, i))

    j = 0
    target_ratio_divisor = -1
    for dataset in results_per_dataset.keys():
        list_eps = results_per_dataset[dataset]['epsilon']
        list_indiv = results_per_dataset[dataset]['indiv_recon_error'][target_ratio_divisor]
        durations = results_per_dataset[dataset]['duration_avg'][target_ratio_divisor]
        #list_all = results_per_dataset[dataset]['all_recon_error']
        list_indiv_simple = results_per_dataset[dataset]['indiv_recon_error_simple']

        # Average errors that correspond to the same epsilon
        unique_epsilons = sorted(set(list_eps))
        avg_indiv_errors = []
        avg_all_errors = []
        std_indiv_errors = []
        std_all_errors = []
        avg_indiv_simple = []
        std_indiv_simple = []
        print("Global duration avg = ", np.mean(durations), " over ", len(durations), " runs.")
        for eps in unique_epsilons:
            indices = [index for index, value in enumerate(list_eps) if value == eps]
            assert(len(indices) == len(list_seed))  # seeds
            
            avg_indiv_errors.append(np.mean([list_indiv[index] for index in indices]))
            #avg_all_errors.append(avg_all)
            std_indiv_errors.append(np.std([list_indiv[index] for index in indices]))
            #std_all_errors.append(np.std([list_all[index] for index in indices]))

            avg_indiv_simple.append(np.mean([list_indiv_simple[index] for index in indices]))
            std_indiv_simple.append(np.std([list_indiv_simple[index] for index in indices]))
        

        plt.figure()
        #plt.title(f"Reconstruction Error vs Epsilon for {dataset}")
        plt.xlabel(r"$\epsilon$") 
        plt.ylabel("Reconstruction Error")

        plt.axhline(y=dataset_medoid_baseline[dataset], color='red', linestyle='--', label='Simple per-coordinate majority baseline')
        
        plt.plot(unique_epsilons, avg_indiv_errors, marker='o', label='Informed Adversary', color='blue')
        plt.fill_between(
            unique_epsilons,
            np.array(avg_indiv_errors) - np.array(std_indiv_errors),
            np.array(avg_indiv_errors) + np.array(std_indiv_errors),
            alpha=0.2, color='blue'
        )

        '''plt.plot(unique_epsilons, avg_all_errors, marker='x', label='Entire Reconstruction', color='orange')
        plt.fill_between(
            unique_epsilons,
            np.array(avg_all_errors) - np.array(std_all_errors),
            np.array(avg_all_errors) + np.array(std_all_errors),
            alpha=0.2, color='orange'
        )'''
        
        plt.plot(unique_epsilons, avg_indiv_simple, marker='s', label='Informed Adversary Baseline [47]', color='green')
        plt.fill_between(
            unique_epsilons,
            np.array(avg_indiv_simple) - np.array(std_indiv_simple),
            np.array(avg_indiv_simple) + np.array(std_indiv_simple),
            alpha=0.2, color='green'
        )
        plt.xscale('log')
        '''plt.xlim(0,40)
        plt.xscale('log')
        ticks = plt.xticks()[0]          # current tick locations
        labels = [item.get_text() for item in plt.gca().get_xticklabels()]  # current tick labels

        # Example: change the label of the tick at value 30
        for i, t in enumerate(ticks):
            if t == 40:
                labels[i] = "1000"   # new label

        plt.xticks(ticks, labels)       # apply updated labels'''

        '''plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=1
        )'''
        plt.grid()
        plt.savefig(f"figures/reconstruction_error_informed_adversary_{dataset}_{N_samples}_{reg_val_1}.pdf", bbox_inches='tight')
        print("Plot saved:", f"figures/reconstruction_error_informed_adversary_{dataset}_{N_samples}_{reg_val_1}.pdf")

        j += 1

    print("Total number of plots generated:", j)

    # Separate legend
    legend_elements = []
    legend_elements.append(Line2D([0], [0], marker='s', label='Informed adversary baseline (adapted from [47])', color='green' )) #lw=5, 
    legend_elements.append(Line2D([0], [0], marker=None, color='red', linestyle='--', label='Simple per-coordinate majority baseline')) #lw=5, 
    legend_elements.append(Line2D([0], [0], marker='o', label='Informed adversary (ours)', color='blue' )) #lw=5, 

    
    legendFig = plt.figure(figsize=(2, 2))
    ax = legendFig.add_subplot(111)
    ax.axis('off')

    legend = ax.legend(handles=legend_elements, loc='center', ncol=1)

    legendFig.canvas.draw()
    bbox = legend.get_window_extent().transformed(legendFig.dpi_scale_trans.inverted())

    plt.axis('off')
    legendFig.savefig('figures/plot_informed_experiments_legend.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()