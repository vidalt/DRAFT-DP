import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 


figures_sizes = (7.0,4.0)
 #"milp" or "cp-sat" or "bench"
list_datasets = ['compas','default_credit', 'adult']
folder = "experiments_results/Results_partial_reconstruction/partial_reconstr_exps"
nattrs_datasets = {"compas":14-4-3, "adult":19-5, "default_credit":21-3-2} # All attributes OHEncoding the same original feature count only once

for dataset in list_datasets:
    print("==== EXPERIMENT: " + str(dataset) + " reconstr. ====")
    # Experiment (locate the right folder)
    results_file = f'{folder}_{dataset}_results.json'
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    reconstr_error_results = {}
    for a_result in results:
        assert(a_result['N_samples'] == 100)
        assert(a_result['N_trees'] == 10)
        assert(a_result['epsilon'] == 5)
        if a_result["solver_status"] in ["OPTIMAL", "FEASIBLE"]:
            if not a_result["known_attributes"] in reconstr_error_results.keys():
                reconstr_error_results[a_result["known_attributes"]] = [a_result["reconstruction_error"]]
            else:
                reconstr_error_results[a_result["known_attributes"]].append(a_result["reconstruction_error"])
        else:
            print("Uncompleted experiment: ", a_result)
    nb_attrs_list = np.sort(list(reconstr_error_results.keys()))
    reconstr_error_list = []
    reconstr_error_std_list = []
    for nb_attr in nb_attrs_list:
        assert(len(reconstr_error_results[nb_attr]) == 5)
        reconstr_error_list.append(np.average(reconstr_error_results[nb_attr]))
        reconstr_error_std_list.append(np.std(reconstr_error_results[nb_attr]))
    print(nb_attrs_list)
    print(reconstr_error_list)
    plt.rcParams["figure.figsize"] = (9,6)
    plt.rcParams.update({'font.size': 14})
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    base_color = base_colors[2 % len(base_colors)]

    plt.plot(nb_attrs_list, reconstr_error_list, color=base_color)
    plt.fill_between(nb_attrs_list, np.asarray(reconstr_error_list) - np.asarray(reconstr_error_std_list), np.asarray(reconstr_error_list) + np.asarray(reconstr_error_std_list), alpha=0.2, color=base_color)
    plt.xlabel("#known attributes")
    plt.ylabel("Reconstruction Error (for unknown attributes)")
    plt.savefig("figures/partial_reconstruction_results_%s.pdf" %dataset, bbox_inches='tight')
    plt.clf()
    #for nattrs in range(nattrs_datasets[dataset]):
        