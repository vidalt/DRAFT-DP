import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
#from datasets_infos import random_reconstruction_error
list_N_samples = [100,200,300,400,500] #[25, 50, 100, 200, 300, 400, 500]
N_trees = 10
epsilon = 5
depth = 5
list_seed = [0,1,2,3,4] # [0,1,2,3,4]
res_path = "N_fixed" 
list_datasets = ['adult', 'compas','default_credit']
size = (10,5)
plt.rcParams.update({'font.size': 14})
for dataset in list_datasets:
    plt.figure(figsize=size)
    keys = ["anytime_errors", "anytime_sols_times", "baseline_reconstr"] #"generalization_error"
    global_res_avg= {}
    global_res_min =  {}
    global_res_max=  {}
    global_res_std=  {}
    for k in keys:
        global_res_avg[k] = []
        global_res_min[k] = []
        global_res_max[k] = []
        global_res_std[k] = []

    for train_size in list_N_samples:
        local_res = {}
        for k in keys:
            local_res[k] = []
        n_seeds = 0
        for seed in list_seed:
            res_path = "experiments_results/Results_scalability/scalability_exps%d_%.2f_%d_%d_%d" %(N_trees, epsilon, seed, depth, train_size)

            results_file = f'{res_path}_{dataset}_results.json'
            file_exists = exists(results_file)
            if file_exists:
                try:
                    f = open(results_file)
                    data = json.load(f)[0]

                    # Just ensure everything is right
                    assert(data["N_samples"] == train_size)
                    assert(data["N_trees"] == N_trees)
                    assert(data["epsilon"] == epsilon)
                    assert(data["seed"] == seed)
                    assert(data["depth"] == depth)
                    assert(data["dataset"] == "data/%s.csv" %dataset)
                    assert(data["obj_active"] == 1)
                    if not(data["solver_status"] in ["FEASIBLE", "OPTIMAL"]):
                        continue
                    n_seeds+=1
                    local_res["anytime_errors"].append(data["anytime_errors"])
                    local_res["anytime_sols_times"].append(data["anytime_sols_times"])
                    local_res["baseline_reconstr"].append(data["Random_Baseline"])
                except json.decoder.JSONDecodeError:
                    print("Problem with file %s" %results_file)
            else :
                print("missing file %s" %results_file)
                
        #assert(len(local_res["reconstr_error"]) == 5) # TODO modify the verif

        '''for a_metric in global_res_avg.keys():
            global_res_avg[a_metric].append(np.average(local_res[a_metric]))
            global_res_min[a_metric].append(min(local_res[a_metric]))
            global_res_max[a_metric].append(max(local_res[a_metric]))
            global_res_std[a_metric].append(np.std(local_res[a_metric]))'''
        # Make an average over the 5 seeds with interpolation
        #assert(len(local_res["anytime_sols_times"]) == len(list_seed))
        #assert(len(local_res["anytime_errors"]) == len(list_seed))
        interpols = [] # per-seed 1-dimensional interpols
        for a_seed in range(n_seeds):
            #interpols.append(interp1d(local_res["anytime_sols_times"][a_seed],local_res["anytime_errors"][a_seed], kind='previous', bounds_error=False, fill_value=(random_reconstruction_error[dataset],local_res["anytime_errors"][a_seed][-1])))
            interpols.append(interp1d(local_res["anytime_sols_times"][a_seed],local_res["anytime_errors"][a_seed], kind='previous', bounds_error=False, fill_value=(local_res["baseline_reconstr"][a_seed],local_res["anytime_errors"][a_seed][-1])))
        
        min_time = 0#min([min(one_seed_times) for one_seed_times in local_res["anytime_sols_times"]])
        max_time = max([max(one_seed_times) for one_seed_times in local_res["anytime_sols_times"]])

        #all_times = np.linspace(min_time, max_time ,100000)
        all_times = np.concatenate(local_res["anytime_sols_times"])
        
        #all_times = np.append(all_times, [0,36200])
        all_times = np.sort(all_times)
        print(dataset, train_size, all_times)
        averaged_errors = []
        std_errors = []
        for j,t in enumerate(all_times):
            time_results = [interpols[i](t) for i in range(n_seeds)]
            averaged_errors.append(np.average(time_results))
            std_errors.append(np.std(time_results))
            #if dataset == "adult" and train_size == 50 and j == 2:
            #    print(time_results)
            #    print(averaged_errors, "+-", std_errors)
            #    plt.scatter([t],[averaged_errors[j]], marker='x')
        averaged_errors = np.array(averaged_errors)
        std_errors = np.array(std_errors)
        plt.plot(all_times, averaged_errors, label="%d" %train_size)
        #plt.fill_between(all_times, averaged_errors - std_errors, averaged_errors + std_errors, alpha=0.2)
    #plt.xlim(10,36200)
    plt.xscale('log')
    #plt.legend(title="$N$")
    plt.xlabel("Running time (s)")
    plt.ylabel("Reconstruction error")
    #plt.title(dataset)
    plt.savefig("figures/plot_anytime_scalability_%s.pdf" %dataset, bbox_inches="tight")
    plt.clf()
    #plt.show()


# Separate legend
legend_elements = []
for train_size in list_N_samples:
    color = plt.plot([], [])[0].get_color()
    legend_elements.append(Line2D([0], [0], marker=None, color=color, lw=5, label="%d" %train_size))

legendFig = plt.figure("Legend plot")
legendFig.legend(title="$N$", handles=legend_elements, loc='center', ncol=5)
plt.axis('off')
legendFig.savefig('figures/plot_anytime_scalability_legend.pdf', bbox_inches='tight')
plt.close()