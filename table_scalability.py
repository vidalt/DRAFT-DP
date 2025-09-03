import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D


list_N_samples = [25, 50, 100, 200, 300, 400, 500]
N_trees = 10
epsilon = 5
depth = 5
list_seed = [0,1,2,3,4]
res_path = "N_fixed" 
list_datasets = ['compas','default_credit', 'adult']

for dataset in list_datasets:

    keys = ["reconstr_error", "acc_train", "acc_test", "time_to_first_solution"] #"generalization_error"
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

        for seed in list_seed:
            res_path = "experiments_results/Results_scalability/scalability_exps%d_%.2f_%d_%d_%d" %(N_trees, epsilon, seed, depth, train_size)

            results_file = f'{res_path}_{dataset}_results.json'
            file_exists = exists(results_file)
            if file_exists:
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
                if not data["solver_status"] in ["FEASIBLE", "OPTIMAL"]:
                    print("Missing data for experiment: ", results_file)
                    continue

                #print(data)
                local_res["reconstr_error"].append(data["reconstruction_error"])
                #local_res["solve_time"].append(data["duration"])
                local_res["acc_train"].append(data["accuracy_train"])
                local_res["acc_test"].append(data["accuracy_test"])
                local_res["time_to_first_solution"].append(data["time_to_first_solution"])
                #local_res["generalization_error"].append(data["accuracy train"] - data["accuracy test"])
        
            else :
                print("missing file %s" %results_file)
                
        assert(len(local_res["reconstr_error"]) <= 5)

        for a_metric in global_res_avg.keys():
            global_res_avg[a_metric].append(np.average(local_res[a_metric]))
            global_res_min[a_metric].append(min(local_res[a_metric]))
            global_res_max[a_metric].append(max(local_res[a_metric]))
            global_res_std[a_metric].append(np.std(local_res[a_metric]))

    from scipy.optimize import curve_fit

    def power(x, a, b, c):
        return a + b * x ** c

    popt, pcov = curve_fit(power, list_N_samples, global_res_avg["time_to_first_solution"], maxfev=5000)
    
    import matplotlib.pyplot as plt 
    
    best_func = "%.2f+%.4f*x**%.2f" %(popt[0], popt[1], popt[2])

    plt.scatter(list_N_samples, global_res_avg["time_to_first_solution"])
    plt.plot(list_N_samples, [popt[0] + popt[1] * x**popt[2]for x in list_N_samples], label=best_func, color='orange')
    plt.legend()
    plt.show()


    import csv 
    with open('tables/scalability_%s.csv' %(dataset), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["", 'Reconstr. Error', '', 'Time to first sol (s)', 'RF Error', ''])
        csv_writer.writerow(["#examples", 'Avg', 'Std', '', 'Train', 'Test'])
        for i, train_size in enumerate(list_N_samples):
            #print("Max. depth " + str(one_depth_val) + " avg solving time is " + str(np.average(times_per_max_depth[one_depth_val])) + ", std is " + str(np.std(times_per_max_depth[one_depth_val])) + "min is " + str(np.min(times_per_max_depth[one_depth_val])) + ", max is " + str(np.max(times_per_max_depth[one_depth_val])))
            csv_writer.writerow([train_size, "%.3f" %global_res_avg["reconstr_error"][i], "%.3f" %global_res_std["reconstr_error"][i], "%.3f" %global_res_avg["time_to_first_solution"][i], "%.3f" %global_res_avg["acc_train"][i], "%.3f" %global_res_avg["acc_test"][i]])
        csv_writer.writerow(["best_fct_fit=", best_func])