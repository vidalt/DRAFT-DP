import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D


list_N_samples = [100]
N_trees = 10#[1,5,10,20,30]
list_epsilon = [30,20,10,5,1,0.1]
list_obj_active = [1]
depth = 5 #[3,5,7]
list_seed = [0,1,2,3,4]
list_datasets = ['compas','default_credit', 'adult']
train_size = 100

for dataset in list_datasets:

    keys = ["reconstr_error", "acc_train", "acc_test", "time_to_first_solution", "nb_perfect_reconstrs", "worst_reconstr", "inliers_avg_reconstr", "outliers_avg_reconstr", "inliers_prop_perfect_reconstr", "outliers_prop_perfect_reconstr", "outlier_prop"] #"generalization_error"
    global_res_avg= {}
    global_res_min =  {}
    global_res_max=  {}
    global_res_std=  {}
    for k in keys:
        global_res_avg[k] = []
        global_res_min[k] = []
        global_res_max[k] = []
        global_res_std[k] = []

    list_epsilon_read = []
    for epsilon in list_epsilon:
        local_res = {}
        for k in keys:
            local_res[k] = []

        for seed in list_seed:
            res_path = "N_fixed"
            res_path += "%d_%.2f_%d_%d" %(N_trees, epsilon, seed, depth)
            results_file = f'experiments_results/Results_individual_reconstruction/{res_path}_{dataset}_results.json'

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
                assert(data["solver_status"] in ["FEASIBLE", "OPTIMAL"])

                #print(data)
                local_res["reconstr_error"].append(data["reconstruction_error"])
                #local_res["solve_time"].append(data["duration"])
                local_res["acc_train"].append(data["accuracy_train"])
                local_res["acc_test"].append(data["accuracy_test"])
                local_res["time_to_first_solution"].append(data["time_to_first_solution"])
                #local_res["generalization_error"].append(data["accuracy train"] - data["accuracy test"])
                local_res["nb_perfect_reconstrs"].append(data["all_distances"].count(0))
                local_res["worst_reconstr"].append(max(data["all_distances"]))

                # Outlier analysis
                data["all_distances_outlier_scores"] = np.asarray(data["all_distances_outlier_scores"])
                data["all_distances"] = np.asarray(data["all_distances"])
                outliers = np.where(data["all_distances_outlier_scores"] <= 0)
                inliers = np.where(data["all_distances_outlier_scores"] > 0)
                outliers_avg = np.average(data["all_distances"][outliers])
                inliers_avg = np.average(data["all_distances"][inliers])
                outliers_prop_perfect_reconstr = 100*(list(data["all_distances"][outliers]).count(0) / len(outliers[0])) # percentage
                inliers_prop_perfect_reconstr = 100*(list(data["all_distances"][inliers]).count(0) / len(inliers[0])) # percentage
                outlier_prop = len(outliers[0])/train_size
                local_res["outliers_avg_reconstr"].append(outliers_avg)
                local_res["inliers_avg_reconstr"].append(inliers_avg)
                local_res["outliers_prop_perfect_reconstr"].append(outliers_prop_perfect_reconstr)
                local_res["inliers_prop_perfect_reconstr"].append(inliers_prop_perfect_reconstr)
                local_res["outlier_prop"].append(outlier_prop)
            else :
                print("missing file %s" %results_file)
                
        if not(len(local_res["reconstr_error"]) == 5):
            continue

        list_epsilon_read.append(epsilon)

        for a_metric in global_res_avg.keys():
            global_res_avg[a_metric].append(np.average(local_res[a_metric]))
            global_res_min[a_metric].append(min(local_res[a_metric]))
            global_res_max[a_metric].append(max(local_res[a_metric]))
            global_res_std[a_metric].append(np.std(local_res[a_metric]))
            

    import csv 
    with open('tables/individual_reconstrs_%s.csv' %(dataset), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Epsilon", 'Reconstr_Error_Avg+-Reconstr_Error_Std', 'NB_PERFECT_AVG+-NB_PERFECT_STD', 'WORST_AVG+-WORST_STD',  "INLIERS_ERR_AVG+-INLIERS_ERR_STD", "OUTLIERS_ERR_AVG+-OUTLIERS_ERR_STD", "INLIERS_PROP_PERFECT_AVG+-INLIERS_PROP_PERFECT_STD", "OUTLIERS_PROP_PERFECT_AVG+-OUTLIERS_PROP_PERFECT_STD", "OUTLIERS_PROP_AVG+-OUTLIERS_PROP_STD"])
        for i, epsilon in enumerate(list_epsilon_read):
            #print("Max. depth " + str(one_depth_val) + " avg solving time is " + str(np.average(times_per_max_depth[one_depth_val])) + ", std is " + str(np.std(times_per_max_depth[one_depth_val])) + "min is " + str(np.min(times_per_max_depth[one_depth_val])) + ", max is " + str(np.max(times_per_max_depth[one_depth_val])))
            csv_writer.writerow([epsilon, "$%.3f \pm %.3f$" %(global_res_avg["reconstr_error"][i], global_res_std["reconstr_error"][i]), "$%.1f \pm %.1f$" %(global_res_avg["nb_perfect_reconstrs"][i], global_res_std["nb_perfect_reconstrs"][i]), "$%.3f \pm %.3f$" %(global_res_avg["worst_reconstr"][i], global_res_std["worst_reconstr"][i]), "$%.3f \pm %.3f$" %(global_res_avg["inliers_avg_reconstr"][i], global_res_std["inliers_avg_reconstr"][i]), "$%.3f \pm %.3f$" %(global_res_avg["outliers_avg_reconstr"][i], global_res_std["outliers_avg_reconstr"][i]), "$%.1f \pm %.1f$" %(global_res_avg["inliers_prop_perfect_reconstr"][i], global_res_std["inliers_prop_perfect_reconstr"][i]), "$%.1f \pm %.1f$" %(global_res_avg["outliers_prop_perfect_reconstr"][i], global_res_std["outliers_prop_perfect_reconstr"][i]), "$%.3f \pm %.3f$" %(global_res_avg["outlier_prop"][i], global_res_std["outlier_prop"][i])]) 
