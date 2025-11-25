from DP_RF import DP_RF
from DP_RF_solver import DP_RF_solver
import pandas as pd
from utils import * 
from datasets_infos import datasets_ohe_vectors, predictions
import argparse
import os
import json
import copy
from sklearn.metrics import accuracy_score

verbose = False 

parser = argparse.ArgumentParser(description='Dataset reconstruction from random forest')
parser.add_argument('--expe_id', type=int, default=0)
args = parser.parse_args()
expe_id=args.expe_id

list_N_samples = [100]
list_N_trees = [10]
list_epsilon = [1000]
list_obj_active = [1]
list_depth = [5]
list_seed = [0,1,2,3,4]
list_datasets = ['compas' ,'default_credit', 'adult']

list_config = []

for obj_active_bool in list_obj_active:
    for depth in list_depth:
        for Ntrees in list_N_trees:
            for epsi in list_epsilon: 
                for Nsamp in list_N_samples:
                    for dataset in list_datasets:
                        for seed in list_seed:
                            list_config.append([Ntrees, epsi, Nsamp, obj_active_bool, f"data/{dataset}.csv", seed,depth, dataset])

N_trees = list_config[expe_id][0]
epsilon = list_config[expe_id][1]
N_samples = list_config[expe_id][2]
N_fixed = N_samples #If N is known, set N_fixed = N_samples, else set N_fixed = None
obj_active = list_config[expe_id][3]
path = list_config[expe_id][4]
seed = list_config[expe_id][5]
depth = list_config[expe_id][6]
dataset = list_config[expe_id][7]

ohe_groups = datasets_ohe_vectors[dataset]
prediction = predictions[dataset]

np.random.seed(seed)

if verbose:
    print("N_trees :", N_trees)
    print("epsilon :", epsilon)
    print("N_samples :", N_samples)
    print("Max Depth = ", depth)
    print("obj_active :", obj_active)
    print("dataset :", path)
    print("seed :", seed)
    print("#configs = ", len(list_config))

# Solver parameters
verbosity = int(verbose)
n_threads = 16
time_out = 7200

data = pd.read_csv(path)
X = data.drop(columns=[prediction])
y = data[prediction]
sample_size = len(X)

X_train, X_test, y_train, y_test = data_splitting(data, prediction, sample_size - N_samples, seed)

# Creation of a DP RF
clf = DP_RF(path, dataset, N_samples, N_trees, ohe_groups, depth, seed, verbosity)
clf.fit(X_train,y_train)

# Store the unnnoised RF
clf_unnoise = copy.deepcopy(clf)

clf.add_noise(epsilon)

#print(clf_unnoise.format_nb(), "(%d feuilles)" %(len(clf_unnoise.format_nb()[0])))
#print(clf.format_nb(), "(%d feuilles)" %(len(clf.format_nb()[0])))

# Solve the reconstruction problem
solver = DP_RF_solver(clf,epsilon)
dict_res = solver.fit(N_fixed, seed, time_out, n_threads, verbosity, obj_active)

# Evaluate actual reconstruction
N_reconstruit = 0
N_min = 0
N_max = 0
e_mean = 1
success = False
check = check_ohe(X_train, ohe_groups)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

accuracy_test = accuracy_score(y_test, clf.predict(X_test))
accuracy_train = accuracy_score(y_train, clf.predict(X_train))

if dict_res['status'] == 'OPTIMAL' or dict_res['status'] == 'FEASIBLE':
    # Retrieve solving time and reconstructed data
    duration = dict_res['duration']
    
    # Compute reconstruction error
    e_mean, list_matching, all_distances = average_error(dict_res['reconstructed_data'],X_train, seed, return_all_distances=True)

    if verbose:
        print("Complete solving duration :", duration)
        print("Reconstruction Error: ", e_mean)
    
    if N_fixed is not None:
        # New
        anytime_errors = []
        for a_sol in dict_res['anytime_sols']:
            e_mean_a_sol, _ = average_error(a_sol,X_train, seed)
            anytime_errors.append(e_mean_a_sol)

        # New
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(random_state=0)
        clf.fit(X_train)
        all_distances_outlier_scores = clf.decision_function([X_train[i] for i in list_matching]) # Average anomaly score of X of the base classifiers.

    success = clf_unnoise.format_nb() == dict_res['nb_recons']
    if verbose:
        print("Complete solving duration :", dict_res['duration'])
        print("Reconstruction Error: ", e_mean)
    N_min = dict_res['N_min']
    N_max = dict_res['N_max']
    N_reconstruit = dict_res['N']

    # Additional metrics:
    # (i) Distribution-aware baseline
    # (ii) From-distribution comparative matching
    if N_fixed is not None:
        n_random = 100
        e_baseline_distrib = 0
        e_mean_distrib = 0
        list_reconstr_errors_random_datasets = []
        for i in range(n_random):
            #print("sampling randomly %d examples from %d" %(N_samples, X_test.shape[0]))
            sampled_indices = np.random.choice(X_test.shape[0], size=N_samples, replace=False)
            sampled_from_distrib = X_test[sampled_indices]

            baseline_distrib, _ = average_error(sampled_from_distrib,X_train, seed) # compute error between X_train and another dataset from the same distrib
            e_distrib, _ = average_error(dict_res['reconstructed_data'],sampled_from_distrib, seed) # compute error between reconstructed data and another dataset from the same distrib

            e_baseline_distrib += baseline_distrib
            e_mean_distrib += e_distrib
            list_reconstr_errors_random_datasets.append(e_distrib)

        e_baseline_distrib /= n_random
        e_mean_distrib /= n_random
        all_distances_outlier_scores = list(all_distances_outlier_scores)
    dict_res = {
        "N_samples": N_samples,
        "N_trees": N_trees,
        "epsilon": epsilon,
        "epsilon_etape": epsilon/N_trees,
        "obj_active": obj_active,
        "reconstruction_error": e_mean,
        "all_distances":all_distances,
        "all_distances_x_train_ids":list_matching,
        "duration": dict_res['duration'],
        "solver_status": dict_res['status'],
        "N_reconstruit": N_reconstruit,
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "dataset": path,
        "time_out": time_out,
        "check_ohe": check,
        "succes_total_debruitage": success,
        "seed": seed,
        "depth": depth,
        "id": expe_id                     
    }

    if N_fixed is not None:
        dict_res['anytime_sols_times'] =  dict_res['anytime_sols_times'] 
        dict_res['anytime_errors'] =  dict_res['anytime_errors'] 
        dict_res['time_to_first_solution'] =  dict_res['time_to_first_solution']
        dict_res['reconstruction_error_matching_distrib_list'] =  list_reconstr_errors_random_datasets
        dict_res['reconstruction_baseline_distrib'] =  e_baseline_distrib # the reconstruction value for a baseline knowing the distrib (matching other datasets from the same distrib.)
        dict_res['reconstruction_error_matching_distrib'] =  e_mean_distrib # the reconstruction value when matching with other datasets from the same distrib
        dict_res['all_distances_outlier_scores'] =  all_distances_outlier_scores
        
else:
    if verbosity:
        print("Solver failed to retrieve feasible solution.")
    dict_res = {
        "N_samples": N_samples,
        "N_trees": N_trees,
        "epsilon": epsilon,
        "epsilon_etape": epsilon/N_trees,
        "obj_active": obj_active,
        "reconstruction_error": "NA",
        "reconstruction_error_matching_distrib": "NA", # the reconstruction value when matching with other datasets from the same distrib
        "reconstruction_baseline_distrib": "NA", # the reconstruction value for a baseline knowing the distrib (matching other datasets from the same distrib.)
        "time_to_first_solution": "NA",
        "duration": dict_res['duration'],
        "solver_status": dict_res['status'],
        "N_reconstruit": "NA",
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "dataset": path,
        "time_out": time_out,
        "check_ohe": "NA",
        "succes_total_debruitage": "NA",
        "seed": seed,
        "depth": depth,
        "id": expe_id,
    }

res_path = "N_fixed" if N_fixed is not None else "N_free"
res_path += "%d_%.2f_%d_%d" %(N_trees, epsilon, seed, depth)
if N_fixed is not None:
    results_file = f'experiments_results/Results_main_paper/{res_path}_{dataset}_results.json'
else:
    results_file = f'experiments_results/Results_main_paper_N_unknown/{res_path}_{dataset}_results.json'

'''try:
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
        
    all_results.append(dict_res)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

except json.decoder.JSONDecodeError: # just in case, do not lose the computed results!'''

all_results = []      
all_results.append(dict_res)


with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=4)
