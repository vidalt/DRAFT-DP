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

list_N_samples = [25, 50, 100, 200, 300, 400, 500]
list_N_trees = [10]
list_epsilon = [5]
list_obj_active = [1]
list_depth = [5]
list_seed = [0,1,2,3,4]
list_datasets = ['compas','default_credit', 'adult']

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
    print("obj_active :", obj_active)
    print("dataset :", path)
    print("seed :", seed)
    print("Total #expes: ", len(list_config))

# Solver parameters
verbosity = int(verbose)
n_threads = 48
time_out = 36000


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

if verbose:
    print("Noisy tree computed, creating the reconstructor.")

accuracy_test = accuracy_score(y_test, clf.predict(X_test))
accuracy_train = accuracy_score(y_train, clf.predict(X_train))

# Solve the reconstruction problem
solver = DP_RF_solver(clf,epsilon)

if verbose:
    print("Calling the fit.")

dict_res = solver.fit(N_fixed, seed, time_out, n_threads, verbosity, obj_active)

if verbose:
    print("Fitted.")

# Retrieve solving time and reconstructed data
duration = dict_res['duration']
try:

    N_reconstruit = 0
    N_min = 0
    N_max = 0
    e_mean = 1
    success = False
    check = check_ohe(X_train, ohe_groups)

    if dict_res['status'] == 'OPTIMAL' or dict_res['status'] == 'FEASIBLE':
        e_mean, list_matching = average_error(dict_res['reconstructed_data'],X_train.to_numpy(), seed)

        '''anytime_errors = []
        for a_sol in dict_res['anytime_sols']:
            e_mean_a_sol, _ = average_error(a_sol,X_train, seed)
            anytime_errors.append(e_mean_a_sol)'''
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = executor.map(lambda a_sol: average_error(a_sol, X_train, seed), dict_res['anytime_sols'])

        anytime_errors = [e_mean for e_mean, _ in results]

        success = clf_unnoise.format_nb() == dict_res['nb_recons']
        if verbose:
            print("Complete solving duration :", dict_res['duration'])
            print("Reconstruction Error: ", e_mean)
        N_min = dict_res['N_min']
        N_max = dict_res['N_max']
        N_reconstruit = dict_res['N']
        
        if verbose:
            print("Complete solving duration :", duration)
            print("Reconstruction Error: ", e_mean)

    dict_res = {
        "N_samples": N_samples,
        "N_trees": N_trees,
        "epsilon": epsilon,
        "epsilon_etape": epsilon/N_trees,
        "obj_active": obj_active,
        "reconstruction_error": e_mean,
        "time_to_first_solution": dict_res['time_to_first_solution'],
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
        "id": expe_id,
        'anytime_errors' : anytime_errors,
        'anytime_sols_times' : dict_res['anytime_sols_times']                           
    }
except:
    if verbosity:
        print("Solver failed to retrieve feasible solution.")
    dict_res = {
        "N_samples": N_samples,
        "N_trees": N_trees,
        "epsilon": epsilon,
        "epsilon_etape": epsilon/N_trees,
        "obj_active": obj_active,
        "reconstruction_error": "NA",
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

res_path = "experiments_results/Results_scalability/scalability_exps"

results_file = f'{res_path}_{dataset}_results.json'

try:
    all_results = []
    all_results.append(dict_res)
    res_path += "%d_%.2f_%d_%d_%d" %(N_trees, epsilon, seed, depth, N_samples)
    results_file = f'{res_path}_{dataset}_results.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

except json.decoder.JSONDecodeError:
    all_results = []      
    all_results.append(dict_res)
    res_path += "%d_%.2f_%d_%d" %(N_trees, epsilon, seed, depth)
    results_file = f'{res_path}_{dataset}_results.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
