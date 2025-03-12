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

list_N_samples = [10]
list_N_trees = [10]
list_epsilon = [5]
list_obj_active = [1]
list_depth = [5]
list_seed = [0,1,2,3,4]
list_datasets = ['compas','default_credit', 'adult']
nattrs_datasets = {"compas":14-4-3, "adult":19-5, "default_credit":21-3-2} # All attributes OHEncoding the same original feature count only once

list_config = []

for obj_active_bool in list_obj_active:
    for depth in list_depth:
        for Ntrees in list_N_trees:
            for epsi in list_epsilon: 
                for Nsamp in list_N_samples:
                    for dataset in list_datasets:
                        for seed in list_seed:
                            for nattrs in range(nattrs_datasets[dataset]):
                                list_config.append([Ntrees, epsi, Nsamp, obj_active_bool, f"data/{dataset}.csv", seed,depth, dataset,nattrs])

N_trees = list_config[expe_id][0]
epsilon = list_config[expe_id][1]
N_samples = list_config[expe_id][2]
N_fixed = N_samples #If N is known, set N_fixed = N_samples, else set N_fixed = None
obj_active = list_config[expe_id][3]
path = list_config[expe_id][4]
seed = list_config[expe_id][5]
depth = list_config[expe_id][6]
dataset = list_config[expe_id][7]
known_attributes_nb = list_config[expe_id][8]

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
n_threads = 16
time_out = 7200

data = pd.read_csv(path)
X = data.drop(columns=[prediction])
y = data[prediction]
sample_size = len(X)

X_train, X_test, y_train, y_test = data_splitting(data, prediction, sample_size - N_samples, seed)

assert(nattrs_datasets[dataset] == X_train.shape[1] + len(ohe_groups) - sum([len(one_ohe) for one_ohe in ohe_groups]))

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

# everything below is to convert the choice of original features into the actual choice of encoded (with OHE) attrs
known_original_attributes_list = np.random.choice([i for i in range(nattrs_datasets[dataset])],size=known_attributes_nb, replace=False) # randomly pick the known_attributes_nb (original) attributes we assume knowledge of
original_attributes_list = []
one_encoded_attr = 0
while one_encoded_attr < X_train.shape[1]:
    for one_ohe in ohe_groups:
        if one_encoded_attr in one_ohe:
            original_attributes_list.append(one_ohe)
            one_encoded_attr = max(one_ohe) + 1
            continue
    original_attributes_list.append([one_encoded_attr])
    one_encoded_attr += 1
known_attributes_list = []
for one_original_attr in known_original_attributes_list:
    known_attributes_list.extend(original_attributes_list[one_original_attr])
unknown_attributes_list = list(set([i for i in range(X_train.shape[1])]) - set(known_attributes_list))
# -----------------------------------------------------------------------------------------------------------------

dict_res = solver.fit(N_fixed, seed, time_out, n_threads, verbosity, obj_active, X_known = np.asarray(X_train), known_attributes=known_attributes_list)

if verbose:
    print("Fitted.")

# Retrieve solving time and reconstructed data
duration = dict_res['duration']
#try:

N_reconstruit = 0
N_min = 0
N_max = 0
e_mean = 1
success = False


if dict_res['status'] == 'OPTIMAL' or dict_res['status'] == 'FEASIBLE':
    x_sol = dict_res['reconstructed_data']
    
    check = check_ohe(pd.DataFrame(x_sol), ohe_groups)

    x_sol = np.asarray(x_sol)
    X_train = np.asarray(X_train)
    e_mean, list_matching = average_error(x_sol,X_train, seed) # computes the matching over the entire features vectors...
    print(e_mean)
    x_sol = x_sol[:,unknown_attributes_list]
    X_train = X_train[:,unknown_attributes_list]
    x_train_list_unknown = X_train.tolist() # ground truth
    moyenne = 0
    for i in range(len(x_train_list_unknown)):
        moyenne += dist_individus(x_sol[i], x_train_list_unknown[list_matching[i]]) # ... but only report distance for unknown attributes
    moyenne = moyenne/len(x_train_list_unknown)
    e_mean = moyenne
    print(e_mean)
    success = clf_unnoise.format_nb() == dict_res['nb_recons']

    if verbose:
        print("Complete solving duration :", dict_res['duration'])
        print("Reconstruction Error: ", e_mean)

    N_min = dict_res['N_min']
    N_max = dict_res['N_max']
    N_reconstruit = dict_res['N']
    
    dict_res = {
        "N_samples": N_samples,
        "known_attributes": known_attributes_nb,
        "N_trees": N_trees,
        "epsilon": epsilon,
        "epsilon_etape": epsilon/N_trees,
        "obj_active": obj_active,
        "reconstruction_error": e_mean,
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
    }
else:
    if verbosity:
        print("Solver failed to retrieve feasible solution.")
    dict_res = {
        "N_samples": N_samples,
        "known_attributes": known_attributes_nb,
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

res_path = "experiments_results/results_partial_reconstruction/partial_reconstr_exps"

results_file = f'{res_path}_{dataset}_results.json'

try:
    all_results = []
    all_results.append(dict_res)
    res_path += "%d_%.2f_%d_%d_%d" %(N_trees, epsilon, seed, depth, known_attributes_nb)
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
