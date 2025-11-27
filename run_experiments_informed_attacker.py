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
list_epsilon = [0.1, 1, 5, 10, 20, 30, 1000]
list_obj_active = [1]
list_depth = [5]
list_seed = [0,1,2,3,4]
list_datasets = ['compas' ,'default_credit', 'adult']
target_ratio_divisors = [0.001] #[1, 2, 5]
list_config = []

for obj_active_bool in list_obj_active:
    for depth in list_depth:
        for Ntrees in list_N_trees:
            for epsi in list_epsilon: 
                for Nsamp in list_N_samples:
                    for dataset in list_datasets:
                        for seed in list_seed:
                            for target_ratio_divisor in target_ratio_divisors:
                                list_config.append([Ntrees, epsi, Nsamp, obj_active_bool, f"data/{dataset}.csv", seed,depth, dataset, target_ratio_divisor])

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
target_ratio_divisor = list_config[expe_id][8]

target_ratio = epsilon/target_ratio_divisor
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
time_out = 3600

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


accuracy_test = accuracy_score(y_test, clf.predict(X_test))
accuracy_train = accuracy_score(y_train, clf.predict(X_train))

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

durations = []
per_exemple_errors = []
status_list = []
nb_failures = 0

# Solve the reconstruction problem
for ex_id in range(N_samples):
    solver = DP_RF_solver(clf, epsilon)
    dict_res_ = solver.fit(N_fixed, seed, time_out, n_threads, verbosity, obj_active, X_partial_expe=X_train, y_partial_expe=y_train, ex_id=ex_id, target_ratio=target_ratio)

    if dict_res_['status'] == 'OPTIMAL' or dict_res_['status'] == 'FEASIBLE':
        # Retrieve solving time and reconstructed data
        duration = dict_res_['duration']
        
        # Compute reconstruction error
        e_mean_example = dist_individus(dict_res_['reconstructed_data'][ex_id], X_train[ex_id])

        if verbose:
            print("Complete solving duration :", duration)
            print("Reconstruction Error (exemple %d): " % ex_id, e_mean_example)
            print("Reconstructed Example %d: " % ex_id, dict_res_['reconstructed_data'][ex_id])
            print("True          Example %d: " % ex_id, X_train[ex_id])
        
        per_exemple_errors.append(e_mean_example)
        
    else:
        if verbosity:
            print("Solver failed to retrieve feasible solution for exemple %d." % ex_id)
        nb_failures += 1
        per_exemple_errors.append(-1)
    
    durations.append(dict_res_['duration'])
    status_list.append(dict_res_['status'])
    

dict_res = {
    "N_samples": N_samples,
    "N_trees": N_trees,
    "epsilon": epsilon,
    "epsilon_etape": epsilon/N_trees,
    "obj_active": obj_active,
    "example_reconstruction_error_avg": np.mean(per_exemple_errors),
    "example_reconstruction_error_list":per_exemple_errors,
    "nb_failures": nb_failures,
    "duration_avg": np.mean(durations),
    "duration_list": durations,
    "status_list": status_list,
    "accuracy_train": accuracy_train,
    "accuracy_test": accuracy_test,
    "dataset": path,
    "time_out": time_out,
    "seed": seed,
    "depth": depth,
    "id": expe_id,
    'target_ratio_divisor': target_ratio_divisor                    
}

res_path = "N_fixed" if N_fixed is not None else "N_free"
res_path += "%d_%.2f_%d_%d_%.3f" %(N_trees, epsilon, seed, depth, target_ratio_divisor)
if N_fixed is not None:
    results_file = f'experiments_results/Results_informed_adversary/{res_path}_{dataset}_results.json'
else:
    results_file = f'experiments_results/Results_main_paper_N_unknown/{res_path}_{dataset}_results.json'

all_results = []      
all_results.append(dict_res)


with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=4)
