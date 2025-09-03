import pandas as pd
from utils import * 
from datasets_infos import datasets_ohe_vectors, predictions
import argparse
import os
import json
import copy
from sklearn.metrics import accuracy_score
import time 

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

data = pd.read_csv(path)
X = data.drop(columns=[prediction])
y = data[prediction]
sample_size = len(X)

X_train, X_test, y_train, y_test = data_splitting(data, prediction, sample_size - N_samples, seed)

n_random_sols=100
n_threads = 25

check = check_ohe(X_train, ohe_groups)

start_1 = time.time()
liste_random_sol = generate_random_sols(X_train.size, X_test.shape[1], dataset_ohe_groups=datasets_ohe_vectors[dataset], n_sols=n_random_sols, seed=seed)

def process_solution(a_sol):
    return average_error(a_sol, X_train, seed)

from concurrent.futures import ProcessPoolExecutor

start_1 = time.time()
with ProcessPoolExecutor(max_workers=n_threads) as executor:
    results = executor.map(process_solution, liste_random_sol)
all_random_errors = [e_mean for e_mean, _ in results]


if verbose:
    print("Average random reconstr: ", np.average(all_random_errors), "+-", np.std(all_random_errors))

res_path = "experiments_results/Results_scalability/scalability_exps"
res_path += "%d_%.2f_%d_%d_%d" %(N_trees, epsilon, seed, depth, N_samples)
results_file = f'{res_path}_{dataset}_results.json'

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        all_results = json.load(f)

all_results[0]["Random_Baseline"] = np.average(all_random_errors)
all_results[0]["Random_Baseline_Std"] = np.std(all_random_errors)


if os.path.exists(results_file):
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)