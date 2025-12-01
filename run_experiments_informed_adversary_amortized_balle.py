from DP_RF import DP_RF
from balle_informed_adversary_amortized_attack import train_reconstructor, serialize_forest_leaf_counts_minus_known_fixed
import pandas as pd
from utils import * 
from datasets_infos import datasets_ohe_vectors, predictions
import argparse
import os
import json
import copy
from sklearn.metrics import accuracy_score
import torch 
import time 

verbose = False 
n_threads = 16
parser = argparse.ArgumentParser(description='Dataset reconstruction from random forest')
parser.add_argument('--expe_id', type=int, default=0)
args = parser.parse_args()
expe_id=args.expe_id

list_N_samples = [10000,20000]#[100]
list_N_trees = [10]
list_epsilon = [1000,30,20,10,5,1,0.1]
list_obj_active = [1]
list_depth = [5]
list_seed = [0,1,2,3,4]
list_datasets = ['default_credit', 'adult'] #'compas' 

list_config = []

for obj_active_bool in list_obj_active:
    for depth in list_depth:
        for Ntrees in list_N_trees:
            for epsi in list_epsilon: 
                for Nsamp in list_N_samples:
                    for dataset in list_datasets:
                        for seed in list_seed:
                            if (dataset == 'compas' and Nsamp > 2000) or (dataset == 'default_credit' and Nsamp > 10000):
                                continue
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

if True:
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

data = pd.read_csv(path)
X = data.drop(columns=[prediction])
y = data[prediction]
sample_size = len(X)

X_train, X_test, y_train, y_test = data_splitting(data, prediction, sample_size - N_samples, seed)

# Creation of a DP RF
clf = DP_RF(path, dataset, N_samples, N_trees, ohe_groups, depth, seed, verbosity, seed_noise=seed)
clf.fit(X_train,y_train)

# Store the unnnoised RF
clf_unnoise = copy.deepcopy(clf)

clf.add_noise(epsilon)

'''from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plot_tree(clf.clf.estimators_[0])
plt.savefig("figures/balle_tree_plot_origRF.pdf")
plt.clf()'''
accuracy_test = accuracy_score(y_test, clf.predict(X_test))
accuracy_train = accuracy_score(y_train, clf.predict(X_train))

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

per_exemple_errors = []
reconstructed_samples = []
def project_to_binary(x_continuous, threshold=0.5):
    return (x_continuous >= threshold).astype(np.float32)
start = time.time()
mean, std = None, None
reconstructor = train_reconstructor(
        X=X_test,
        y=y_test.to_numpy(),
        dp_rf_class=DP_RF,
        dataset_name=dataset,
        ohe_groups=ohe_groups,
        n_trees=N_trees,
        max_depth=depth,
        eps=epsilon,
        n_pairs=1600,
        batch_size=32,
        n_epochs=5,
        lr=1e-3,
        device="cpu",
        mean=mean,      
        std=std,
        N_samples=N_samples,
        seed=seed,
        n_threads=n_threads
    )
#######################################################

# If N_samples is small enough, reconstruct all examples
if N_samples <= 1000:
    subsampled_examples = list(range(N_samples))
else:
    subsampled_examples = np.random.choice(range(N_samples), size=100, replace=False)

    
# Solve the reconstruction problem
for ex_id in subsampled_examples:
    
    # TRAIN THE RECONSTRUCTION NETWORK (using data from the same distribution == the very large test set is fine)
    mask = np.array([i for i in range(N_samples) if i != ex_id])
    x_star = X_train[ex_id]

    ##################################
    # RUN THE ATTACK ON THE TRAINED DP RF
    forest_vec = serialize_forest_leaf_counts_minus_known_fixed(
        clf,
        X_train[mask],
        y_train[mask],
        n_classes=2,
        max_leaves=2 ** depth,
        mean=mean,  
        std=std,
    ) 

    forest_tensor = torch.from_numpy(forest_vec.astype(np.float32)).unsqueeze(0)
    reconstructor.eval()
    with torch.no_grad():
        #x_hat = reconstructor(forest_tensor).cpu().numpy()[0]  # reconstructed x*
        #x_hat_binary = project_to_binary(x_hat)
        logits = reconstructor(forest_tensor)      # (1, d)
        probs = torch.sigmoid(logits)[0]   # (d,)
        x_hat_binary = (probs >= 0.5).cpu().numpy().astype(np.float32)
    
    if verbose:
        print("Reconstructed: ", x_hat_binary)
        
    # Compute reconstruction error
    e_mean_example = dist_individus(x_hat_binary, X_train[ex_id])

    if verbose:
        print("Reconstruction Error (exemple %d): " % ex_id, e_mean_example)
        
    per_exemple_errors.append(e_mean_example)
    reconstructed_samples.append(x_hat_binary)

duration = time.time() - start
    
dict_res = {
    "N_samples": N_samples,
    "N_trees": N_trees,
    "epsilon": epsilon,
    "epsilon_etape": epsilon/N_trees,
    "obj_active": obj_active,
    "example_reconstruction_error_avg": np.mean(per_exemple_errors),
    "example_reconstruction_error_list":per_exemple_errors,
    "duration_total": duration,
    "accuracy_train": accuracy_train,
    "accuracy_test": accuracy_test,
    "dataset": path,
    "seed": seed,
    "depth": depth,
    "id": expe_id                     
}

res_path = "N_fixed" if N_fixed is not None else "N_free"
res_path += "%d_%.2f_%d_%d_%d" %(N_trees, epsilon, seed, depth, N_samples)
if N_fixed is not None:
    results_file = f'experiments_results/Results_informed_adversary_balle/{res_path}_{dataset}_results.json'
else:
    results_file = f'experiments_results/Results_main_paper_N_unknown/{res_path}_{dataset}_results.json'

all_results = []      
all_results.append(dict_res)


with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=4)
