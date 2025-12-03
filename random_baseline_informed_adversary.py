import pandas as pd
from utils import generate_random_sols, data_splitting, dist_individus
from datasets_infos import datasets_ohe_vectors, predictions
from sklearn.metrics import accuracy_score
import numpy as np


import numpy as np
verbose = False
def pairwise_l1_distances(X):
    """
    X: array of shape (N, d)
    returns D[i, j] = L1 distance between X[i] and X[j]
    """
    # X[:, None, :] shape (N, 1, d), X[None, :, :] shape (1, N, d)
    diff = np.abs(X[:, None, :] - X[None, :, :])
    return diff.sum(axis=2)  # (N, N)

def dataset_medoid(X):
    """
    Return the index and value of the example that minimizes
    the average L1 distance to all examples in X.
    """
    D = pairwise_l1_distances(X)       # (N, N)
    avg_dist = D.mean(axis=1)          # (N,)
    i_star = np.argmin(avg_dist)
    return i_star, X[i_star]

def binary_medoid_any_vector(X):
    """
    X: (N, d) array of 0/1
    Returns the binary vector z* in {0,1}^d that minimizes
    the average Hamming/L1 distance to all rows of X.
    """
    # fraction of ones per feature
    p = X.mean(axis=0)          # shape (d,)

    # majority vote per coordinate
    z_star = (p > 0.5).astype(np.float32)

    # optional: break ties (p == 0.5), e.g. set to 0 or 1 arbitrarily
    # z_star[p == 0.5] = 0.0

    return z_star

list_datasets = ['compas' ,'default_credit', 'adult']
seeds = [0]
for dataset in list_datasets:
    ohe_groups = datasets_ohe_vectors[dataset]
    prediction = predictions[dataset]
    path = f"data/{dataset}.csv"
    data = pd.read_csv(path)
    X = data.drop(columns=[prediction])
    y = data[prediction]
    sample_size = len(X)
    if dataset == 'compas':
        N_samples = 2000
    elif dataset == 'default_credit':
        N_samples = 10000
    elif dataset == 'adult':
        N_samples = 20000
   

    
    reconstructed_samples = []
    per_seed_avg = []
    for seed in seeds: 
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = data_splitting(data, prediction, sample_size - N_samples, seed)


        X_train = X_train.to_numpy()

        #i_star, x_star = dataset_medoid(X_train)
        per_exemple_errors = []
        for ex_id in range(len(X_train)):
            X_train_known = np.delete(X_train, ex_id, axis=0)
            x_inferred = binary_medoid_any_vector(X_train_known)  
            #x_inferred = generate_random_sols(1, X_train.shape[1], dataset_ohe_groups=[], n_sols=1, seed=ex_id)[0][0]
            if verbose:
                print("True sample (exemple %d): " % ex_id, X_train[ex_id].tolist())
                print("Reconstructed Samples: ", x_inferred)

            # Compute reconstruction error
            e_mean_example = dist_individus(x_inferred, X_train[ex_id])

            if verbose:
                print("Reconstruction Error (exemple %d): " % ex_id, e_mean_example)
            
            per_exemple_errors.append(e_mean_example)

        assert(len(per_exemple_errors) == len(X_train))

        per_seed_avg.append(np.mean(per_exemple_errors))

    assert(len(per_seed_avg) == len(seeds))
    print("Dataset: ", dataset, "average random reconstr. error over all examples: ", np.mean(per_seed_avg))
    print("(detailed per-seed averages: ", per_seed_avg, ")")
