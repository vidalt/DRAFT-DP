import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import json
from utils import * 
from datasets_infos import datasets_ohe_vectors, predictions, datasets_ordinal_attrs, datasets_numerical_attrs

def generate_random_predictions(num_samples, num_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(low=0, high=num_classes, size=num_samples)

datasets = ["adult", "compas", "default_credit"]
num_classes = 1

for dataset_file in datasets:
    df = pd.read_csv(f"data/{dataset_file}.csv")
    y_col = predictions[dataset_file]
    
    true_labels = df[y_col].values

    num_samples = len(df)

    predicted_labels = generate_random_predictions(num_samples, num_classes)

    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Dataset: {dataset_file}, Random prediction accuracy : {accuracy}")


for N in {'N_fixed', 'N_free'}:
    for dataset in ["compas", "adult", "default_credit"]:
        with open(f'{N}_{dataset}_results.json', 'r') as f:
            results = json.load(f)


        script_seed = 42
        use_bootstrap = False
        sample_size = 125
        train_size = 100

        size = (7,5)

        ohe_vector = datasets_ohe_vectors[dataset] # list of sublists indicating sets of binary attributes one-hot-encoding the same original one
        ordinal_attrs = datasets_ordinal_attrs[dataset] # list of ordinal attributes
        numerical_attrs = datasets_numerical_attrs[dataset] # list of numerical attributes
        prediction = predictions[dataset] # the attribute we want to predict

        df = pd.read_csv("data/%s.csv" %dataset)

        df = df.sample(n=125, random_state = 0, ignore_index= True)
        X_train, X_test, y_train, y_test = data_splitting(df, prediction, test_size=sample_size-train_size, seed=script_seed)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

        nb_random_sols = 100
        random_sols = generate_random_sols(X_train.shape[0], X_train.shape[1], dataset_ohe_groups=ohe_vector, n_sols=nb_random_sols, seed=script_seed)
        rand_sum = 0
        for e in random_sols:
            rand_sum += average_error(e, X_train, seed=script_seed)[0]
        rand_avg = rand_sum/len(random_sols)
        rand_std = np.std([average_error(e, X_train, seed=script_seed)[0] for e in random_sols])
        print("Baseline (Random) Error: ", rand_avg)
        print("Baseline (Random) Std: ", rand_std)