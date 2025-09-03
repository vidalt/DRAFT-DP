from sklearn.ensemble import RandomForestClassifier
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

for dataset in ["compas", "adult", "default_credit"]:
    N_trees = 10
    N_samples = 100
    N_fixed = N_samples #If Nlist_config[expe_id][3] is known, set N_fixed = N_samples, else set N_fixed = None
    path = f"data/{dataset}.csv"
    seeds = [0,1,2,3,4]
    depth = 5


    prediction = predictions[dataset]
    train_accs = []
    test_accs = []
    for seed in seeds:
        np.random.seed(seed)

        if verbose:
            print("N_trees :", N_trees)
            print("N_samples :", N_samples)
            print("dataset :", path)
            print("seed :", seed)

        data = pd.read_csv(path)
        X = data.drop(columns=[prediction])
        y = data[prediction]
        sample_size = len(X)

        X_train, X_test, y_train, y_test = data_splitting(data, prediction, sample_size - N_samples, seed)

        # Creation of a DP RF
        clf = RandomForestClassifier(n_estimators=N_trees, max_depth=depth, random_state=seed)
        clf.fit(X_train,y_train)

        accuracy_test = accuracy_score(y_test, clf.predict(X_test))
        accuracy_train = accuracy_score(y_train, clf.predict(X_train))

        train_accs.append(accuracy_train)
        test_accs.append(accuracy_test)
    print("Dataset %s" %dataset)
    print("sklearn")
    print("$%.2f \pm %.2f$" %(np.average(train_accs), np.std(train_accs)))
    print("$%.2f \pm %.2f$"%(np.average(test_accs), np.std(test_accs)))
    print("DP RF")
    train_accs = []
    test_accs = []
    with open(f'experiments_results/Results_main_paper/N_fixed_{dataset}_results.json', 'r') as f:
        results = json.load(f)
        
        # Iterate through results to extract epsilon, N_trees, and accuracy_train
        for result in results:
            if result['depth'] == 5 and result['N_trees'] == 10 and result["epsilon"] == 30:
                train_accs.append(result['accuracy_train'])
                test_accs.append(result['accuracy_test'])
    assert(len(train_accs) == len(seeds))
    assert(len(test_accs) == len(seeds))
    print("$%.2f \pm %.2f$" %(np.average(train_accs), np.std(train_accs)))
    print("$%.2f \pm %.2f$"%(np.average(test_accs), np.std(test_accs)))