# DRAFT-DP

This repository contains the code for the paper "Training Set Reconstruction from Differentially Private Forests: How Effective is DP?". The Arxiv preprint is available [here](https://arxiv.org/abs/2502.05307).

In particular, it contains a class for building differentially private Random Forests and a reconstruction attack against these same forests.

## Setup and dependencies

The proposed dataset reconstruction attack against Differentially Private Random Forests is based upon a Constraint Programming formulation, and uses the`OR-Tools` CP-SAT solver to solve it. Setup instructions are available on: [here](https://developers.google.com/optimization/install/python)
 
The other necessary libraries for the proper functioning of the various modules are listed in the `requirements.txt` file.

## Usage

We provide a simple example use of our module in the `toy_example.py` file to help understand how the different classes work:

```python

from DP_RF import DP_RF # Training algorithm to learn DP RFs
from DP_RF_solver import DP_RF_solver # Attack to reconstruct DP RFs' training data
import pandas as pd
from utils import * 
from datasets_infos import datasets_ohe_vectors, predictions

dataset = "compas"
path = "data/%s.csv" %dataset
ohe_groups = datasets_ohe_vectors[dataset]
N_samples = 100
N_fixed = N_samples  #If N is known, set N_fixed = N_sample, else set N_fixed = None

# DP RF Hyperparameters
N_trees = 5
eps = 30 # DP Budget
depth = 3
seed = 0

time_out = 120 # seconds

np.random.seed(seed)

data = pd.read_csv(path)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
sample_size = len(X)

X_train, X_test, y_train, y_test = data_splitting(data, predictions[dataset], sample_size - N_samples, seed)

# Build a DP RF 
clf = DP_RF(path, dataset, N_samples, N_trees, ohe_groups, depth, seed)

clf.fit(X_train,y_train)

clf.add_noise(eps) 

# Conduct the dataset reconstruction attack against the trained DP RF
solver = DP_RF_solver(clf,eps)

dict_res = solver.fit(N_fixed, seed, time_out, n_threads)

# Retrieve solving time and reconstructed data
duration = dict_res['duration']
x_sol = dict_res['reconstructed_data']

# Evaluate and display the reconstruction rate
e_mean, list_matching = average_error(x_sol,X_train.to_numpy(),seed)

print("Complete solving duration :", duration)
print("Reconstruction Error: ", e_mean)
```

The following output is expected : 

``` bash

Complete solving duration : 0.39089369773864746
Reconstruction Error:  0.26142857142857134

```

## Detailed files description

Here is a description of the various files:

* `DP_RF.py` contains the implementation of the module for constructing differentially private Random Forests.

* `DP_RF_solver.py` contains the implementation of the module for performing the reconstruction attack on DP Random Forests.

* `toy_example.py` is a simple example demonstrating the functionality of the two aforementioned classes.

* `utils.py` contains some functions used during the experiments.

* The `data` directory contains the 3 datasets used in our experiments.

* `datasets_infos.py` groups various information about the datasets used in the experiments (labels, one-hot encoded features, number of features, etc.).

* `baseline.py` calculates the random baseline for reconstruction error and training accuracy. The data is then added to `datasets_infos.py`

* The `experiments_results` and `figures` (or `tables`) directories contain all the results of the experiments and associated figures.

* `run_experiments.py` and `run_experiments_batch.sh` allow reproduction of the obtained results.

* `plots_main.py` reproduces the figures directly related to the experiments.

* `plots_others.py` reproduces the remaining figures appearing in the paper.

* `table1.py` and `plot_fig2.py` can be used to generate Table 1 and Figure 2 (respectively) appearing in the paper.

* `run_experiments_scalability.py` and `run_experiments_scalability_batch.sh` allow reproduction of the obtained results for the scalability appendix.

* `table_scalability.py` produces the tables for the scalability appendix.

* `run_experiments_partial_reconstr.py` and `run_experiments_partial_reconstr_batch.sh` allow reproduction of the obtained results for the partial reconstruction appendix.

* `plot_results_partial_reconstr.py` reproduces the figures for the partial reconstruction appendix.