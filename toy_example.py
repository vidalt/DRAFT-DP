from DP_RF import DP_RF
from DP_RF_solver import DP_RF_solver
import pandas as pd
from utils import * 
from datasets_infos import datasets_ohe_vectors, predictions

path = "data/compas.csv"
predictions = "recidivate-within-two-years:1"
dataset = "compas"
ohe_groups = datasets_ohe_vectors[dataset]
N_samples = 100
N_fixed = 100  #If N is known, set N_fixed = N_sample, else set N_fixed = None

N_trees = 1
eps = 30
depth = 3
seed = 0

time_out = 100
n_threads = 16
verbosity = 1
obj_active = 1

np.random.seed(seed)

data = pd.read_csv(path)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
sample_size = len(X)

X_train, X_test, y_train, y_test = data_splitting(data, predictions, sample_size - N_samples, seed)

clf = DP_RF(path, dataset, N_samples, N_trees, ohe_groups, depth, seed, verbosity)

clf.fit(X_train,y_train)

clf.add_noise(eps)

solver = DP_RF_solver(clf,eps)

dict_res = solver.fit(N_fixed, seed, time_out, n_threads, verbosity, obj_active)

# Retrieve solving time and reconstructed data
duration = dict_res['duration']
x_sol = dict_res['reconstructed_data']

# Evaluate and display the reconstruction rate
e_mean, list_matching = average_error(x_sol,X_train.to_numpy(),seed)

print("Complete solving duration :", duration)
print("Reconstruction Error: ", e_mean)