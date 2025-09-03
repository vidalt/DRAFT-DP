def data_splitting(data, label, test_size, seed):
    """
    Splits data between train and test sets using the label column as prediction and test_size examples for the test set (can be a proportion as well).
    """
    from sklearn.model_selection import train_test_split
    y = data[label]
    X = data.drop(labels = [label], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size , shuffle = False, random_state = seed)
    return X_train, X_test, y_train, y_test

def dist_individus(ind1,ind2, non_binary_attrs=[]):
    """
    Computes the distance between two examples.
    Binary attributes: manhattan distance
    Ordinal/numerical attributes: normalized distance (abs difference between the two values divided by attribute range)
    """
    nbfd = 0 # Error counter
    m = len(ind1)
    num_indices = []
    for f in non_binary_attrs:
        num_indices.append(f[0])
    
    for i in range(m):
        if i in num_indices:
            diff = abs( ind1[i] - ind2[i] )
            idx = num_indices.index(i)
            diffrange = non_binary_attrs[idx][2] - non_binary_attrs[idx][1]
            nbfd += diff / diffrange
        else:
            if ind1[i]!=ind2[i]:
                nbfd += 1.0
    return nbfd/m


def matrice_matching(x_sol,x_train, non_binary_attrs=[]):
    """
    Computes the distance matrix (using manhattan distance) between two datasets (i.e., the distance between each pair of reconstructed and actual examples).
    """
    import numpy as np
    n = len(x_sol)
    Matrice_match = np.empty([n,n])
    for i in range(n):
        for j in range(n):
            Matrice_match[i][j] = dist_individus(x_sol[i],x_train[j], non_binary_attrs=non_binary_attrs)
    return Matrice_match

import numpy as np
from scipy.optimize import linear_sum_assignment
import random

def average_error(x_sol, x_train, seed, return_all_distances=False):
    """
    Computes the average reconstruction error between the proposed reconstruction x_sol and the actual training set x_train.
    If the dimensions of x_sol and x_train are the same, the function proceeds as usual.
    If x_train has more elements than x_sol, it randomly selects len(x_sol) elements from x_train and computes the error.
    If x_train has fewer elements than x_sol, it randomly selects len(x_sol)-len(x_train) elements from x_train with replacement
    to match the size of x_sol and then computes the error.
    """

    if isinstance(x_sol, dict):
        x_sol = np.array(list(x_sol.values()))
    elif not isinstance(x_sol, np.ndarray):
        x_sol = np.array(x_sol)

    if isinstance(x_train, dict):
        x_train = np.array(list(x_train.values()))
    elif not isinstance(x_train, np.ndarray):
        x_train = np.array(x_train)


    # Convert lists to numpy arrays
    x_sol_array = np.asarray(x_sol)
    x_train_array = np.asarray(x_train)
    
    random.seed(seed)
    
    # Case 1: x_sol and x_train have the same dimensions
    if x_sol_array.shape == x_train_array.shape:
        pass  # No changes needed, proceed as usual

    # Case 2: x_train has more elements than x_sol
    elif x_train_array.shape[0] > x_sol_array.shape[0]:
        random.seed(seed)
        additional_indices = random.choices(range(x_sol_array.shape[0]), k=x_train_array.shape[0] - x_sol_array.shape[0])
        x_sol_array = np.concatenate((x_sol_array, x_sol_array[additional_indices]), axis=0) # randomly oversample x_sol to match x_train

    # Case 3: x_train has fewer elements than x_sol
    else:
        random.seed(seed)
        selected_indices = random.sample(range(x_sol_array.shape[0]), x_train_array.shape[0]) 
        x_sol_array = x_sol_array[selected_indices] # randomly subsample x_sol to match x_train

    # Now both arrays have the same size, proceed with computing the error
    cost = matrice_matching(x_sol_array, x_train_array)
    row_ind, col_ind = linear_sum_assignment(cost)

    if return_all_distances:
        all_distances = []

    total_distance = 0
    for i, j in zip(row_ind, col_ind):
        a_dist = dist_individus(x_sol_array[i], x_train_array[j])
        if return_all_distances:
            all_distances.append(a_dist)
        total_distance += a_dist


    average_distance = total_distance / len(row_ind)

    if return_all_distances:
        return average_distance, col_ind.tolist(), all_distances
    else: 
        return average_distance, col_ind.tolist()


def generate_random_sols(N,M, dataset_ohe_groups=[], n_sols=10, seed=42):
    """
    Generates n_sols random reconstructions of shape (N,M) that conform with the one-hot encoding information provided through dataset_ohe_groups.
    """
    import numpy as np
    np.random.seed(seed)
    randlist = []
    for i in range(n_sols):
        temporary_random = np.random.randint(2,size = (N,M))
        for j in range(N):
            for w in dataset_ohe_groups:
                list_draw = [1] + [0]*(len(w) - 1) # exactly one zero
                drawn = np.random.choice(np.array(list_draw), len(list_draw), replace=False) # random order
                for drawn_index, w_index in enumerate(w):
                    temporary_random[j][w_index] = drawn[drawn_index]

        randlist.append(temporary_random.tolist())
    return randlist

def check_ohe(X, ohe_vectors, verbose = True):
    ''' 
    Debugging function: use to check whether the stated one-hot encoding is verified on a given dataset

    Arguments
    ---------
    X: np array of shape [n_examples, n_attriutes]
        The one-hot encoded dataset to be verified
    
    ohe_vectors: list, optional
        List of lists, where each sub-list contains the IDs of a group of attributes corresponding to a one-hot encoding of the same original feature

    verbose: boolean, optional (default True)
        If an example for which the encoding is not correct is found, whether to print it or not

    Returns
    -------
    output: boolean
        False if an example for which the provided one-hot encoding is not verified 
            (i.e., for some subgroup of binary features one-hot encoding the same original attribute, their sum is not 1)
        True otherwise
    '''
    for a_ohe_group in ohe_vectors:
        for an_example in range(X.shape[0]):
            check_sum = sum(X.iloc[an_example][a_ohe_group])
            if check_sum != 1:
                print("Found non-verified OHE: example %d, ohe group: " %(an_example), a_ohe_group)
                print("Example is: ", X.iloc[an_example], "with incorrect subset: ", X.iloc[an_example][a_ohe_group])
                return False
    return True