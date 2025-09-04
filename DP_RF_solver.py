from ortools.sat.python import cp_model
from utils import *
import time 
import ortools
import copy
from scipy import integrate as intg
import sklearn


class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, x_vars, M, N):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.start_time = time.time() # Recalls start time
        self.first_sol = True
        self.sol_list = []
        self.time_list = []
        self.x_var = [[ x_vars[k][i] for i in range(M)] for k in range(N)]
        self.M = M
        self.N = N

    def on_solution_callback(self):
        sol_time = time.time() - self.start_time
        if self.first_sol:
            self.first_sol = False
            self.time_to_first_sol = sol_time
        x_sol = [[self.Value(self.x_var[k][i]) for i in range(self.M)] for k in range(self.N)]
        self.sol_list.append(x_sol)
        self.time_list.append(sol_time)

class DP_RF_solver:
    def __init__(self, clf,eps):
        self.clf = clf
        self.estimators = clf.estimators_
        self.eps = eps
        self.eps_v = eps/clf.N_trees
        self.result_dict = {}

    def interval_N(self):
        nb = self.format_nb()
        N_avg = round(sum(nb[t][v][c] for t in range(self.clf.N_trees) for v in range(len(nb[0])) for c in range(self.clf.n_classes_)) / self.clf.N_trees)
        t = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697]
        N_leaves = 2 ** self.clf.estimators_[0].tree_.max_depth
        N_classes = self.clf.n_classes_
        std = np.sqrt(2.0 * N_leaves * N_classes) / self.eps
        bound_inf = int(N_avg - t[self.clf.N_trees - 2] * std)
        bound_sup = int(N_avg + t[self.clf.N_trees - 2] * std)
        return N_avg, bound_inf, bound_sup
    
    def log_liste_laplace(self):
        #bound = round(12 / self.eps_v)
        bound = round(np.ceil(12/self.eps_v)) # Rather use ceiling
        res = [0] * (bound + 1)
        res[0] = np.log(self.proba_laplace(-1, 1)[0])
        for i in range(1, bound + 1):
            res[i] = np.log(2 * self.proba_laplace(i, i + 1)[0])
        return res
    
    def proba_laplace(self, a, b):
        def f(x):
            return self.eps_v / 2 * np.exp(-self.eps_v * np.abs(x))
        return intg.quad(f, a, b)
 
    def format_nb(self):
        num_leaves = self.get_numeros_leaves(self.clf.estimators_[0].tree_)
        nb_b = [[[round(t.tree_.value[v][0][c]) for c in range(self.clf.n_classes_)] for v in num_leaves] for t in self.clf.estimators_]
        return nb_b

    def get_numeros_leaves(self, tree):
        leaves = []
        def browse_nodes(nodes):
            if tree.children_left[nodes] == tree.children_right[nodes]:
                leaves.append(nodes)
            else:
                if tree.children_left[nodes] != -1:
                    browse_nodes(tree.children_left[nodes])
                if tree.children_right[nodes] != -1:
                    browse_nodes(tree.children_right[nodes])
        browse_nodes(0)
        return leaves
    
    def parse_forest(self, clf, verbosity=False):
        T = clf.estimators_

        def retrieve_branches(number_nodes, children_left_list, children_right_list, nodes_features_list, nodes_value_list):
            is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
            paths = []
            for i in range(number_nodes):
                if is_leaves_list[i]:
                    end_node = [path[-1] for path in paths]
                    if i in end_node:
                        output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                        output = output[:-1]
                        yield (output, list(nodes_value_list[i][0]))
                else:
                    origin, end_l, end_r = i, children_left_list[i], children_right_list[i]
                    for index, path in enumerate(paths):
                        if origin == path[-1]:
                            path[-1] = -nodes_features_list[origin]
                            paths[index] = path + [end_l]
                            path[-1] = nodes_features_list[origin]
                            paths.append(path + [end_r])
                    if i == 0:
                        paths.append([-nodes_features_list[i], children_left_list[i]])
                        paths.append([nodes_features_list[i], children_right_list[i]])

        trees_branches = []
        for tree in T:
            t = tree.tree_
            sklearn_version = str(sklearn.__version__).split(".")
            if int(sklearn_version[0]) <= 1 and int(sklearn_version[1]) <= 3:
                nodes_value = t.value
            else:
                total_examples = t.weighted_n_node_samples
                nodes_value = copy.deepcopy(t.value)
                for i in range(len(nodes_value)):
                    for j in range(len(nodes_value[i][0])):
                        nodes_value[i][0][j] = np.round(nodes_value[i][0][j] * total_examples[i], decimals=0)
                    assert(sum(nodes_value[i][0]) == total_examples[i])
            n_nodes = t.node_count
            children_left = t.children_left
            children_right = t.children_right
            nodes_features = copy.deepcopy(t.feature)
            nodes_features += 1
            all_branches = list(retrieve_branches(n_nodes, children_left, children_right, nodes_features, nodes_value))
            trees_branches.append(all_branches)
        if verbosity:
            print("Parsing done")
        return trees_branches
    
    def fit(self, N_fixed, seed, time_out, n_threads, verbosity, obj_active, X_known = None, known_attributes=[]):
        start = time.time()
        
        model = cp_model.CpModel()
        
        if verbosity:
            print("Constructing CP model.")

        # Model creation
        nb_noise = self.format_nb()
        card_c = self.clf.n_classes_
        N_trees = self.clf.n_estimators
        N_leaves = 2**self.clf.estimators_[0].tree_.max_depth
        #bound = round(12/self.eps_v)
        bound = round(np.ceil(12/self.eps_v)) # Rather use ceiling
        p = self.log_liste_laplace()
        one_hot_encoded_groups = self.clf.ohe_groups
        M = self.clf.estimators_[0].n_features_in_
        trees_branches = self.parse_forest(self.clf, verbosity=verbosity)
        N_min = 0
        N_max = 0
        
        if N_fixed is None:
            N_avg, N_min, N_max = self.interval_N() #self.clf, self.eps_v, N_trees
            N_min = max([1, N_min]) # can't be negative
            if verbosity:
                print('N_avg', N_avg,'N_max :', N_max, "N_min :", N_min)

            # Variables definitions
            N = model.NewIntVar(N_min, N_max, "N")
            nb = [[[model.NewIntVar(0, N_max, 'nb_%d_%d_%d' % (t, v, c)) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] 
            delta = [[[model.NewIntVar(-bound, bound, 'delta_%d_%d_%d' % (t, v, c)) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] 

            liste_p = []
            liste_bool = []
            
            x = [[model.NewBoolVar('x_%d_%d' % (k,j)) for j in range(M)] for k in range(N_max)]
            y = [[[[model.NewBoolVar('y_%d_%d_%d_%d' % (t,v,k,c)) for c in range(card_c)] for k in range(N_max)] for v in range(N_leaves)] for t in range(N_trees)]
            z = [[model.NewBoolVar('Z_%d_%d' % (i, c)) for c in range(card_c)] for i in range(N_max)]
            
            delta_bool_list = [[[[model.NewBoolVar(f'delta_val_{i}_{t}_{v}_{c}') for i in range(0, bound+1)] for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            abs_delta = [[[model.NewIntVar(0, bound, 'delta_%d_%d_%d' % (t, v, c)) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] 
            
            for t in range(N_trees):
                #Constraint ensuring that all trees have N training examples
                model.Add(sum(nb[t][v][c] for v in range(N_leaves) for c in range(card_c)) == N)
                
                for c in range(card_c):
                    for v in range(N_leaves):
                        # Constraint that computes the discrepancies between the noised value and the estimated count values
                        model.Add(delta[t][v][c] == nb_noise[t][v][c]-nb[t][v][c])
                        model.AddAbsEquality(abs_delta[t][v][c], delta[t][v][c])
                        
                        # Constraint defining bool_proba and delta_val as a function of delta[t][v][c]
                        delta_bool_constraint = []
                        ortools_version = str(ortools.__version__).split(".")
                        if int(ortools_version[0]) <= 9 and int(ortools_version[1]) <= 8:
                            model.AddMapDomain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset = 0)  
                        else:
                            model.add_map_domain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset = 0)
                            
                        delta_bool_constraint.extend(delta_bool_list[t][v][c])
                        liste_p.extend(p)         
                        liste_bool.extend(delta_bool_constraint)
                        
                        # Constraint ensuring that nb[t][v][c] is positif or null
                        model.Add(nb[t][v][c] >= 0)  

            
            #Each example is assigned to only one class 
            for k in range(N_max):
                model.Add(sum(z[k][c] for c in range(card_c)) == 1)
                
            #An example appears only in the counts of its class
            for k in range(N_max):
                for c in range(card_c):
                    model.Add(sum(y[t][v][k][c] for t in range(N_trees) for v in range(N_leaves)) == 0).OnlyEnforceIf(z[k][c].Not())

                    
            #The values of the features align with the splits of the branch
            ex_k_not_classified_by_leaf_v_in_tree_t = [[[model.NewBoolVar(f'ex_k_not_classified_by_leaf_v_in_tree_t{t}_{v}_{k}_{c}') for k in range(N_max)] for v in range(N_leaves)] for t in range(N_trees)]
            for idx_tree, liste_branches in enumerate(trees_branches):
                #Reverse the direction of liste_branches due to the diffprivlib numbering being reversed
                liste_branches = liste_branches[::-1]   
                for idx_branch, branche in enumerate(liste_branches):
                    #print("idx_branch :", idx_branch, "branche :", branche)
                    for k in range(N_max):
                        for feature in branche[0]:
                            model.Add(cp_model.LinearExpr.Sum(y[idx_tree][idx_branch][k]) == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k])
                            if feature > 0:
                                model.Add(x[k][abs(feature)-1] == 1).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())
                            if feature < 0:
                                model.Add(x[k][abs(feature)-1] == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())
            
            
            #Excess examples must be removed from the final reconstruction
            N_inf_k = [model.NewBoolVar(f'N_inf_{k}') for k in range(0, N_max)]
            for k in range(N_max):
                
                model.Add( N < k).OnlyEnforceIf(N_inf_k[k])
                model.Add(N >= k).OnlyEnforceIf(N_inf_k[k].Not())
                model.Add(sum(y[t][v][k][c] for t in range(N_trees) for v in range(N_leaves) for c in range(card_c)) == 0).OnlyEnforceIf(N_inf_k[k-N_min])
            
            
            #The counts correspond to the number of assigned examples
            for t in range(N_trees):
                for v in range(N_leaves):
                    for c in range(card_c):
                        model.Add(sum(y[t][v][k][c] for k in range(N_max)) == nb[t][v][c])
            
                                    
            #OHE constraint
            for k in range(N_max):
                for w in range(len(one_hot_encoded_groups)): # for each group of binary attributes one-hot encoding the same attribute
                    model.Add(cp_model.LinearExpr.Sum([x[k][i] for i in one_hot_encoded_groups[w]]) == 1)
            
                
                
        else:
            if verbosity:
                print("For N_fixed.")

            N = N_fixed
                
            # Variables definitions
            nb = [[[model.NewIntVar(0, N, 'nb_%d_%d_%d' % (t, v, c)) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] 
            delta = [[[model.NewIntVar(-bound, bound, 'delta_%d_%d_%d' % (t, v, c)) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] 
            
            liste_p = []
            liste_bool = []
            
            x = [[model.NewBoolVar('x_%d_%d' % (k,j)) for j in range(M)] for k in range(N)]
            y = [[[[model.NewBoolVar('y_%d_%d_%d_%d' % (t,v,k,c)) for c in range(card_c)] for k in range(N)] for v in range(N_leaves)] for t in range(N_trees)]
            z = [[model.NewBoolVar('Z_%d_%d' % (i, c)) for c in range(card_c)] for i in range(N)]
            
            # Assume knowledge of dataset
            if not(X_known is None):
                assert(X_known.shape[1] >= len(known_attributes))
                assert(X_known.shape[1] == M) # not mandatory actually but that's what I do in the experiments
                assert(X_known.shape[0] == N)

                for i in known_attributes:
                    for k in range(N):
                        model.Add( x[k][i] == X_known[k][i] )
            # ----------------------------------------------------------------------------------------------


            delta_bool_list = [[[[model.NewBoolVar(f'delta_val_{i}_{t}_{v}_{c}') for i in range(0, bound+1)] for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            abs_delta = [[[model.NewIntVar(0, bound, 'delta_%d_%d_%d' % (t, v, c)) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] 
                
            if verbosity:
                print("Created variables.")

            for t in range(N_trees):
                #Constraint ensuring that all trees have N training examples
                model.Add(sum(nb[t][v][c] for v in range(N_leaves) for c in range(card_c)) == N)
                
                for c in range(card_c):
                    for v in range(N_leaves):
                        # Constraint that computes the discrepancies between the noised value and the estimated count values
                        model.Add(delta[t][v][c] == nb_noise[t][v][c]-nb[t][v][c])
                        model.AddAbsEquality(abs_delta[t][v][c], delta[t][v][c])
                        
                        # Constraint defining bool_proba and delta_val as a function of delta[t][v][c]
                        delta_bool_constraint = []
                        ortools_version = str(ortools.__version__).split(".")
                        if int(ortools_version[0]) <= 9 and int(ortools_version[1]) <= 8:
                            model.AddMapDomain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset = 0)  
                        else:
                            model.add_map_domain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset = 0)
                            
                        delta_bool_constraint.extend(delta_bool_list[t][v][c])
                        liste_p.extend(p)         
                        liste_bool.extend(delta_bool_constraint)
                        
                        # Constraint ensuring that nb[t][v][c] is positif or null
                        model.Add(nb[t][v][c] >= 0)  

            if verbosity:
                print("Created tree constraints.")

            # Each example is assigned to only one class 
            for k in range(N):
                model.Add(sum(z[k][c] for c in range(card_c)) == 1)
                
            # An example appears only in the counts of its class
            #for k in range(N):
            #    for c in range(card_c):
            #        model.Add(sum(y[t][v][k][c] for t in range(N_trees) for v in range(N_leaves)) == 0).OnlyEnforceIf(z[k][c].Not())
            # Alternative, stronger formulation:
            for t in range(N_trees):
                for k in range(N):
                    for c in range(card_c):
                        model.Add(sum(y[t][v][k][c] for v in range(N_leaves)) == z[k][c])
                    
            if verbosity:
                print("Created other constraints.")


            #The values of the features align with the splits of the branch
            ex_k_not_classified_by_leaf_v_in_tree_t = [[[model.NewBoolVar(f'ex_k_not_classified_by_leaf_v_in_tree_t{t}_{v}_{k}_{c}') for k in range(N)] for v in range(N_leaves)] for t in range(N_trees)]
            for idx_tree, liste_branches in enumerate(trees_branches):
                #Reverse the direction of liste_branches due to the diffprivlib numbering being reversed
                liste_branches = liste_branches[::-1]   
                for idx_branch, branche in enumerate(liste_branches):
                    #print("idx_branch :", idx_branch, "branche :", branche)
                    for k in range(N):
                        for feature in branche[0]:
                            model.Add(cp_model.LinearExpr.Sum(y[idx_tree][idx_branch][k]) == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k])
                            if feature > 0:
                                model.Add(x[k][abs(feature)-1] == 1).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())
                            if feature < 0:
                                model.Add(x[k][abs(feature)-1] == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())
            
            if verbosity:
                print("Created other constraints bis.")

            #The counts correspond to the number of assigned examples
            for t in range(N_trees):
                for v in range(N_leaves):
                    for c in range(card_c):
                        model.Add(sum(y[t][v][k][c] for k in range(N)) == nb[t][v][c])
            
            if verbosity:
                print("Created other constraints ter.")
                        
            #OHE Constraint
            for k in range(N):
                for w in range(len(one_hot_encoded_groups)): # for each group of binary attributes one-hot encoding the same attribute
                    model.Add(cp_model.LinearExpr.Sum([x[k][i] for i in one_hot_encoded_groups[w]]) == 1)
        
        if verbosity:
            print("Beginning search.")

        # Solver creation
        solver = cp_model.CpSolver()

        solver.parameters.log_search_progress = verbosity
        solver.parameters.max_time_in_seconds = time_out
        solver.parameters.num_workers = n_threads
        solver.parameters.random_seed = seed

        if obj_active:
            model.Maximize(cp_model.LinearExpr.WeightedSum(liste_bool, liste_p))

        if N_fixed is None:   
            solver.Solve(model)
        else:
            # Create the callback used to log time to first solution
            solcallback = MySolutionCallback(x, M)

            # Solving the problem
            #solver.Solve(model)
            solver.SolveWithSolutionCallback(model, solcallback)
            solver.ResponseStats()
        
        end = time.time()
        duration = end - start

        # Printing results
        if solver.StatusName() == 'OPTIMAL' or solver.StatusName() == 'FEASIBLE':
            if verbosity:
                print('Value of the maximized objective function:', solver.ObjectiveValue())
            
            N = solver.Value(N)
            if verbosity:
                print("N :", N)
            
            x = [[solver.Value(x[k][i]) for i in range(M)] for k in range(N)]
            #if verbosity:
            #    print("x_sol :", x)
            
            y = [[[solver.Value(y[t][v][k][c]) for c in range(card_c)] for k in range(N)] for v in range(N_leaves) for t in range(N_trees)]
            #print("y :", y)
            
            z = [[solver.Value(z[k][c]) for c in range(card_c)] for k in range(N)]
            #print("z :", z)
                    
            ex_k_not_classified_by_leaf_v_in_tree_t = [[[solver.Value(ex_k_not_classified_by_leaf_v_in_tree_t[t][v][k]) for k in range(N)] for v in range(N_leaves)] for t in range(N_trees)]
            #print("ex_k_not_classified_by_leaf_v_in_tree_t :", ex_k_not_classified_by_leaf_v_in_tree_t)
                
            #print("delta:", [[[solver.Value(delta[t][v][c]) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)])
            #print("abs_delta:", [[[solver.Value(abs_delta[t][v][c]) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)])
            #print("delta_bool_list :", [[[solver.Value(delta_bool_list[t][v][c][i]) for i in range(0, max(4,round(12/eps))+1)] for c in range(card_c)] for v in range(N_leaves) for t in range(N_trees)], "de taille", len(delta_bool_list[0][0]*len(delta_bool_list[0][0][0])))
            #print("liste_bool:", liste_bool, "de taille", len(liste_bool))
            #print("liste_p:", liste_p, "de taille", len(liste_p))
            
            values_nb = [[[solver.Value(nb[t][v][c]) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            
            #if verbosity:
            #    print("Bruite :", nb_noise)
            #    print("Recons :", values_nb)
            if N_fixed is None:
                self.result_dict = {'status':solver.StatusName(), 'nb_recons': values_nb, 'duration': duration, 'reconstructed_data':x, 'N_min': N_min, 'N_max': N_max, 'N' : N}
            else:
                self.result_dict = {'status':solver.StatusName(), 'nb_recons': values_nb, 'duration': duration, 'reconstructed_data':x, 'N_min': N_min, 'N_max': N_max, 'N' : N, 'time_to_first_solution' : solcallback.time_to_first_sol, 'anytime_sols' : solcallback.sol_list, 'anytime_sols_times' : solcallback.time_list}

        else :
            self.result_dict = {'status':solver.StatusName(), 'duration': duration}
        
        return self.result_dict
