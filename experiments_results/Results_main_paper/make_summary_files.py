import json
import os 

list_N_samples = [100]
list_N_trees = [1,5,10,20,30]
list_epsilon = [30,20,10,5,1,0.1]
list_obj_active = [1]
list_depth = [3,5,7]
list_seed = [0,1,2,3,4]
list_datasets = ['compas','default_credit', 'adult']
N_fixed = True

list_config = []
retrieved = 0
res = dict()
for dataset in list_datasets:
    res[dataset] = []

expe_id = 0
total_missing_ids = []
for obj_active_bool in list_obj_active:
    for depth in list_depth:
        for N_trees in list_N_trees:
            for epsilon in list_epsilon: 
                for Nsamp in list_N_samples:
                    for dataset in list_datasets:
                        for seed in list_seed:
                            res_path = "N_fixed" if N_fixed is not None else "N_free"
                            res_path += "%d_%.2f_%d_%d" %(N_trees, epsilon, seed, depth)
                            results_file = f'./{res_path}_{dataset}_results.json'
                            good = False
                            if os.path.exists(results_file):
                                try:
                                    good = True
                                    with open(results_file, 'r') as f:
                                        new_result = json.load(f)
                                        assert(len(new_result) == 1)
                                        res[dataset].append(new_result[0])
                                        #if (new_result[0]["id"]) == 68 or (new_result[0]["id"]) == 83:
                                        #    print(new_result)
                                        retrieved += 1
                                except json.decoder.JSONDecodeError:
                                    good = False
                            if not good:
                                print("Missing: ", results_file)
                                total_missing_ids.append(expe_id)
                            expe_id += 1

res_path = "N_fixed" if N_fixed is not None else "N_free"
for dataset in list_datasets:
    results_file = f'{res_path}_{dataset}_results.json'
    with open(results_file, 'w') as f:
        json.dump(res[dataset], f, indent=4)
        print("Successfully saved %s" %results_file)
