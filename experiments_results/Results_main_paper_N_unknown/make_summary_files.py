import json
import os 

list_N_samples = [100]
list_N_trees = [1,5,10,20,30]
list_epsilon = [30,20,10,5,1,0.1]
list_obj_active = [1]
list_depth = [3,5,7]
list_seed = [0,1,2,3,4]
list_datasets = ['compas','default_credit', 'adult']
N_fixed = None

list_config = []
retrieved = 0
res = dict()
for dataset in list_datasets:
    res[dataset] = []

expe_id = 0
total_expes = 0
total_missing_ids = []
total_missing_skipped = []
for obj_active_bool in list_obj_active:
    for depth in list_depth:
        for N_trees in list_N_trees:
            for epsilon in list_epsilon: 
                for Nsamp in list_N_samples:
                    for dataset in list_datasets:
                        for seed in list_seed:
                            total_expes += 1
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

                                        if not(new_result[0]["solver_status"] in ["OPTIMAL", "FEASIBLE"]):
                                            good = False
                                            print("Bad status: ", new_result[0]["solver_status"], "(id=", expe_id, ")")

                                except json.decoder.JSONDecodeError:
                                    good = False
                            if not good:
                                
                                if epsilon == 0.1 and N_trees == 30 and depth == 7: # already not solved even when N is known
                                    total_missing_skipped.append(expe_id)
                                else:
                                    total_missing_ids.append(expe_id)
                                    print("Missing: ", results_file)
                            expe_id += 1

res_path = "N_fixed" if N_fixed is not None else "N_free"
for dataset in list_datasets:
    results_file = f'{res_path}_{dataset}_results.json'
    with open(results_file, 'w') as f:
        json.dump(res[dataset], f, indent=4)
        print("Successfully saved %s (%d results)" %(results_file, len(res[dataset])))

print("Retrieved %d files, %d/%d experiments completed successfully." %(retrieved, total_expes-len(total_missing_ids), total_expes))

print("Skipped (not counted in missing): ", total_missing_skipped)

if len(total_missing_ids) > 0:
    liste = str(sorted(total_missing_ids)).replace(" ", "")
    print("To relaunch: ", liste, "Total=", len(total_missing_ids))