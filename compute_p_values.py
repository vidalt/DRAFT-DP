import json
import numpy as np
from decimal import Decimal

datasets = ["adult"] #["adult", "compas", "default_credit"]
depth_values = [5] #[3, 5, 7]
epsilon_values = [0.1,1,30] #[0.1, 1, 5, 10, 20, 30]
tree_values = [10] #[1, 5, 10, 20, 30]
seeds_list = [0,1,2,3,4] #[0, 1, 2, 3, 4]

# Random baseline
random_reconstruction_error = {"compas": 0.2081, "adult": 0.2539, "default_credit": 0.2642}

table_data = {dataset: {n_trees: {epsilon: {depth: "-" for depth in depth_values} for epsilon in epsilon_values} for n_trees in tree_values} for dataset in datasets}

for dataset in datasets:
    with open(f'experiments_results/Results_main_paper/N_fixed_{dataset}_results.json', 'r') as f:
        results = json.load(f)
    
    for depth in depth_values:
        for n_trees in tree_values:
            for epsilon in epsilon_values:
                error = []
                distrib_errors = []
                average_cdf_val = 0
                for seed in seeds_list:
                    for result in results:
                        if (result['depth'] == depth and result['epsilon'] == epsilon 
                                and result['N_trees'] == n_trees and result['solver_status'] != 'UNKNOWN') and result['seed'] == seed:
                            error.append(result['reconstruction_error'])
                            distrib_errors.append(result['reconstruction_error_matching_distrib_list'])

                            random_errs_sorted = sorted(result['reconstruction_error_matching_distrib_list'])
                            actual_error = result['reconstruction_error']
                            print("Plots for expe: dataset %s, %d trees of depth %d, epsilon=%.2f, seed %d" %(dataset, n_trees, depth, epsilon, seed))
                            import matplotlib.pyplot as plt 

                            #plt.plot([i for i in range(len(random_errs_sorted))], random_errs_sorted)
                            #plt.plot(len(random_errs_sorted)/2, actual_error, marker='x')

                            prop = sum(i > actual_error for i in random_errs_sorted)
                            print("Prop = %.3f" %prop, "(actual error %.3f)" %actual_error)
                            #plt.title("Prop: %.3f" %prop)
                            #plt.show()

                            import scipy.stats as stats
                            assert(len(random_errs_sorted) == 100) # just double-checking

                            df_mean = np.mean(random_errs_sorted)
                            df_std = np.std(random_errs_sorted)


                            all_errors = [actual_error] + random_errs_sorted 

                            # Just to make the plot look better
                            smallest_val = min([actual_error, min(random_errs_sorted), df_mean-5*df_std])
                            largest_val = max([abs(df_mean-actual_error), abs(df_mean-min(random_errs_sorted)), max(random_errs_sorted), df_mean+5*df_std])
                            all_errors.extend(np.linspace(start=smallest_val, stop=largest_val, num=500))
                            # ---------------------------------

                            all_errors = sorted(all_errors)
                            actual_error_index = all_errors.index(actual_error)
                            pdf = stats.norm.pdf(all_errors, df_mean, df_std)

                            cdf = stats.norm.cdf(all_errors, df_mean, df_std)
                            
                            actual_cdf_val = cdf[actual_error_index]
                            if actual_cdf_val < 0.001:
                                val = "RF training set\n CDF = %.2E" %(Decimal(actual_cdf_val))
                            else:
                                val = "RF training set\n CDF = %.3f" %actual_cdf_val

                            plt.rcParams["figure.figsize"] = (9,6)
                            plt.rcParams.update({'font.size': 14})
                            fig, ax1 = plt.subplots()
                            colorCDF = '#FF4242'
                            colorPDF = '#235FA4'
                            colorHIST = '#6FDE6E'

                            ax1.set_ylabel('Reconstruction error PDF', color=colorPDF) #for datasets\n randomly drawn from the data distribution
                            ax1.tick_params(axis='y', labelcolor=colorPDF)
                            ax1.set_xlabel('Reconstruction error')

                            ax1.plot(all_errors, pdf, color=colorPDF)
                            ax1.fill_between(all_errors[:actual_error_index+1], pdf[:actual_error_index+1], alpha=0.5, color=colorCDF)

                            if actual_cdf_val < 0.01:
                                plt.annotate(val, (actual_error-0.002, pdf[actual_error_index]+3), color=colorCDF, rotation=0)
                            else:
                                plt.annotate(val, (actual_error-0.014, pdf[actual_error_index]-1), color=colorCDF, rotation=0) # add more neg shift on x to stay far from the curve
                            ax1.plot(actual_error, pdf[actual_error_index], marker='x', markeredgewidth=3, markersize=10, color=colorCDF)

                            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
                            ax2.hist(random_errs_sorted, density=False, bins='auto', histtype='stepfilled', alpha=0.4, color=colorHIST)
                            ax2.tick_params(axis='y', labelcolor=colorHIST)
                            ax2.set_ylabel('Datasets randomly drawn from data distribution', color=colorHIST) #for datasets\n randomly drawn from the data distribution
          

                            ax1.set_ylim(0)
                            #fig.tight_layout()
                            #plt.show()
                            plt.savefig('figures/plot-df-%s-epsilon%.2f-%dtrees-depth%d-seed%d.pdf' %(dataset, epsilon, n_trees, depth, seed), bbox_inches='tight')
                            '''plt.plot(all_errors, cdf, color="green")
                            plt.scatter(actual_error, cdf[actual_error_index], color='red')
                            
                            plt.annotate(val, (actual_error+0.002, actual_cdf_val+0.02))
                            plt.show()'''
                            average_cdf_val+=actual_cdf_val
                average_cdf_val = average_cdf_val/5
                print("average cdf = %.9f" %average_cdf_val)