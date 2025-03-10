import numpy as np
import matplotlib 

import matplotlib.pyplot as plt

from scipy import integrate as intg

def proba_laplace(a, b, eps_etape):
        def f(x):
            return eps_etape / 2 * np.exp(-eps_etape * np.abs(x))
        return intg.quad(f, a, b)

list_epsi = np.arange(0.1, 30, 0.01)
list_proba_lim = []


for epsi_etape in list_epsi:
    cpt = 0
    somme_proba = 0
    while somme_proba < 0.999:
        somme_proba = proba_laplace(-(cpt+1), cpt+1, epsi_etape)[0]
        cpt+=1
    list_proba_lim.append(cpt-1)

liste_intervalle = [np.ceil(12/epsi_etape) for epsi_etape in list_epsi]
plt.figure(figsize=(8, 5))
plt.plot(list_epsi, list_proba_lim, alpha=0.8, label = r'Theoretical value of $\delta$ such that $\mathbb{P}(\text{int}(Y_{tvc}) \in \{ -\delta,\dots, \delta \} ) \geq 0.999$', )
plt.plot(list_epsi, liste_intervalle,"--",label=r'$\lceil 12 / \varepsilon_v \rceil$', alpha=0.8)
plt.xlabel(r"$\varepsilon_v$")
plt.ylabel(r"Width of $\Delta_{tvc}$ search interval")
#plt.title("Width of the delta search interval")
plt.legend()
plt.savefig("figures/epsilon_delta.pdf", bbox_inches='tight')
plt.clf()


from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier
import matplotlib 
import re
import pandas as pd
import numpy as np
import copy

def fill_forest(clf, X, y):
    X = pd.DataFrame(X) 
    y = pd.Series(y)  

    def split_feature(X, y, feature, threshold):
        mask = X.iloc[:, feature] <= threshold
        return X[mask], X[~mask], y[mask], y[~mask]

    def fill_tree(tree, X, y, node_id):
        if tree.feature[node_id] == -2:  # Node is a leaf
            return

        feature, threshold = tree.feature[node_id], tree.threshold[node_id]
        left_X, right_X, left_y, right_y = split_feature(X, y, feature, threshold)

        left_child, right_child = tree.children_left[node_id], tree.children_right[node_id]

        for c in range(len(tree.value[0][0])):
            if left_child != -1:
                tree.value[left_child][0][c] = (left_y == c).sum()
            if right_child != -1:
                tree.value[right_child][0][c] = (right_y == c).sum()

        fill_tree(tree, left_X, left_y, left_child)
        fill_tree(tree, right_X, right_y, right_child)

    for tree in clf.estimators_:
        for c in range(len(tree.tree_.value[0][0])):
            tree.tree_.value[0][0][c] = np.sum(y == c)  
        fill_tree(tree.tree_, X, y, 0)

    return clf
    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import integrate as intg
import copy
from utils import *
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier


def format_nb(clf):
    num_leaves = get_num_leaves(clf.estimators_[0].tree_) #-1 because diffprivlib numerotation is reversed
    nb_b = [[[round(t.tree_.value[v][0][c]) for c in range(clf.n_classes_)] for v in num_leaves] for t in clf.estimators_]
    
    return nb_b

def get_num_leaves(tree):
    leaves = []  
    def parcourir_noeud(noeud):
        if tree.children_left[noeud] == tree.children_right[noeud]:
            leaves.append(noeud)
        else:
            if tree.children_left[noeud] != -1:
                parcourir_noeud(tree.children_left[noeud])
            if tree.children_right[noeud] != -1:
                parcourir_noeud(tree.children_right[noeud])

    parcourir_noeud(0)
    return leaves

def log_liste_laplace(eps):
    borne = max(4, round(12/eps))
    res = [0]*(borne+1)
    res[0] = np.log(proba_laplace(-1, 1, eps)[0])
    
    for i in range(1,borne+1):
        res[i] = np.log(2*proba_laplace(i, i+1, eps)[0])
    return res

def proba_laplace(a,b, eps):
    def f(x):
        return eps/2*np.exp(-eps*np.abs(x))   
    return intg.quad(f,a,b) 

def create_laplace_tree(path, N_samples, N_trees, eps, seed):
    data = pd.read_csv(path)
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = X.sample(N_samples, random_state=seed)
    y = y.loc[X.index]
        
    nb_features = X.shape[1]
    maximum_depth = nb_features//2

    #print("Nb features :", nb_features)
    #print("Maximum depth :", maximum_depth)
    
    clf = DP_RandomForestClassifier(n_estimators=N_trees, max_depth=maximum_depth, bounds=(0,1), classes = [0,1])
    clf.fit(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    def fill_forest(clf, X, y):
        def split_feature(X, y, feature, threshold):
            mask = X.iloc[:, feature] <= threshold
            return X[mask], X[~mask], y[mask], y[~mask]

        def fill_tree(tree, X, y, node_id):
            if tree.feature[node_id] == -2:  # Node is a leaf
                return
            
            feature, threshold = tree.feature[node_id], tree.threshold[node_id]
            #print(f"feature: {feature}\nthreshold: {threshold}")
            
            left_X, right_X, left_y, right_y = split_feature(X, y, feature, threshold)
            
            left_child, right_child = tree.children_left[node_id], tree.children_right[node_id]
            
            for c in range(len(tree.value[0][0])):
                if left_child != -1:
                    tree.value[left_child][0][c] = (left_y == c).sum()
                if right_child != -1:
                    tree.value[right_child][0][c] = (right_y == c).sum()
                
            fill_tree(tree, left_X, left_y, left_child)
            fill_tree(tree, right_X, right_y, right_child)

        for tree in clf.estimators_:
            for c in range(len(tree.tree_.value[0][0])):
                tree.tree_.value[0][0][c] = len(y[y == c])
            fill_tree(tree.tree_, X, y, 0)
            
        return clf

    clf = fill_forest(clf, X_train, y_train)
    clf_noised = copy.deepcopy(clf)
    eps_etape = eps/N_trees

    N_leaves = 2**clf.estimators_[0].tree_.max_depth
    card_c = clf.n_classes_
    np.random.seed(seed)
    delta = [np.random.laplace(scale=1.0/eps_etape) for i in range(N_trees*N_leaves*card_c)] #Il faut une liste de delta sinon on a toujours la même valeur à cause de la seed
    
    print("delta :", delta)
    
    # Adding Laplace noise using the Laplace mechanism
    for tree in clf_noised.estimators_:
        # Loop over the leaves
        for i in range(len(tree.tree_.value)):
            if tree.tree_.children_left[i] == -1:
                for j in range(len(tree.tree_.value[i][0])):
                    #print("delta[-1] :", delta[-1])
                    tree.tree_.value[i][0][j] = tree.tree_.value[i][0][j]+int(delta[-1])
                    
                    #tree.tree_.value[i][0][j] = max(0,tree.tree_.value[i][0][j]+int(delta[-1]))
                    delta.pop()
            else :
                for j in range(len(tree.tree_.value[i][0])):
                    tree.tree_.value[i][0][j] = 0
        
    #accuracy = accuracy_score(y_test, clf_noised.predict(X_test))

    return clf_noised, clf, eps_etape, X_train, X_test, y_train, y_test

def intervalle_N(clf, eps_etape, N_trees):
    nb = format_nb(clf)
    N_avg = round(sum(nb[t][v][c] for t in range(N_trees) for v in range(len(nb[0])) for c in range(clf.n_classes_))/N_trees)

    t = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697]
    N_leaves = 2**clf.estimators_[0].tree_.max_depth
    N_classes = clf.n_classes_
    std = np.sqrt(2.0*N_leaves*N_classes)/eps_etape

    N_max = round(N_avg + t[N_trees-1]*std/np.sqrt(N_trees))
    N_min = max(1, round(N_avg - t[N_trees]*std/np.sqrt(N_trees)))

    return N_avg, N_min, N_max


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datasets_infos import random_predictions_accuracy
from matplotlib.lines import Line2D

# Define parameters N and datasets
for N in ['N_fixed']:
    for dataset in ["compas", "adult", "default_credit"]:
        
        # Initialize lists for data
        epsilons = []
        N_trees = []
        accuracies = []
        
        rd_acc = random_predictions_accuracy[dataset]
        
        # Load results from JSON file
        with open(f'experiments_results/{N}_{dataset}_results.json', 'r') as f:
            results = json.load(f)
            
            # Iterate through results to extract epsilon, N_trees, and accuracy_train
            for result in results:
                if result['depth'] == 7:
                    epsilons.append(result['epsilon'])
                    N_trees.append(result['N_trees'])
                    accuracies.append(result['accuracy_train'])
        
        # Create a DataFrame with filtered data
        df = pd.DataFrame({
            'epsilon': epsilons,
            'N_trees': N_trees,
            'accuracy_train': accuracies
        })
        
        # Compute the mean of accuracy_train for each combination of epsilon and N_trees
        pivot_df = df.pivot_table(index='N_trees', columns='epsilon', values='accuracy_train', aggfunc='mean')
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar=True)        
        plt.xlabel(r'$\varepsilon$')
        plt.ylabel(r'#trees')
        
        # Create a custom legend with Line2D
        legend_line = Line2D([0], [0], color='black', linestyle='-', linewidth=2)
        plt.legend([legend_line], [f'Majority Classifier Accuracy: {rd_acc:.2f}'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='large', frameon=False)
        
        plt.savefig(f'figures/{N}_{dataset}_heatmap_depth7.pdf', bbox_inches='tight')
        plt.clf()


from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib 
import re
import pandas as pd
import numpy as np
import copy

def fill_forest(clf, X, y):
    X = pd.DataFrame(X)  
    y = pd.Series(y)  

    def split_feature(X, y, feature, threshold):
        mask = X.iloc[:, feature] <= threshold
        return X[mask], X[~mask], y[mask], y[~mask]

    def fill_tree(tree, X, y, node_id):
        if tree.feature[node_id] == -2:  # Node is a leaf
            return

        feature, threshold = tree.feature[node_id], tree.threshold[node_id]
        left_X, right_X, left_y, right_y = split_feature(X, y, feature, threshold)

        left_child, right_child = tree.children_left[node_id], tree.children_right[node_id]

        for c in range(len(tree.value[0][0])):
            if left_child != -1:
                tree.value[left_child][0][c] = (left_y == c).sum()
            if right_child != -1:
                tree.value[right_child][0][c] = (right_y == c).sum()

        fill_tree(tree, left_X, left_y, left_child)
        fill_tree(tree, right_X, right_y, right_child)

    for tree in clf.estimators_:
        for c in range(len(tree.tree_.value[0][0])):
            tree.tree_.value[0][0][c] = np.sum(y == c)  
        fill_tree(tree.tree_, X, y, 0)

    return clf
    

from sklearn.tree import DecisionTreeClassifier
import matplotlib 
import re

for removed_feat in [0,1,2,3]:
    plot_extension = 'pdf'

    X = [[0,0,0,1],
        [1,0,0,0],
        [0,1,0,0],
        [1,0,1,1]]
    y = [0, 0, 1, 1]
    features = ["$f_1$", "$f_2$", "$f_3$", "$f_4$"] #, "$X_3$","$X_4$"

    for an_ex in X:
        del an_ex[removed_feat]
    del features[removed_feat]

    clf = DP_RandomForestClassifier(n_estimators =1, max_depth=2)
    clf.fit(X, y) 
    
    clf = fill_forest(clf, X, y)
    
    for tree in clf.estimators_:
        for i in range(len(tree.tree_.value)):
            if tree.tree_.children_left[i] != -1:
                for j in range(len(tree.tree_.value[i][0])):
                    tree.tree_.value[i][0][j] = 0
    
    clf_bruit = copy.deepcopy(clf)
    
    eps = 1.0
    
    for tree in clf_bruit.estimators_:
        for i in range(len(tree.tree_.value)):
            if tree.tree_.children_left[i] == -1:
                for j in range(len(tree.tree_.value[i][0])):
                    delta = np.random.laplace(scale=1.0/eps)
                    tree.tree_.value[i][0][j] = tree.tree_.value[i][0][j]+int(delta)
            else :
                for j in range(len(tree.tree_.value[i][0])):
                    tree.tree_.value[i][0][j] = 0
    
    clf_bruit = clf_bruit.estimators_[0]
    
    clf = clf.estimators_[0]
    
    print("DT accuracy = ", clf.score(X, y))

    import sklearn
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    import numpy as np

    def strike(text):
        result = ''
        for c in text:
            result = result + c + '\u0336'
        return result

    matplotlib.use('Agg')
    from matplotlib import rc

    rc('text',usetex=True)
    rc('text.latex', preamble=r'\usepackage{color, soul}')
    fig, ax = plt.subplots(figsize=(9,4))

    plot_tree(clf, ax=ax, feature_names = features, fontsize=18)

    def replace_text(obj):
        global leaf_id
        global line_id
        # parses the text within one node (internal node or leaf)
        if type(obj) == matplotlib.text.Annotation:
            txt = obj.get_text()
            txt = txt.split("\n")
            newtxt = ""
            first = True
            isleaf = True
            # parses the text line by line
            clean_counts = [[r'\textcolor{red}{\st{1}} ', r'\textcolor{red}{\st{0}} ', r'\textcolor{red}{\st{1}} ', r'\textcolor{red}{\st{0}} '], [r'\textcolor{red}{\st{0}} ', r'\textcolor{red}{\st{1}} ', r'\textcolor{red}{\st{1}} ', r'\textcolor{red}{\st{0}} ']]
            noisy_counts = [[r'\textcolor{blue}{\textbf{1}}' , r'\textcolor{blue}{\textbf{1}}', r'\textcolor{blue}{\textbf{1}}', r'\textcolor{blue}{\textbf{0}}'], [r'\textcolor{blue}{\textbf{0}}', r'\textcolor{blue}{\textbf{1}}', r'\textcolor{blue}{\textbf{3}}', r'\textcolor{blue}{\textbf{1}}']]
            lines = ["$f_1 \leq 0.573$", "$f_2 \leq 0.421$", "$f_3 \leq 0.545$"]
            for line in txt:
                if "impurity = 0.0" not in line:
                    if ("<=" in line) or (">" in line):
                        isleaf=False
                    if not ("gini" in line) and not("samples" in line):
                        if "value" in line:
                            if isleaf:
                                #line = line.replace("value =", "per-class #examples:")
                                line = re.split('[ | ]', line)
                                line_els = [l.replace('[','').replace(']','').replace(',','') for l in line]
                                cards = line_els[line_els.index('=')+1:]
                                line = ''
                                for i, c in enumerate(cards):
                                    if i > 0:
                                        line += '\n'
                                    line += '\#Class %d: ' %i + clean_counts[i][leaf_id] + noisy_counts[i][leaf_id]  #\u0336  example
                                    #if int(noisy_counts[i][leaf_id]) > 1:
                                    #    line += 's'
                                    #line += "\n"
                                    #line += '             ' + '$\it{%d~example}$' %(noisy_counts[i][leaf_id])
                                    #if int(noisy_counts[i][leaf_id]) > 1:
                                    #    line += 's'
                                leaf_id += 1
                            else: # If it is an internal node, do not keep the number of samples!
                                continue
                        if first:
                            if ("<=" in line) or (">" in line):
                                line = lines[line_id]
                                line_id += 1
                            newtxt = newtxt + line
                            first = False
                        else:
                            newtxt = newtxt + "\n" + line
            print(newtxt)
            obj.set_text(newtxt)
        return obj
    leaf_id = 0
    line_id = 0
    #matplotlib.use('agg')
    ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]
    # Scale the nodes by intercepting their rendering
    for text in ax.texts:  # Iterate through all text elements in the tree
        bbox = text.get_bbox_patch()
        if bbox:  # If there's a bounding box around the text
            bbox.set_boxstyle("Square,pad=0.4")  # Keep the rounded style
            #bbox.set_width(bbox.get_width() * 3)  # Scale box width
            #bbox.set_height(bbox.get_height() * 3)  # Scale box height
            #bbox.set_facecolor("lightblue")  # Optional: change fill color
            #bbox.set_edgecolor("black")  # Optional: change border color

    fig.savefig("figures/tree_before_after_noise.ps", bbox_inches='tight')
    plt.clf()

    def retrieve_branches(number_nodes, children_left_list, children_right_list, nodes_features_list, nodes_value_list):
        """Retrieve decision tree branches"""
        
        # Calculate if a node is a leaf
        is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
        
        # Store the branches paths
        paths = []
        
        for i in range(number_nodes):
            if is_leaves_list[i]:
                # Search leaf node in previous paths
                end_node = [path[-1] for path in paths]

                # If it is a leave node yield the path
                if i in end_node:
                    output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                    output = output[:-1]
                    yield (output, list(nodes_value_list[i][0]))

            else:
                
                # Origin and end nodes
                origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

                # Iterate over previous paths to add nodes
                for index, path in enumerate(paths):
                    if origin == path[-1]:
                        path[-1] = -nodes_features_list[origin]
                        paths[index] = path + [end_l]
                        path[-1] = nodes_features_list[origin]
                        paths.append(path + [end_r])

                # Initialize path in first iteration
                if i == 0:
                    paths.append([-nodes_features_list[i], children_left[i]])
                    paths.append([nodes_features_list[i], children_right[i]])


    t = clf.tree_


    n_nodes = t.node_count
    children_left = t.children_left # For all nodes in the tree, list of their left children (or -1 for leaves)
    children_right = t.children_right # For all nodes in the tree, list of their right children (or -1 for leaves)
    nodes_features = t.feature # For all nodes in the tree, list of their used feature (or -2 for leaves)
    # Depending on sklearn version different parsing must be done here
    sklearn_version = str(sklearn.__version__).split(".")
    if int(sklearn_version[0]) <= 1 and int(sklearn_version[1]) <= 3:
        nodes_value = t.value # For all nodes in the tree, list of their value (support for both classes)
    else:
        total_examples = t.n_node_samples
        nodes_value = t.value # For all nodes in the tree, list of their value (relative support for both classes)
        for i in range(len(nodes_value)):     # For each node           
            #print(total_examples[i], nodes_value[i])
            for j in range(len(nodes_value[i][0])):
                nodes_value[i][0][j] = np.round(nodes_value[i][0][j] * total_examples[i], decimals=0)
            #print(total_examples[i], nodes_value[i], '\n')
            assert(sum(nodes_value[i][0]) == total_examples[i]) # just make sure there were no rounding error

    nodes_features += 1

    all_branches = list(retrieve_branches(n_nodes, children_left, children_right, nodes_features, nodes_value))
    print(all_branches)
    

"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Parameters for the plot
N_trees = np.arange(1, 30, 1)
N_samples = 125
epsilon = [0.1, 1, 5, 10, 20, 30]
iterations = 10

# Initialize dictionaries to store the values
N_avg_values = {eps: np.zeros(len(N_trees)) for eps in epsilon}
N_max_values = {eps: np.zeros(len(N_trees)) for eps in epsilon}
N_min_values = {eps: np.zeros(len(N_trees)) for eps in epsilon}

# Loop over the number of iterations
for i in range(iterations):
    # Loop over the values of N_trees
    for idx, ntrees in enumerate(N_trees):
        # Loop over the values of epsilon
        for eps in epsilon:
            eps_etape = eps / ntrees
            clf_noised, clf, eps_etape, X_train, X_test, y_train, y_test = create_laplace_tree('data/compas.csv', N_samples, ntrees, eps, i)
            N_avg, N_max, N_min = intervalle_N(clf_noised, eps_etape, ntrees)

            # Store the values in the corresponding lists
            N_avg_values[eps][idx] += N_avg
            N_max_values[eps][idx] += N_max
            N_min_values[eps][idx] += N_min

# Average the values over the number of iterations
for eps in epsilon:
    N_avg_values[eps] /= iterations
    N_max_values[eps] /= iterations
    N_min_values[eps] /= iterations

# Plot the curves
plt.figure(figsize=(10, 6))
for eps in epsilon:
    color = next(plt.gca()._get_lines.prop_cycler)['color']  # to get consistent colors
    plt.plot(N_trees, N_max_values[eps], label=f'N_min, N_max (ε = {eps})', alpha=0.4, color=color)
    plt.plot(N_trees, N_min_values[eps], alpha=0.4, color=color)
    plt.fill_between(N_trees, N_min_values[eps], N_max_values[eps], alpha=0.2, color=color)

# Dashed line representing N_samples
plt.axhline(y=N_samples * 0.8, color='black', linestyle='--', label='N_vrai')

# Adjust labels and legend
plt.xlabel('#trees')
plt.ylabel('N')
plt.legend()
#plt.title('Search interval of N around N_avg as a function of N_trees (with probability > 0.95)')
plt.savefig("figures/interval_N.pdf", bbox_inches='tight')
plt.clf()"""