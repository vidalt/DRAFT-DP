
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

def compute_forest_serialization_normalization(dp_rf_class, path, dataset, N_samples, N_trees, ohe_groups, depth, seed, verbosity, epsilon, X, y, n_iters=500):
    forest_samples = []
    for i in range(n_iters):
        np.random.seed(seed + i)
        # subsample training data
        indices = np.random.choice(len(X), size=N_samples, replace=False)
        X_train_sample = X[indices]
        y_train_sample = y[indices]
        clf = dp_rf_class(path, dataset, N_samples, N_trees, ohe_groups, depth, seed, verbosity)
        clf.fit(pd.DataFrame(X_train_sample), y_train_sample)
        clf.add_noise(epsilon)
        forest_vec = serialize_forest_leaf_counts_minus_known_fixed(
            clf,
            X_train_sample[:-1],
            y_train_sample[:-1],
            n_classes=2,
            max_leaves=2 ** depth,
        )  # forest_vec: shape (D_forest,)

        forest_samples.append(forest_vec)

    forest_mat = np.stack(forest_samples, axis=0)

    # Per-feature stats: shape (D_forest,)
    mean = forest_mat.mean(axis=0)
    std = forest_mat.std(axis=0)
    
    return mean.astype(np.float32), std.astype(np.float32)

def serialize_dp_forest_minus_known(
    rf,
    X_known,
    y_known,
    max_depth,
    n_classes,
    mean=None,
    std=None,
):
    """
    Serialize a sklearn-like RandomForest into a fixed-length vector, after subtracting
    the contributions of known examples from the leaf counts and propagating the
    adjustment up to internal nodes (so internal counts = sum of child counts).

    Parameters
    ----------
    rf : object
        RandomForest-like object with an `estimators_` attribute.
    X_known : array-like, shape (N_known, d)
        Known feature matrix.
    y_known : array-like, shape (N_known,)
        Known labels in {0, ..., n_classes-1}.
    max_depth : int
        Max depth of each tree (used to set a fixed max_nodes).
    n_classes : int
        Number of classes.
    mean, std : np.ndarray or None
        Optional per-feature normalization stats of shape (D_forest,).

    Returns
    -------
    forest_vec : np.ndarray, shape (D_forest,)
        Flattened forest representation with known contributions removed and
        internal node counts recomputed.
    """
    X_known = np.asarray(X_known)
    y_known = np.asarray(y_known, dtype=np.int32)

    estimators = rf.estimators_
    n_trees = len(estimators)

    max_nodes = 2 ** (max_depth + 1) - 1  # full binary tree upper bound per tree

    per_tree_dim = (
        max_nodes              # feature indices
        + max_nodes            # thresholds
        + max_nodes * n_classes  # values (class counts)
    )

    forest_vec = np.zeros(n_trees * per_tree_dim, dtype=np.float32)

    offset = 0
    for est in estimators:
        tree = est.tree_
        n_nodes = tree.node_count

        # --- 1. Original values (counts) ---
        raw_val = tree.value  # (n_nodes, 1, n_classes) or (n_nodes, n_classes)
        if raw_val.ndim == 3:
            raw_val = raw_val[:, 0, :]  # -> (n_nodes, n_classes)
        raw_val = raw_val.astype(np.float32, copy=True)

        children_left = tree.children_left
        children_right = tree.children_right

        # --- 2. Compute known contributions per leaf ---
        # leaf_ids_known[i] = leaf index reached by X_known[i] in this tree
        leaf_ids_known = est.apply(X_known)  # shape (N_known,)
        known_counts = np.zeros_like(raw_val, dtype=np.float32)
        # accumulate #known examples of class c in each leaf
        np.add.at(known_counts, (leaf_ids_known, y_known), 1.0)
        # subtract, clip at 0
        adjusted_val = raw_val - known_counts
        adjusted_val = np.maximum(adjusted_val, 0.0)

        # --- 3. Propagate counts up to internal nodes ---
        # after this, each internal node = sum(children), leaves keep adjusted leaf counts
        def propagate(node_id):
            left = children_left[node_id]
            right = children_right[node_id]

            # leaf
            if left == -1 and right == -1:
                return adjusted_val[node_id]

            total = np.zeros(n_classes, dtype=np.float32)
            if left != -1:
                total += propagate(left)
            if right != -1:
                total += propagate(right)

            adjusted_val[node_id] = total
            return total

        # launch from root
        propagate(0)

        # --- 4. Features and thresholds (same as before) ---
        feat = np.full(max_nodes, -1, dtype=np.int32)
        feat[:n_nodes] = tree.feature

        thr = np.zeros(max_nodes, dtype=np.float32)
        thr[:n_nodes] = tree.threshold

        # store adjusted values into (max_nodes, n_classes)
        values = np.zeros((max_nodes, n_classes), dtype=np.float32)
        values[:n_nodes, :] = adjusted_val

        # --- 5. Flatten ---
        tree_vec = np.concatenate(
            [
                feat.astype(np.float32),
                thr,
                values.reshape(-1),
            ],
            axis=0,
        )
        assert tree_vec.shape[0] == per_tree_dim

        forest_vec[offset:offset + per_tree_dim] = tree_vec
        offset += per_tree_dim

    if mean is not None and std is not None:
        forest_vec = (forest_vec - mean.astype(np.float32)) / (std.astype(np.float32) + 1e-8)

    return forest_vec

def serialize_dp_forest(rf, max_depth, n_classes, mean=None, std=None):
    """
    Serialize a sklearn-like RandomForest (inside your DP_RF) into a fixed-length 1D numpy array.

    Parameters
    ----------
    dp_rf_model : object
        Your DP RF *forest* object. This should behave like sklearn.ensemble.RandomForestClassifier
        (i.e., have an `estimators_` attribute with sklearn DecisionTreeClassifiers).
        Typically  something like `dp_rf_model.rf_` or `dp_rf_model.clf_`.
    max_depth : int
        Maximum depth of each tree.
    n_classes : int
        Number of classes.

    Returns
    -------
    forest_vec : np.ndarray, shape (D_forest,)
        Flattened representation of the entire forest.
    """
    n_trees = len(rf.estimators_)
    max_nodes = 2 ** (max_depth + 1) - 1  # full binary tree upper bound

    per_tree_dim = (           # for each tree we store:
        max_nodes              #  - feature indices
        + max_nodes            #  - thresholds
        + max_nodes * n_classes  #  - values (class logits / noisy counts)
    )

    forest_vec = np.zeros(n_trees * per_tree_dim, dtype=np.float32)

    offset = 0
    for est in rf.estimators_:
        tree = est.tree_

        n_nodes = tree.node_count

        # feature indices (int -> float for concat convenience)
        feat = np.full(max_nodes, -1, dtype=np.int32)
        feat[:n_nodes] = tree.feature

        # thresholds
        thr = np.zeros(max_nodes, dtype=np.float32)
        thr[:n_nodes] = tree.threshold

        # values: shape (n_nodes, n_classes)
        raw_val = tree.value
        if raw_val.ndim == 3:
            # shape (n_nodes, 1, n_classes)
            raw_val = raw_val[:, 0, :]
        values = np.zeros((max_nodes, n_classes), dtype=np.float32)
        values[:n_nodes, :] = raw_val

        # flatten everything
        tree_vec = np.concatenate(
            [
                feat.astype(np.float32),
                thr,
                values.reshape(-1),
            ],
            axis=0,
        )
        assert tree_vec.shape[0] == per_tree_dim

        forest_vec[offset:offset + per_tree_dim] = tree_vec
        offset += per_tree_dim

    if mean is not None and std is not None:
        forest_vec = (forest_vec - mean.astype(np.float32)) / (std.astype(np.float32) + 1e-8)

    return forest_vec  # shape (n_trees * per_tree_dim,)

def serialize_forest_leaf_counts_minus_known_fixed(
    rf, X_known, y_known, n_classes, max_leaves, mean=None, std=None
):
    """
    Same as above but produces a fixed-length flat vector:
        (n_trees * max_leaves * n_classes,)
    """
    X_known = np.asarray(X_known)
    y_known = np.asarray(y_known, dtype=np.int32)

    n_trees = len(rf.estimators_)
    forest_vec = np.zeros(n_trees * max_leaves * n_classes, dtype=np.float32)
    offset = 0

    for est in rf.estimators_:
        tree = est.tree_

        raw_val = tree.value
        if raw_val.ndim == 3:
            raw_val = raw_val[:, 0, :]
        raw_val = raw_val.astype(np.float32)

        children_left = tree.children_left
        children_right = tree.children_right
        is_leaf = (children_left == -1) & (children_right == -1)
        leaf_indices = np.where(is_leaf)[0]

        # subtract known examples
        leaf_ids_known = est.apply(X_known)
        known_counts = np.zeros(raw_val.shape, dtype=np.float32)
        np.add.at(known_counts, (leaf_ids_known, y_known), 1.0)
        adjusted_val = np.maximum(raw_val - known_counts, 0.0)

        leaf_counts = adjusted_val[leaf_indices]    # shape (n_leaves, n_classes)
        n_leaves = leaf_counts.shape[0]
        assert n_leaves <= max_leaves

        padded = np.zeros((max_leaves, n_classes), dtype=np.float32)
        padded[:n_leaves] = leaf_counts

        forest_vec[offset:offset + max_leaves * n_classes] = padded.reshape(-1)
        offset += max_leaves * n_classes

    if mean is not None and std is not None:
        forest_vec = (forest_vec - mean) / (std + 1e-8)

    return forest_vec


class DPRFReconstructorBCE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, forest_vec):
        # returns logits; apply sigmoid outside if needed
        return self.net(forest_vec)

    
class DPRFReconstructionDataset(Dataset):
    """
    On-the-fly generator of (serialized_forest, x_star) pairs for an informed-adversary attack.

    At each index:
        - draw a random seed
        - sample a target index i*
        - train a DP_RF on the full dataset
        - serialize the DP RF
        - return (forest_vec, x_star)
    """

    def __init__(
        self,
        X,
        y,
        dp_rf_class,
        dataset_name,
        ohe_groups,
        n_trees,
        max_depth,
        eps,
        base_seed=0,
        n_pairs=10,
        N_samples=100,
        mean=None,
        std=None,
    ):
        """
        Parameters
        ----------
        X_train : np.ndarray, shape (N, d)
            Feature matrix used to train each DP RF. (= actual DP RF training data minus the unknown example)
        y_train : np.ndarray, shape (N,)
            Labels used to train each DP RF.  (= actual DP RF training labels minus the unknown example)
        X : np.ndarray, shape (N, d)
            Feature matrix of the underlying data distribution.
        y : np.ndarray, shape (N,)
            Labels.
        dp_rf_class : class
            Your DP_RF class (e.g. imported from DP_RF.py).
        dataset_name : str
            As used in your DP_RF constructor (e.g., "compas").
        ohe_groups : list
            One-hot groups used by your DP_RF.
        n_trees : int
            Number of trees in the DP RF.
        max_depth : int
            Maximum depth of each tree.
        eps : float
            DP budget to pass to `add_noise`.
        base_seed : int
            Base seed; will be offset by the dataset index.
        """
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y
        self.N, self.d = self.X.shape
        self.n_pairs = n_pairs
        self.dp_rf_class = dp_rf_class
        self.dataset_name = dataset_name
        self.ohe_groups = ohe_groups
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.eps = eps
        self.base_seed = base_seed
        self.gen_items = 0
        self.n_classes = len(np.unique(self.y))
        self.N_samples = N_samples
        self.mean = mean
        self.std = std
    def __len__(self):
        # finite number of (forest, x*) pairs
        return self.n_pairs


    def __getitem__(self, idx):
        # Use idx to derive a deterministic seed
        seed = self.base_seed #+ idx
        

        data_sampling_seed = self.base_seed + idx
        rng = np.random.RandomState(data_sampling_seed)

        # ----- 1. Subsample N_samples training points -----
        indices = rng.choice(self.N, size=self.N_samples, replace=False)
        X_training = self.X[indices]
        y_training = self.y[indices]
       
        i_star = -1  # last one in the sampled indices
        x_star = self.X[indices[i_star]]

        # ----- 2. Build a training set for the DP RF -----
        # concatenate x_star to training data
        assert(X_training.shape[0] == self.N_samples)
        # ----- 3. Train DP RF -----
        dp_rf = self.dp_rf_class(
            path=None,                    # or some dummy path
            dataset=self.dataset_name,
            N_samples=X_training.shape[0],
            N_trees=self.n_trees,
            ohe_groups=self.ohe_groups if hasattr(self.dp_rf_class, "one_hot_groups") else self.ohe_groups,
            depth=self.max_depth,
            seed=seed,
        )

        dp_rf.fit(pd.DataFrame(X_training), y_training)
        dp_rf.add_noise(self.eps)
        '''from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt
        plot_tree(dp_rf.clf.estimators_[0])
        plt.savefig(f"figures/balle_tree_plot_{idx}advRF.pdf")
        plt.clf()'''
        # ----- 4. Serialize the noisy DP RF -----
        assert(X_training[:-1].shape[0] == self.N_samples - 1)
        forest_vec = serialize_forest_leaf_counts_minus_known_fixed(
            dp_rf,
            X_training[:-1], 
            y_training[:-1],
            n_classes=self.n_classes,
            max_leaves=2 ** self.max_depth,
            mean=self.mean,
            std=self.std,
        )

        forest_tensor = torch.from_numpy(forest_vec.astype(np.float32))
        x_star_tensor = torch.from_numpy(x_star.astype(np.float32))
        self.gen_items += 1
        #print("Generated items:", self.gen_items)
        return forest_tensor, x_star_tensor
    
def train_reconstructor(
    X, # other data from the same distribution
    y, # other data from the same distribution
    dp_rf_class,
    dataset_name,
    ohe_groups,
    n_trees,
    max_depth,
    eps,
    n_pairs=10,
    batch_size=16,
    n_epochs=10,
    lr=1e-3,
    device="cpu",
    mean=None,
    std=None,
    N_samples=100,
    seed=0,
    n_threads=1
):
    """
    Train a reconstructor network R_theta such that:
        R_theta(serialize_dp_forest(DP_RF(D))) ~ x_star

    Returns the trained model.
    """
    # Build dataset and a finite __len__ = n_pairs
    dataset = DPRFReconstructionDataset(
        X=X,
        y=y,
        dp_rf_class=dp_rf_class,
        dataset_name=dataset_name,
        ohe_groups=ohe_groups,
        n_trees=n_trees,
        max_depth=max_depth,
        eps=eps,
        base_seed=seed,
        n_pairs=n_pairs,
        N_samples=N_samples,
        mean=mean,
        std=std,
    )

    # Instantiate one example to get input_dim / output_dim
    forest_example, x_example = dataset[0]
    input_dim = forest_example.numel()
    output_dim = x_example.numel()

    # Enable multithreading for PyTorch
    torch.set_num_threads(int(n_threads/2))        
    torch.set_num_interop_threads(int(n_threads/2))

    model = DPRFReconstructorBCE(input_dim=input_dim, output_dim=output_dim)
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(n_threads/2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        total = 0
        for forest_batch, x_batch in dataloader:
            forest_batch = forest_batch.to(device)
            x_batch = x_batch.to(device).float()  # in {0,1}

            optimizer.zero_grad()
            logits = model(forest_batch)          # (B, d)
            loss = criterion(logits, x_batch)

            loss.backward()
            optimizer.step()

            bs = forest_batch.size(0)
            epoch_loss += loss.item() * bs
            total += bs

        epoch_loss /= max(total, 1)
        print(f"[Epoch {epoch+1}/{n_epochs}] BCE loss = {epoch_loss:.6f}")

    return model
