import numpy as np
from collections import Counter


class Node:
    """Decision tree node for XGBoost."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Leaf value (for regression)


class XGBoostTree:
    """Regression tree for XGBoost built from scratch."""
    
    def __init__(self, max_depth=5, min_child_weight=1, gamma=0, lambda_reg=1.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma              # Minimum loss reduction to split
        self.lambda_reg = lambda_reg    # L2 regularization
        self.root = None
    
    def fit(self, X, y, gradients, hessians):
        """Build XGBoost tree using gradients and hessians."""
        self.root = self._build_tree(X, y, gradients, hessians, depth=0)
        return self
    
    def _build_tree(self, X, y, gradients, hessians, depth=0):
        """Recursively build the regression tree."""
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < 2:
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            return Node(value=leaf_value)
        
        # Find best split
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                # Split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_child_weight or \
                   np.sum(right_mask) < self.min_child_weight:
                    continue
                
                # Calculate gain
                gain = self._calculate_gain(
                    gradients, hessians,
                    left_mask, right_mask
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_idx = left_mask
                    best_right_idx = right_mask
        
        # If no split found, create leaf
        if best_feature is None:
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            return Node(value=leaf_value)
        
        # Recursive tree building
        left_gradients = gradients[best_left_idx]
        left_hessians = hessians[best_left_idx]
        right_gradients = gradients[best_right_idx]
        right_hessians = hessians[best_right_idx]
        
        left_X = X[best_left_idx]
        right_X = X[best_right_idx]
        left_y = y[best_left_idx]
        right_y = y[best_right_idx]
        
        left_subtree = self._build_tree(
            left_X, left_y, left_gradients, left_hessians, depth + 1
        )
        right_subtree = self._build_tree(
            right_X, right_y, right_gradients, right_hessians, depth + 1
        )
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree)
    
    def _calculate_gain(self, gradients, hessians, left_mask, right_mask):
        """Calculate information gain for XGBoost split."""
        # Get split groups
        gl = np.sum(gradients[left_mask])
        hl = np.sum(hessians[left_mask])
        gr = np.sum(gradients[right_mask])
        hr = np.sum(hessians[right_mask])
        
        left_score = (gl ** 2) / (hl + self.lambda_reg)
        right_score = (gr ** 2) / (hr + self.lambda_reg)
        parent_score = ((gl + gr) ** 2) / (hl + hr + self.lambda_reg)
        
        gain = 0.5 * (left_score + right_score - parent_score) - self.gamma
        
        return gain
    
    def _calculate_leaf_value(self, gradients, hessians):
        """Calculate optimal leaf value using Newton step."""
        g_sum = np.sum(gradients)
        h_sum = np.sum(hessians)
        
        # Newton-Raphson step: -G/H
        learning_rate = 0.1  # Shrinkage parameter
        leaf_value = -learning_rate * (g_sum / (h_sum + 1.0))
        
        return leaf_value
    
    def predict(self, X):
        """Predict values for X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class XGBoostClassifierScratch:
    """XGBoost classifier built from scratch for binary classification."""
    
    def __init__(self, n_rounds=100, max_depth=5, learning_rate=0.1, 
                 min_child_weight=1, gamma=0, subsample=1.0,
                 colsample_bytree=1.0, lambda_reg=1.0, random_state=None):
        """
        Initialize XGBoost classifier.
        
        Parameters:
        - n_rounds: Number of boosting rounds
        - max_depth: Maximum depth of trees
        - learning_rate: Shrinkage parameter (eta)
        - min_child_weight: Minimum sum of hessians in child
        - gamma: Minimum loss reduction to split
        - subsample: Fraction of samples for each tree
        - colsample_bytree: Fraction of features for each tree
        - lambda_reg: L2 regularization parameter
        - random_state: Random seed
        """
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        
        self.trees = []
        self.base_score = 0
        self.train_scores = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _log_loss_gradient(self, y, pred):
        """Gradient of log loss for binary classification."""
        return self._sigmoid(pred) - y
    
    def _log_loss_hessian(self, y, pred):
        """Hessian of log loss for binary classification."""
        p = self._sigmoid(pred)
        return p * (1 - p)
    
    def fit(self, X, y):
        """Build XGBoost classifier."""
        n_samples, n_features = X.shape
        
        # Initialize predictions with log odds
        p = np.mean(y)
        self.base_score = np.log(p / (1 - p) + 1e-10)
        predictions = np.full(n_samples, self.base_score, dtype=np.float32)
        
        self.trees = []
        self.train_scores = []
        
        # Boosting rounds
        for round_idx in range(self.n_rounds):
            # Calculate gradients and hessians
            gradients = self._log_loss_gradient(y, predictions)
            hessians = self._log_loss_hessian(y, predictions)
            
            # Subsample data
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    n_samples, 
                    int(n_samples * self.subsample),
                    replace=False
                )
            else:
                sample_indices = np.arange(n_samples)
            
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            grad_sample = gradients[sample_indices]
            hess_sample = hessians[sample_indices]
            
            # Feature subsampling
            if self.colsample_bytree < 1.0:
                n_cols = max(1, int(n_features * self.colsample_bytree))
                feature_indices = np.random.choice(
                    n_features, n_cols, replace=False
                )
                X_sample = X_sample[:, feature_indices]
            else:
                feature_indices = np.arange(n_features)
            
            # Build tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                lambda_reg=self.lambda_reg
            )
            tree.fit(X_sample, y_sample, grad_sample, hess_sample)
            self.trees.append((tree, feature_indices))
            
            # Update predictions
            # For full X, need to handle feature subsampling
            if self.colsample_bytree < 1.0:
                tree_pred = np.zeros(n_samples)
                X_full_features = X[:, feature_indices]
                tree_pred = tree.predict(X_full_features)
            else:
                tree_pred = tree.predict(X)
            
            predictions += self.learning_rate * tree_pred
            
            # Track training score
            train_pred_binary = (predictions > 0).astype(int)
            train_accuracy = np.mean(train_pred_binary == y)
            self.train_scores.append(train_accuracy)
        
        return self
    
    def predict_proba(self, X):
        """Predict probability for class 1."""
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.base_score, dtype=np.float32)
        
        for tree, feature_indices in self.trees:
            if len(feature_indices) < X.shape[1]:
                X_features = X[:, feature_indices]
            else:
                X_features = X
            
            tree_pred = tree.predict(X_features)
            predictions += self.learning_rate * tree_pred
        
        # Apply sigmoid to get probabilities
        proba = self._sigmoid(predictions)
        return proba
    
    def predict(self, X):
        """Predict class labels for X."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
