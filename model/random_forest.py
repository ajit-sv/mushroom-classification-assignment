import numpy as np
from collections import Counter


class Node:
    """Decision tree node."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Class value if leaf node


class DecisionTree:
    """Decision tree classifier built from scratch."""
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X, y):
        """Build decision tree classifier."""
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                # Split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                # Information gain
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # If no split found, create leaf
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Recursive tree building
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree)
    
    def _gini(self, y):
        """Calculate Gini impurity."""
        counter = Counter(y)
        gini = 1.0
        for count in counter.values():
            prob = count / len(y)
            gini -= prob ** 2
        return gini
    
    def _information_gain(self, parent, left_child, right_child):
        """Calculate information gain using Gini impurity."""
        n = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_gini = self._gini(parent)
        left_gini = self._gini(left_child)
        right_gini = self._gini(right_child)
        
        child_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        
        return parent_gini - child_gini
    
    def predict(self, X):
        """Predict class for X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class RandomForestScratch:
    """Random Forest classifier built from scratch."""
    
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, 
                 n_features=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        """Build random forest."""
        self.trees = []
        n_features = X.shape[1]
        
        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Build tree
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict class for X."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        
        return np.array(final_predictions)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
