import numpy as np

class DecisionTreeScratch:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def gini(self, y):
        classes = np.unique(y)
        gini = 1
        for cls in classes:
            p = np.sum(y==cls)/len(y)
            gini -= p**2
        return gini

    def split(self, X, y, feature, threshold):
        left = X[:,feature] <= threshold
        right = X[:,feature] > threshold
        return X[left], X[right], y[left], y[right]

    def best_split(self, X, y):
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                X_l, X_r, y_l, y_r = self.split(X,y,feature,threshold)
                if len(y_l)==0 or len(y_r)==0:
                    continue
                g = (len(y_l)/len(y))*self.gini(y_l) + \
                    (len(y_r)/len(y))*self.gini(y_r)
                if g < best_gini:
                    best_gini = g
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build(self, X, y, depth):
        if depth>=self.max_depth or len(np.unique(y))==1:
            return np.bincount(y).argmax()

        feature, threshold = self.best_split(X,y)
        if feature is None:
            return np.bincount(y).argmax()

        X_l, X_r, y_l, y_r = self.split(X,y,feature,threshold)
        return {
            'feature':feature,
            'threshold':threshold,
            'left':self.build(X_l,y_l,depth+1),
            'right':self.build(X_r,y_r,depth+1)
        }

    def fit(self, X, y):
        self.tree = self.build(X,y,0)

    def predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_sample(x, tree['left'])
        return self.predict_sample(x, tree['right'])

    def predict(self, X):
        return np.array([self.predict_sample(x,self.tree) for x in X])