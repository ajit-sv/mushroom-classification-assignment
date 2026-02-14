import numpy as np

class NaiveBayesScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.likelihoods = {}

        for cls in self.classes:
            X_c = X[y==cls]
            self.priors[cls] = len(X_c)/len(X)
            self.likelihoods[cls] = (np.sum(X_c, axis=0)+1) / \
                                    (np.sum(X_c)+X.shape[1])

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                likelihood = np.sum(x*np.log(self.likelihoods[cls]))
                posteriors.append(prior+likelihood)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)