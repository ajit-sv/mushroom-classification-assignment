import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            preds = self.sigmoid(linear)

            dw = np.dot(X.T, (preds - y))/len(y)
            db = np.sum(preds - y)/len(y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        probs = self.sigmoid(linear)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        probs = self.sigmoid(linear)
        # Store for debugging
        self.last_predict_proba = probs
        # Return as 2D array: [P(neg), P(pos)]
        return np.vstack([1-probs, probs]).T