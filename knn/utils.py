import numpy as np
from collections import Counter
class KNN:
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def _computeDistance(self, X_test):
        M = np.dot(X_test, self.X_train.T)
        t, n = M.shape
        te = np.diag(np.dot(X_test, X_test.T))
        tr = np.diag(np.dot(self.X_train, self.X_train.T))
        te = np.repeat(te, n).reshape(M.shape)
        tr = np.tile(tr, t).reshape(M.shape)
        return np.sqrt(te + tr - 2 * M)
    
    def predict(self, X_test, k):
        return self._predict(X_test, k)
    
    def _predict(self, X_test, k=1):
        distances = self._computeDistance(X_test)
        num_test = distances.shape[0]
        k_indexes = distances.argsort()[:, :k]
        y_pred_labels = np.zeros(num_test)
        for i in range(num_test):
            counter = Counter(self.y_train[k_indexes[i, :]])
            y_pred_labels[i] = np.squeeze(counter.most_common(1))[0]
        return y_pred_labels
    
    def compute_error_rate(self, X_test, y_test, k):
        pred_labels = self.predict(X_test, k)
        error_count = y_test.shape[0] - np.sum(y_test == pred_labels)
        return error_count / y_test.shape[0]