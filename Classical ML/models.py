import numpy as np


class LinearRegression:
    def __init__(self, add_bias=False, add_reg=False, lamb=0.01):
        self.wts = None
        self.add_bias = add_bias
        if add_reg:
            self.lamb = lamb
        else:
            self.lamb = 0

    def fit(self, X, y):
        # assumption here is X is N x d where N is number of data points and d is its feature dimension
        # y is N x 1
        # weights will be d x 1

        if self.add_bias:
            N = X.shape[0]
            X = np.hstack(
                np.ones(N, 1), X
            )  # so now X_train becomes N x (d + 1) dimension

        d = X.shape[1]
        self.wts = np.linalg.pinv(X.T @ X + (np.identity(d) * self.lamb)) @ X.T @ y
        return self.wts

    def predict(self, x_test):
        if self.wts is None:
            raise ValueError("Run fit first to obtain the weights for training data!")
        
        if self.add_bias:
            n = x_test.shape[0]
            x_test = np.hstack(np.ones(n, 1), x_test)
            
        y_test = x_test @ self.wts
        return y_test


class LogisticRegression:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, x_test):
        pass

class SVM:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, x_test):
        pass



class KNN:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, x_test):
        pass



class KMeansClustering:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, x_test):
        pass

