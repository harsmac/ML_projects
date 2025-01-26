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

# add GD version of LR


class LogisticRegression:
    def __init__(self, n_iters=10, lr=0.01):
        self.wts = None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters
    
    def sigmoid(self, X, wts, b):
        num = X @ wts + b * np.ones((X.shape[0], 1))
        return 1/(1 + np.exp(-num))
    
    def fit(self, X, y):
        N, d = X.shape
        self.wts = np.random.rand(d, 1)
        self.bias = 0
        for _ in range(self.n_iters):
            sigm_x = self.sigmoid(X, self.wts, self.bias)
            # neg_log_likelihood_loss = - np.mean(y @ np.log(sigm_x) + (1 - y) @ np.log(1 - sigm_x))
            
            # derivatives
            db = (1/N) * (sigm_x - y.reshape(N, 1))
            dw = X.T @ (db)
            
            self.wts -= self.lr * dw
            self.bias -= self.lr * db
        
        return self.wts, self.bias
    
    def predict(self, x_test):
        return self.sigmoid(x_test, self.wts, self.bias)

class SVM:
    def __init__(self, n_iters=10, lamb=0.01, lr=0.01):
        self.wts = None
        self.bias = None
        self.n_iters = n_iters
        self.lamb = lamb
        self.lr = lr
    
    def fit(self, X, y):
        N, d = X.shape
        self.wts = np.random.rand((d, 1))
        self.bias = 0
        
        for _ in range(self.n_iters):
            curr_prod = y @ (X @ self.wts + self.bias * np.ones(N, 1)) - 1
            
            dw = np.zeros_like(curr_prod)  # Initialize dw with zeros
            dw[curr_prod > 0] = X[curr_prod > 0].T @ y[curr_prod > 0]  # Set dw where curr_prod > 0
            dw = dw + self.lamb * self.wts
            db = np.zeros_like(curr_prod)
            db[curr_prod > 0] = y[curr_prod > 0]
            
            self.wts -= self.lr * dw
            self.bias -= self.lr * db
        
        return self.wts, self.bias
            
    
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

