import numpy
import cvxopt.solvers


#Trains an SVM
class SVM(object):
    def __init__(self, kernel, bias=None, svmultipliers=None, sv=None, svlabels=None, c=None):
        self._kernel = kernel
        self._bias = bias
        self._svmultipliers = svmultipliers
        self._sv = sv
        self._svlabels = svlabels
        self._c = c
        if self._c is not None: self._c = float(self._c)
        
    #returns trained SVM predictor given features (X) & labels (y)
    def fit(self, X, y):
        lagrangeMultipliers = self.multipliers(X, y)
        svindices = lagrangeMultipliers > 1e-5
        svmultipliers = lagrangeMultipliers[svindices]
        sv = X[svindices]
        svlabels = y[svindices]
        
        #compute error assuming zero bias
        bias = numpy.mean(
            [y_i - SVM(self._kernel, 0.00, svmultipliers, sv, svlabels).predict_item(x_i)
            for (y_i, x_i) in zip(svlabels, sv)])  
    
        self._bias = bias
        self._svmultipliers = svmultipliers
        self._sv = sv
        self._svlabels = svlabels
        
    #compute Gram matrix
    def gram(self, X):
        n_samples, n_features = X.shape
        K = numpy.zeros((n_samples, n_samples))
        for i in range(0, n_samples):
            for j in range(0, n_samples):
                K[i, j] = self._kernel(X[i], X[j])
        return K
        
    #compute Lagrangian multipliers
    def multipliers(self, X, y):
        n_samples, n_features = X.shape
        K = self.gram(X)
        
        P = cvxopt.matrix(numpy.outer(y,y) * K)
        q = cvxopt.matrix(numpy.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), 'd')
        b = cvxopt.matrix(0.00)
        
        if self._c is None:
            G = cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1))
            h = cvxopt.matrix(numpy.zeros(n_samples))
        else:
            G_1 = cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1))
            h_1 = cvxopt.matrix(numpy.zeros(n_samples))
            
            G_2 = cvxopt.matrix(numpy.diag(numpy.ones(n_samples)))
            h_2 = cvxopt.matrix(numpy.ones(n_samples) * self._c)
            
            G = cvxopt.matrix(numpy.vstack(G_1, G_2))
            h = cvxopt.matrix(numpy.vstack(h_1, h_2))
            
        soln = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        return numpy.ravel(soln['x'])
    
    #Returns SVM predicton given feature vector
    def predict_item(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._svmultipliers, self._sv, self._svlabels):
            result += z_i * y_i * self._kernel(x_i, x)
        return numpy.sign(result).item()
            
    def predict(self, X):
        y_predict = numpy.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for z_i, x_i, y_i in zip(self._svmultipliers, self._sv, self._svlabels):
                s += z_i * y_i * self._kernel(x_i, X[i])
            y_predict[i] = s
        return numpy.sign(y_predict + self._bias)