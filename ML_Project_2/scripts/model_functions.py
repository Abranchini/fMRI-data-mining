# This file contains the functions concerning the implementation
# of the models for the minimization problems

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor, MultiTaskElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.cluster import KMeans

def leastSquares(X,y):
    linReg_coef = [] # weights to be optimized
    linReg=LinearRegression()
    linReg.fit(X,y)
    linReg_coef=linReg.coef_
    # Make predictions 
    y_pred = linReg.predict(X)
    mse=mean_squared_error(y_pred, y)
    # Display results
    #print('Linear Regression coefficients: \n', linReg_coef)
    #print("Mean squared error: %.4f"
     #   % mean_squared_error(y_pred, y))
    return linReg_coef, mse

def ridgeRegression(X,y,lamda_): # use regularization factor to 1
    ridge = Ridge(alpha=lamda_, copy_X=True, fit_intercept=False, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
    ridge.fit(X, y)
    y_pred=ridge.predict(X)
    mse=mean_squared_error(y_pred, y)
    return ridge.coef_,mse # coefficients

def lassoRegression(X,y,lambda_): # use regularization factor to 1
    lasso=Lasso(alpha=lambda_, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
    lasso.fit(X,y)
    y_pred=lasso.predict(X)
    return lasso.coef_, mean_squared_error(y_pred, y)

def elasticNet(X,y,lambda_1,lambda_2):
    alpha = lambda_1 + lambda_2
    l1_ratio = lambda_1 / (lambda_1 + lambda_2)
    elNet=ElasticNet(alpha=alpha, copy_X=True, fit_intercept=True, l1_ratio=l1_ratio,
    max_iter=30000, normalize=False, positive=False, precompute=False,
    random_state=0, selection='cyclic', tol=0.0001, warm_start=False)
    elNet.fit(X, y)
    y_pred=elNet.predict(X)
    mse=mean_squared_error(y_pred, y)
    return elNet.coef_, mse

def multiTaskElasticNet(X,y,lambda_1,lambda_2):
    alpha = lambda_1 + lambda_2
    l1_ratio = lambda_1 / (lambda_1 + lambda_2)
    MelNet=MultiTaskElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, 
    normalize=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, 
    random_state=None, selection='cyclic')
    MelNet.fit(X, y)
    y_pred=MelNet.predict(X)
    mse=mean_squared_error(y_pred, y)
    return MelNet.coef_, mse

def reshapeCoefficients(coef,N,L):
    # The returned coefficients are stored in a vector of size L*N
    # However we are interested on having a vector of lenght N with the
    # L values (refer to report for more details)
    matCoef=np.reshape(coef.T, (N,L))
    return matCoef

#----------------------------------------------------- Elastic net 

def MSE(actual, predicted, err = []):
    """ Calculate the mean square error
    """

    if len(err) != 0:
        mean_squared_error = err.dot(err)/len(actual)
    else:
        delta = actual - predicted
        mean_squared_error = delta.dot(delta)/len(actual)

    return mean_squared_error

def compute_gradient(X_train, y, beta, l1_ratio, alpha):
    """Compute the gradient for the own implementation elastic net
    """
    err = y - X_train.dot(beta)
    grad = (1/(2*len(y))) * X_train.T.dot(err) + alpha*l1_ratio*np.sign(beta) + alpha*(1-l1_ratio)*0.5*beta
    return grad, err

def elastic_net(iterations, gamma, X_train, y_train, l1,l2, alpha):
    """First try of own implementation of elastic net
    """

    # Intiate variables
    l1_ratio = l1 / (l1 + l2)
    D = X_train.shape[1]
    beta = np.random.randn(D) / np.sqrt(D)

    for n_iter in range(iterations):
        grad, err = compute_gradient(X_train, y_train, beta, l1_ratio, alpha)
        beta = beta - gamma * grad        
        mse = MSE(actual = y_train, predicted = None, err = err)

        print("Elastic Net with Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=iterations - 1, l=mse))       

    

    return beta, mse 

## KMEANS
'''def apply_kmeans(data, numberClusters):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
return '''