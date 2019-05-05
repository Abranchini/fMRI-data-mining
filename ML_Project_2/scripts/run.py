import sys
import importlib
import numpy as np
#sys.path.append('D:/University/EPFL/Project_MPI/ML_Project_2/scripts')
sys.path.append('/media/abranches/Main/University/EPFL/Project_MPI/ML_Project_2/scripts')
import data_functions, model_functions, graphic_functions
# Reupdate functions
importlib.reload(data_functions)
importlib.reload(model_functions)
importlib.reload(graphic_functions)

# This is the main file for getting the causality in the brain.
from data_functions import *
from model_functions import *
from graphic_functions import *


# 1. Get the data
data=obtainData2() # Obtains the data from AAL ATLAS.
T=data.shape[1] # T: number of time samples
N=data.shape[2] # N: number of nodes
S=data.shape[0] # S: number of files

# 2. Select the parameters
# Since we are using lasso, we will apply norm L1. Lambda fixes the sparsity of our results.
# Several values were considered but, after discussing with the lab, with lambda=1 we got the best.
lambda_=1 

L=1 # The selected lag is 1. The temporal resolution the data with 90 regions is sampled every 1s. 
# Thus, the data is relatively slow compared to neural activity, we should not go back a lot in time.

# 3. Data reshaping as explained in the report
X,y=getXYfromData(data,T,L,S,N)

M= (T-L)*S

# 4. Lasso regression 
# Note that for each n=0,...,N-1, a Lx(N-1) vector is obtained (we are not considering the self causality)
# We are interested in the mean over the lags

coefMatrix=np.zeros((N,N)) # Stores the mean coefficients from node i to j (i row, j column)
coefficients=np.zeros((L,N,N)) # Stores all the coefficients for all l=1,...,L
X_withoutNode=np.zeros((M,(N-1)*L)) # The matrix X contains all the nodes, but we are not interested in self causality

for n in range(0,N):
    X_withoutNode=np.concatenate((X[:,0:n*L],X[:,(n+1)*L:]),axis=1) # The matrix X contains all the nodes, but we are not interested in self causality
    coef,msee=elasticNet(X_withoutNode,y[n],0.499482018042742,0.209672573169101) #elastic_net(20, 0.001, X_withoutNode, y[n], 0.4,0.6, 1)# Lasso regression. Note that it returns the coefficients vector of size NxL
    matCoef=reshapeCoefficients(coef,N-1,L) #Reshape coefficients onto a matrix 
    index_nodes=np.concatenate((np.arange(0,n,+1),np.arange(n+1,N,+1))) # Put the results in the adjacency matrix.     
    coefficients[:,index_nodes,n]=matCoef.T  # we are computing the causality from other nodes to node n
    coefMatrix[index_nodes,n]=np.mean(matCoef.T ,axis=0)
saveInMatlab(coefficients,'coefficients_90_lag_1.mat') # For visualization with a tool given by the lab

# Visualize coefficients matrix
plotCoefficients(coefMatrix,N)

# Other results
# To get the rest of the plots in the report execute visualize_brain.mat in matlab