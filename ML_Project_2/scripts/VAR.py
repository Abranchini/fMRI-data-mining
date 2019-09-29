# this file generates a TxN matrix at the end of the data generation and
# has the function to perform the tests using the VAR model

# general imports
import sklearn
import numpy as np
import matplotlib.pyplot as plt

# function imports
from data_functions import *
from model_functions import *

def generate_A(N):
    ''' Creates an adjacency matrix of size NxN
    to randomly define which nodes influence which.
    Inputs:
        integer number size of number of nodes
    Outputs:
        matrix NxN of 0s and 1s
    '''
    A_i_j = np.zeros((N, N)).astype(int)
    for i in range(N):
        for j in range(N):
            A_i_j[i][j]=np.random.binomial(1, 0.2)
    return A_i_j

def generate_C(L, N):
    '''For each time lag, we get a matrix that associates the N(quantity) nodes 
    between themselves, so node 1,2,...,S with all the other and itself regarding a random
    number sampled from an uniform distribution.
    Inputs:
        number of lags we will consider and number of nodes
    Output:
        L matrices NxN
    '''

    C=np.zeros((L,N,N)).astype(float)
    for l in range(L):
        for i in range(N):
            for j in range(N):
                C[l][i][j]=np.random.uniform(-0.8,0.8)
    return C

def generate_A_random_coefficient(P,L,N, feedback = False):
    '''Function that creates the final A matrices (that are as many as the lags we consider)
    Inputs:
        number of lags and numner of Nodes
    Output:
        L matrices NxN
    '''
    # same A matrix for every patient
    A = np.zeros((P,L,N,N)).astype(float)
    A_i_j=generate_A(N)
    np.fill_diagonal(A_i_j,0)

    for p in range(P):
        # different C matrix for every patient
        C = generate_C(L,N)
        for l in range(L):
            for i in range(N):
                for j in range(N):
                    A[p][l][i][j]=C[l][i][j]*A_i_j[i][j]                                                            


    # diagonal as 0
    for p in range(P):
        for l in range(L):
            np.fill_diagonal(A[p][l],0)

    
    return A, A_i_j

def gen_white_noise(L,T,N):
    C=np.zeros((L,T,N)).astype(float)

    for l in range(L):
        for t in range(T):
            for n in range(N):
                C[l][t][n]=np.random.uniform(-0.8,0.8)
    return C

def generateXdata_random(P,N,T,L, function_set = None):
    '''Generates data taking into account the lag
    Input:
        T = time points
        N = number of nodes
        P = number of patient files
        L = number of time lag we want to take into account
    Output:
        matrix TxN
    '''

    # create gen data vector
    Xmat = np.zeros((P,T,N))

    # generate the A matrices according to the experiment
    A, A_i_j = function_set(P,L,N)

    for p in range(P):
        # iterate all the time points
        # We can create the white noise matrix here
        for t in range(T):
            # generate gaussian noise matrix for every time
            # so every patient has different noise and
            # to make sure each point has G dist
            G_noise = gen_white_noise(L,1,N)

            new_row = 0
            # create the new row by adding the points from the rows before multiplied by the respective A lag matrix
            for l in range(1,L+1):
                new_row += np.dot(A[p][l-1],Xmat[p][t-l,:]) + G_noise[l-1][0]
            Xmat[p][t,:] = new_row

    return Xmat, A_i_j

def synthetic_test(P,N,T,L, number_of_tests, lambda_list ,model = None):
    """ run tests using artificial data
    Inputs :
        P = number of patient files
        N =  number of nodes
        T = time 
        L = lag
        model = model to use
        lambda_list = list of lambdas to test
        number_of_tests = test to perform in order for each lambda
    Outputs:
        Final_std = standard deviation of the tests
        Final_metrics = metrics regarding true positives/negatives and false positives/negatives
        Final_accuracies = mean accuracies of the tests
    """

    # list with accuracy values
    Final_std        = []
    Final_accuracies = []
    Final_metrics    = np.zeros((len(lambda_list), 4))
    
    
    final_met_counter = 0
    for l in lambda_list:
        temp_acc = np.zeros(number_of_tests)
        for test in range(number_of_tests):
            # Generate some data and an adjacency matrix
            C,A_i_j= generateXdata_random(P,N,T,L, function_set = generate_A_random_coefficient)

            # Extract dimensions from the file 
            T=C.shape[1] # T: number of time samples
            N=C.shape[2] # N: number of nodes
            P=C.shape[0] # P: number of files
            M = (T-L)*P
            # Data reshaping
            X,y=getXYfromData(C,T,L,P,N)

            # Model regression and mean coefficients computation
            # initialize needed matrices
            coefMatrix=np.zeros((N,N))
            coefficients=np.zeros((L,N,N))
            X_withoutNode=np.zeros((M,(N-1)*L))
            
            threshold = 0
            for n in range(0,N):
                # execute model

                X_withoutNode=np.concatenate((X[:,0:n*L],X[:,(n+1)*L:]),axis=1)
                coef,msee=elasticNet(X_withoutNode,y[n], 1-l, 1 - (1-l))#,0.499482018042742,0.209672573169101) #elastic_net(20, 0.001, X_withoutNode, y[n], 0.4,0.6, l)#lassoRegression(X_withoutNode,y[n],l)
                matCoef=reshapeCoefficients(coef,N-1,L) 

                # choose max of the different coefficient matrices
                index_nodes=np.concatenate((np.arange(0,n,+1),np.arange(n+1,N,+1)))       
                coefficients[:,n,index_nodes]=matCoef.T
                coefMatrix[index_nodes,n]=np.max(matCoef.T ,axis=0)

            # transform coef matrix into binary matrix
            idx_th=np.argwhere(coefMatrix > threshold)
            coefMatrix[idx_th[:,0],idx_th[:,1]]=1
            idx_th=np.argwhere(coefMatrix <= threshold)
            coefMatrix[idx_th[:,0],idx_th[:,1]]=0

            # calculate accuracy and add to list to do average
            #acc = sklearn.metrics.accuracy_score(A_i_j, Max_bi)
            acc = 1 - ( np.sum(np.abs(A_i_j - coefMatrix)) / (N*N) )
            temp_acc[test] = acc

            # get values for recall, precision, etc
            y_real = A_i_j.flatten()
            y_pred = coefMatrix.flatten()

            # # get metrics for model (true positives, false positives, ...)
            # try:
            #     tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_real,y_pred).ravel()
            # except:
            #     tn = sklearn.metrics.confusion_matrix(y_real,y_pred).ravel()
            #     fp = 0
            #     fn = 0
            #     tp = 0

        # append to final lists
        Final_std.append(temp_acc.std())
        Final_accuracies.append(temp_acc.mean())
        # Final_metrics[final_met_counter] = np.array([tn,fp,fn,tp])

        # increase counter
        final_met_counter +=1

    return Final_std, Final_metrics, Final_accuracies

def hypeOpt_elastic_synth(P,N,T,L, number_of_tests, lambda_1, lambda_2):
    """ Function that is called to optimize the hyperparameters
    of the elastic net function
    """

    test_list = []
    for test in range(number_of_tests):
        # Generate some data and an adjacency matrix
        C,A_i_j= generateXdata_random(P,N,T,L, function_set = generate_A_random_coefficient)

        # Extract dimensions from the file 
        T=C.shape[1] # T: number of time samples
        N=C.shape[2] # N: number of nodes
        P=C.shape[0] # P: number of files
        M = (T-L)*P
        # Data reshaping
        X,y=getXYfromData(C,T,L,P,N)

        # Model regression and mean coefficients computation
        # initialize needed matrices
        coefMatrix=np.zeros((N,N))
        coefficients=np.zeros((L,N,N))
        X_withoutNode=np.zeros((M,(N-1)*L))
        
        threshold = 0
        for n in range(0,N):
            # execute model
            X_withoutNode=np.concatenate((X[:,0:n*L],X[:,(n+1)*L:]),axis=1)
            coef,msee=elasticNet(X_withoutNode,y[n],lambda_1,lambda_2) #elastic_net(20, 0.001, X_withoutNode, y[n], 0.4,0.6, l)#lassoRegression(X_withoutNode,y[n],l)
            matCoef=reshapeCoefficients(coef,N-1,L) 

            # choose max of the different coefficient matrices
            index_nodes=np.concatenate((np.arange(0,n,+1),np.arange(n+1,N,+1)))       
            coefficients[:,n,index_nodes]=matCoef.T
            coefMatrix[index_nodes,n]=np.max(matCoef.T ,axis=0)

        # transform coef matrix into binary matrix
        idx_th=np.argwhere(coefMatrix > threshold)
        coefMatrix[idx_th[:,0],idx_th[:,1]]=1
        idx_th=np.argwhere(coefMatrix <= threshold)
        coefMatrix[idx_th[:,0],idx_th[:,1]]=0

        # calculate accuracy and add to list to do average
        acc = 1 - ( np.sum(np.abs(A_i_j - coefMatrix)) / (N*N) )

        test_list.append(acc)
    
    mean_acc = np.mean(test_list)
    return mean_acc


