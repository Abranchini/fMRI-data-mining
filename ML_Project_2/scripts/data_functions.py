# This file contains the functions related with 
# the data generation, obtention, reshaping...
import numpy as np
import os
import scipy.io as scp
from sklearn.metrics.pairwise import euclidean_distances

def obtainData(): # First DB. Finally not used since it was too noisy.
    # Gets the data from the matlab files
    # Returns an array with the data of all the files
    # wiht shape S (number of files)xT(number of time samples)xN(number of nodes).
    directory = '../TCS_Glasser'
    data=[]
    for file in os.listdir(directory):
        filename = file
        if filename.endswith(".mat"): 
            f = scp.loadmat(directory+'/'+filename)
            d = f.get('TCS') 
            d = np.array(d)
            data.append(d.T)
    return np.asarray(data)


def obtainData2(): 
    # Gets the data from the matlab files from the parcellated data with 90 regions
    # Returns an array with shape S (number of files)xT(number of time samples)xN(number of nodes).
    data=[]
    #directory='../data/'
    directory='/media/abranches/Main/University/EPFL/Project_MPI/ML_Project_2/data'#'D:/University/EPFL/Project_MPI/ML_Project_2/data'
    for file in os.listdir(directory):
        filename = file
        if filename.startswith("Activity_avr"): 
            f = scp.loadmat(directory+'/'+filename)
            d = f.get('Activity_avr') 
            d = np.array(d)
            data.append((100*d).T)
    return np.asarray(data)

def obtainICAPs(): 
    # Gets the data from the matlab files from the parcellated data with 90 regions
    # Returns an array with shape S (number of files)xT(number of time samples)xN(number of nodes).
    data=[]
    directory='../data/'
    for file in os.listdir(directory):
        filename = file
        if filename.startswith("IC_AAL"): 
            f = scp.loadmat(directory+'/'+filename)
            d = f.get('IC_AAL') 
            d = np.array(d)
            data.append(d)
    return np.asarray(data)

def getNodesPosition():
    # Returns the node positions
    nodesPositions=[]
    filename = '../data/nodesCenters.mat'
    f = scp.loadmat(filename)
    nodesPositions=f.get('centers')
    return nodesPositions

def getKneighbors(positions,N,k):
    # computes the euclidean distances between the nodes
    # Stores the computed distances in a NxN matrices, being N the number of nodes
    distances=euclidean_distances(positions,positions) #This matrix contains the distances with all the nodes
    neighbors=np.zeros((N,N))
    for node in range(0,N):
        node_dist=distances[node,:]
        idx = np.argsort(node_dist)[:k+1]
        closest=idx[1:k+1]
        neighbors[node,closest]=1
    return neighbors
    '''    minimums=node_dist[selected_idx]
        node_mat[node,selected_idx]=minimums
        node_mat[selected_idx,node]=minimums
        adjacencies.append(node_mat)
    return adjacencies'''

def randomDataGeneration_noSpace(S,T,N):
    # Function for testing purposes. 
    #T: total number of time steps
    #L: number of time samples used for prediction. 5 POINTS: 3.5s
    #N: number of nodes where we need to predict
    #S: number of files
    xx=np.zeros((S,T,N))
    for n in range(0,N):
        for s in range(0,S):
            for t in range(0,T):
                xx[s,t,n]=((t+1)*(10**(s+n*S)))
    return xx

def getXYfromData(data_total,T,L,S,N): 
    # this function reshapes de data in order to have the shape desired for lasso regression as described in the paper. 
    # the y vector contains [x_{L,s0},x_{L+1,s0},...x_{T,s0},x_{L,s1},...,x_{T-1,s1},...,x_{L,s(S-1)},...x_{T-1,s(S-1)}] 
    # As we consider multiple Ns, we will have N y_vectors. However, the matrix X is be common for all of them

    # X_n contains M=(T-L)*S rows, L columns as follows:
    #[[x_{L-1,s0},x_{L-2,s0},x_{L-3,s0}..x_{L-1-(L-1),s0}],   (time L, node s0)
    #[x_{L,s0},x_{L-1,s0},x_{L-2,s0}..x_{L-(L-1),s0}],   (time L+1, node s0)
    # [x_{L+1,s0},x_{L+1-1,s0},x_{L+1-2,s0}..x_{L+1-(L-1),s0}],   (time L+1, file s0)
    # ...
    # [x_{(T-1-1),s0},x_{T-1-2,s0},x_{T-1-3,s0}..x_{(T-1)-(L-1),s0}]            (time T-1, file s0)
    #[[x_{L-1,s1},x_{L-2,s1},x_{L-3,s1}..x_{(L-1)-(L-1),s1}], 
    # [x_{L,s1},x_{L-1,s1},x_{L-2,s1}..x_{(L-1)-(L-1),s1}],   (time L, node s1)
    # [x_{L+1,s1},x_{L+1-1,s1},x_{L+1-2,s1}..x_{L+1-(L-1),s1}],   (time L+1, file s1)
    # ...
    # [x_{L+1-1,s(S-1)},x_{L+1-2,s(S-1)},x_{L+1-3,s(S-1)}..x_{L+1-(L-1),s(S-1)}],   (time L, file s(S-1)) (last file)
    # [x_{L+2-1,s(S-1)},x_{L+2-2,s(S-1)},x_{L+2-3,s(S-1)}..x_{L+2-(L-1),s(S-1)}],
    # ...
    # [x_{(T-1)-1,s(S-1)},x_{(T-1)-2,s(S-1)},x_{(T-1)-3,s(S-1)}..x_{(T-1)-(L-1),s(S-1)}]            (time T, file s(S-1)) 

    # The resulting X matrix is the result of concatenating all X_n for all N, (T-L)*S rows, L*N columns
    # The concatenation is done in columns
    y=[]
    M= (T-L)*S #number of rows of X
    X=np.zeros((M, L*N))
    for n in range(0,N):
        data=data_total[:,:,n]
        # Take the last T-L samples for each node n, for all the files
        Ymat=data[:,L:T]
        # Reshape
        # [x_{L,s0},x_{L+1,s0},...x_{T-1,s0},x_{L,s1},...,x_{T-1,s1},...,x_{L,s(S-1)},...x_{T-1,s(S-1)}] 
        y_n=Ymat.flatten()

        # Concatenate the vectors for the different nodes
        y.append(y_n)

        # Create Xmatrix as explained above
        X_n=np.zeros((M, L))
        for s in range(0,S):
            for l in range(0,L):
                for tt in range (L,T):
                    row=(tt-(L-1))+s*(T-L)-1
                    X_n[row][l]=data[s,tt-(l+1)]
        # Concatenate the matrices
        X[:,(L*n):(L*n+L)]=X_n
    return (X,y)

def saveInMatlab(A,name):
    scp.savemat(name,mdict={'A':A})

def getXYfromData_Space(data_total,T,L,S,N,K): 
    # The resulting X matrix is the result of concatenating all X_n for all N, considering K neighbors, (T-L)*S rows, L*N*k columns
    # The difference with the other is that here we are taking the k closest neighbors
    # The concatenation is done in columns
    y=[]
    M= (T-L)*S*(K+1) #number of rows of X
 
    X=np.zeros((M, L*N*(K+1)))

    # 2. Get neigbors and create groups
    centers=getNodesPosition()
    neigbors=getKneighbors(centers,N,4)
    np.fill_diagonal(neigbors,1)
    for n in range(0,N):

        data_node=data_total[:,:,n]    
        # Take the last T-L samples for each node n, for all the files
        Ymat=data_node[:,L:T]
        # Reshape
        # [x_{L,s0},x_{L+1,s0},...x_{T-1,s0},x_{L,s1},...,x_{T-1,s1},...,x_{L,s(S-1)},...x_{T-1,s(S-1)}] 
        y_n=Ymat.flatten()
        # Concatenate the vectors for the different nodes
        y_n_mat=np.zeros(((T-L)*(K+1)*S,(K+1)))
        for i,yy in enumerate(y_n):
            y_n_mat[i*(K+1):(i+1)*(K+1),:]=yy*np.eye(K+1)
        y.append(y_n_mat)

        closest_n=np.where(neigbors[n,:] == 1)[0]
        # Create Xmatrix as explained above
        X_n=np.zeros((M, L))
        for s in range(0,S):
            for l in range(0,L):
                for tt in range (L,T):
                    row=((tt-(L-1))+s*(T-L)-1)*K
                    for k in range(0,K+1):
                        data_k=data_total[:,:,closest_n[k]]
                        X_n[row+k][l]=data_k[s,tt-(l+1)]

        # Concatenate the matrices
        X[:,(L*n):(L*n+L)]=X_n
    return (X,y)




               


