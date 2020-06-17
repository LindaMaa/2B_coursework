#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np

# step function
def stepFunct(x):
    return (x>0)+0

def task2_hNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # obtain shape N by D+1
    matrix_ones = np.ones((X.shape[0],1))
    augmented_X=np.concatenate((matrix_ones,X), axis=1) #create augmented matrix

    # compute dot product
    Y=[]
    for i in range(X.shape[0]):
        Y.append(np.dot(augmented_X[i],W.T))
    
    # apply step function
    for elem in range(len(Y)):
        Y[elem]=stepFunct(Y[elem])

    # return Y as np array
    Y = np.asarray(Y, dtype=np.double)
    return Y

if __name__ == "__main__":
    X=np.asarray([[10,3],[3,7],[2,3]])#3x2 array
    W = np.asarray([0,0,4]).T #3x1 a
    Y = task2_hNeuron(W,X)
    print(Y)