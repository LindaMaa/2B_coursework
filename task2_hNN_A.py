#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import task2_hNeuron

def task2_hNN_A(X):
    # Input:
    #  X : N-by-2 matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # calculated weights from 2.3, each as a 1x3 vector
    W1= np.asarray([1.0,0.83553068,-0.70874456])
    W2= np.asarray([-3.79464321, 0.48209769, 1.0])
    W3= np.asarray([-0.1264221, -1.0, 0.83776844])
    W4= np.asarray([4.53201229,-0.5291233, -1.0])
    W_second=np.asarray([-3.5,1,1,1,1])

    # matrix representing hidden layer
    Y_hidden = np.zeros((X.shape[0],4))

    # use hNeuron function to determine the values in the hidden layer using weights in 1st layer
    Y_hidden[:,0] = task2_hNeuron.task2_hNeuron(W1, X)
    Y_hidden[:,1] = task2_hNeuron.task2_hNeuron(W2, X)
    Y_hidden[:,2] = task2_hNeuron.task2_hNeuron(W3, X)
    Y_hidden[:,3] = task2_hNeuron.task2_hNeuron(W4, X)

    # compute the final output layer using weights in 2nd layer
    Y = task2_hNeuron.task2_hNeuron(W_second, Y_hidden)

    # Y is Nx1
    return Y

if __name__ == "__main__":
    X=np.asarray([[2.4,3.1],[1.9,3.0],[5,6]])#3x2 array
    Y=task2_hNN_A(X)
    print(Y)


