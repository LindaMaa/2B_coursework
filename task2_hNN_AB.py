#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import task2_hNeuron

def task2_hNN_AB(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # calculated weights for A
    W1 = np.asarray([1.0, 0.83553068, -0.70874456])
    W2 = np.asarray([-3.79464321, 0.48209769, 1.0])
    W3 = np.asarray([-0.1264221, -1.0, 0.83776844])
    W4 = np.asarray([4.53201229, -0.5291233, -1.0])
    W_second_A = np.asarray([-3.5, 1, 1, 1, 1,0,0,0,0])

    # calculated weights for B
    W5 = np.asarray([6.68678821,-0.76129168,-1.0])
    W6 = np.asarray([-2.04559391,0.12041093,1.0])
    W7 = np.asarray([-1.0, -0.3568456, 0.94068988])
    W8 = np.asarray([1.0, 0.73119649, -0.3848288])
    W_second_B = np.asarray([-9, 0, 0, 0, 0, 4, 2, 2, 4])


    # matrix representing hidden layer
    Y_hidden = np.zeros((X.shape[0], 8))

    # use hNeuron function to determine the values in the hidden layer using weights in 1st layer
    Y_hidden[:, 0] = task2_hNeuron.task2_hNeuron(W1, X)
    Y_hidden[:, 1] = task2_hNeuron.task2_hNeuron(W2, X)
    Y_hidden[:, 2] = task2_hNeuron.task2_hNeuron(W3, X)
    Y_hidden[:, 3] = task2_hNeuron.task2_hNeuron(W4, X)
    Y_hidden[:, 4] = task2_hNeuron.task2_hNeuron(W5, X)
    Y_hidden[:, 5] = task2_hNeuron.task2_hNeuron(W6, X)
    Y_hidden[:, 6] = task2_hNeuron.task2_hNeuron(W7, X)
    Y_hidden[:, 7] = task2_hNeuron.task2_hNeuron(W8, X)

    Y1 = task2_hNeuron.task2_hNeuron(W_second_A,Y_hidden)
    Y1 = np.asarray(Y1)
    Y1 = Y1.reshape(-1,1)
    Y2 = task2_hNeuron.task2_hNeuron(W_second_B,Y_hidden)
    Y2 = np.asarray(Y2)
    Y2 = Y2.reshape(-1,1)

    # compute the final output layer
    Y = task2_hNeuron.task2_hNeuron(np.asarray([-1,-2,2]).T,np.concatenate((Y1, Y2), axis=1))

    return Y

if __name__ == "__main__":
    X=np.asarray([[2.4,3.1],[3,4],[2.5,10]]) #3x2 array
    Y=task2_hNN_AB(X)
    print(Y)
