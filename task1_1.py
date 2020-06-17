#
# Version 0.9  (HS 09/03/2020)
import numpy as np
import scipy.io

def task1_1(X, Y):
    # Input:
    #  X : N-by-D data matrix (np.double)
    #  Y : N-by-1 label vector (np.int32)
    # Variables to save
    #  S : D-by-D covariance matrix (np.double) to save as 't1_S.mat'
    #  R : D-by-D correlation matrix (np.double) to save as 't1_R.mat'

    S = calculateCov(X)
    R = calculateCorr(X)

    #save the matrices as required
    scipy.io.savemat('t1_S.mat', mdict={'S': S})
    scipy.io.savemat('t1_R.mat', mdict={'R': R})

# helper function to calculate mean
def calculateMean(X):
    mean = np.sum(X, axis=0)/np.size(X,0)
    return mean

# calculate covariance matrix
def calculateCov(X):
    mean = calculateMean(X)
    covMatrix = X - np.tile(mean, (np.size(X,0),1))
    covMatrix = np.matmul(np.transpose(covMatrix), covMatrix) / (np.size(X,0))
    return covMatrix

# calculate correlation matrix
def calculateCorr(X):
    covMatrix = calculateCov(X)
    diag = covMatrix.diagonal().reshape(-1,1)
    corrM = np.divide(covMatrix, np.sqrt(np.matmul(diag, diag.T)))
    return corrM

if __name__ == "__main__":
    data = scipy.io.loadmat('dset.mat')
    # Y is redundant in the signature
    Y=0
    task1_1(data['X'],Y)