#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io

def task1_3(Cov):
    # Input:
    # Cov : D-by-D covariance matrix (np.double)
    # Variables to save:
    # EVecs : D-by-D matrix of column vectors of eigenvectors (np.double)
    # EVals : D-by-1 vector of eigenvalues (np.double)
    # Cumvar : D-by-1 vector of cumulative variance (np.double)
    # MinDims : 4-by-1 vector (np.int32)

    # compute eigenvalues and eigenvectors of the covariance matrix
    EVals, EVecs = np.linalg.eig(Cov)

    # sort eigenvalues with respective eigenvectors in descending order
    index = EVals.argsort()[::-1]
    EVals = EVals[index]
    EVecs = EVecs[:, index]

    # if first element of an eigenvector is negative, multiply by -1
    for val in range(np.size(EVecs, 1)):
        if EVecs[0, val] < 0:
            EVecs[:, val] = (-1)*EVecs[:, val]

    # compute cumulative variance
    Cumvar = np.cumsum(EVals)

    #plot cumulative variance
    #xls = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    #plt.plot(xls, Cumvar,color = 'm')
    #plt.xticks(np.arange(0, 24, step=1))
    #plt.xlabel('Dimensions')
    #plt.ylabel('Cumvar')
    #plt.title('Cumulative Variance')
    

    # store minimum number of PCA dimensions
    MinDims = np.zeros(4)

    # compute minimum number of dimensions
    count = 0
    percentages = 100 * (Cumvar / Cumvar[-1])
    for val in range(len(percentages)):
        count=count+1
        if (percentages[val]>=70 and MinDims[0]==0):
            MinDims[0]=count
        elif (percentages[val]>=80 and MinDims[1]==0):
            MinDims[1]=count
        elif (percentages[val]>=90 and MinDims[2]==0):
            MinDims[2]=count
        elif (percentages[val]>=95 and MinDims[3]==0):
            MinDims[3]=count

    # save to files as required
    scipy.io.savemat('t1_EVecs.mat', mdict={'EVecs': EVecs})
    scipy.io.savemat('t1_EVals.mat', mdict={'EVals': EVals})
    scipy.io.savemat('t1_Cumvar.mat', mdict={'Cumvar': Cumvar})
    scipy.io.savemat('t1_MinDims.mat', mdict={'MinDims': MinDims})

if __name__ == "__main__":
    S = scipy.io.loadmat('t1_S.mat')
    task1_3(S['S'])
    
    












