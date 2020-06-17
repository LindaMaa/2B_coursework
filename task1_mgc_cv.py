#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import task1_1

def task1_mgc_cv(X, Y, CovKind, epsilon, Kfolds):
    # Input:
    #  X : N-by-D matrix of feature vectors (np.double)
    #  Y : N-by-1 label vector (np.int32)
    #  CovKind : scalar (np.int32)
    #  epsilon : scalar (np.double)
    #  Kfolds  : scalar (np.int32)
    #
    # Variables to save
    #  PMap   : N-by-1 vector of partition numbers (np.int32)
    #  Ms     : C-by-D matrix of mean vectors (np.double)
    #  Covs   : C-by-D-by-D array of covariance matrices (np.double)
    #  CM     : C-by-C confusion matrix (np.double)

    # determine number of classes and samples in Y
    classes_Y, counts = np.unique(Y, return_counts=True)
    index = classes_Y.argsort()[::1]
    counts = counts[index]
    classes_Y = classes_Y[index]
    num_of_classes_Y = len(classes_Y)

    # prepare epsilon matrix
    epsilon_M = np.zeros((X.shape[1], X.shape[1]))
    np.fill_diagonal(epsilon_M, epsilon)

    # number of samples per class per partition
    Mc = (np.floor(counts / Kfolds)).astype(int)
    PMap = np.zeros(np.size(Y, 0)) # part numbers

    # indexing & sepatating classes
    index_Y = [(index, z) for index, z in enumerate(Y)]
    separation_Y = dict((x, None) for x in classes_Y)

    # classes ->  Kfold arrays of indices per partition
    for index, k in enumerate(separation_Y.keys()):
        separation_Y[k] = [j for (j, h) in index_Y if (h == int(k))]
        separation_Y[k] = [np.array(separation_Y[k][e:e + Mc[index]]) for e in range(0, len(separation_Y[k]), int(Mc[index]))]

        # concat the last array
        if (len(separation_Y[k]) > Kfolds):
            separation_Y[k][-2] = np.concatenate((separation_Y[k][-2], separation_Y[k][-1]))
            separation_Y[k].pop()
        separation_Y[k] = np.array(separation_Y[k])

    # populate PMap with partition numbers
    for f in range(1, Kfolds + 1):
        for k in separation_Y.keys():
            for index in separation_Y[k][f - 1]:
                PMap[index] = f

    PMap = PMap.astype(np.int32)

    partition_P, amnts = np.unique(PMap, return_counts=True)
    index = partition_P.argsort()[::1]
    amnts = amnts[index]

    # prior probability for partitions
    prior_probability = dict((u, []) for u in range(1, Kfolds + 1))

    for p in prior_probability.keys():
        for k in separation_Y.keys():
            prior_probability[p].append(len(separation_Y[k][p - 1]) / amnts[p - 1])

    # store training data
    partition2 = dict((i, []) for i in range(1, Kfolds + 1))

    # store indices of the training data
    for p in partition2.keys():
        for k in separation_Y.keys():
            index = np.array([x for i, x in enumerate(separation_Y[k]) if (i + 1 != p)])
            index = np.array([z for subset in index for z in subset])
            partition2[p].append((k, index))

    # mean and cov matrix for class and partition
    PMs_covariance = dict((i, []) for i in range(1, Kfolds + 1))
    
    for p in PMs_covariance.keys():
        for i in range(len(separation_Y.keys())):
            X_mean = X[partition2[p][i][1], :]
            mean = task1_1.calculateMean(X_mean)

            # full matrix/ shared matrix
            if (CovKind == 1 or CovKind == 3):
                coVar = task1_1.calculateCov(X_mean) + epsilon_M
                PMs_covariance[p].append((mean, coVar))

            # diagonal matrix
            elif (CovKind == 2):
                coVar = task1_1.calculateCov(X_mean)
                diagonal_M = np.diag(coVar)
                cov_diagonal_M = np.zeros(coVar.shape)
                np.fill_diagonal(cov_diagonal_M, diagonal_M)
                cov_diagonal_M = cov_diagonal_M + epsilon_M
                PMs_covariance[p].append((mean, cov_diagonal_M))

    # shared matrix
    if (CovKind == 3):
        for p in PMs_covariance.keys():
            covs = [coVar for (mean, coVar) in PMs_covariance[p]]
            shared_cov = (np.sum(covs, axis=0) / len(covs))

            # all classes in partition get diagonal matrix
            for i in range(len(PMs_covariance[p])):
                PMs_covariance[p][i] = (PMs_covariance[p][i][0], shared_cov)

    # separate means and covariance matrices
    PMs = dict((i, None) for i in range(1, Kfolds + 1))
    PCov = dict((j, None) for j in range(1, Kfolds + 1))

    for p in PMs.keys():
        PMs[p] = np.array([mean for (mean, coVar) in PMs_covariance[p]])
        PCov[p] = np.array([coVar for (mean, coVar) in PMs_covariance[p]])

    # confusion matrix for a partition
    P_confusion_matrix = dict((i, None) for i in range(1, Kfolds + 1))
    samples_amt = np.zeros(Kfolds, dtype=np.int32)

    # test samples for partition p
    for p in PMs.keys():
        index = np.where(PMap == p)[0]
        X_partition = X[index, :]
        samples_amt[p - 1] = len(index)  # number of test samples per partition
        
        # log posterior probability
        log_posterior_all = np.zeros((len(index), num_of_classes_Y))

        # caclulate for each class
        for i in range(len(prior_probability[p])):
            mean = PMs[p][i]
            coVar = PCov[p][i]
            log_posterior_probability = np.log(prior_probability[p][i])-0.5 * np.diag((X_partition - mean) @ np.linalg.inv(coVar) @ (X_partition - mean).T) - 0.5 * np.log(
                np.linalg.det(coVar))
            log_posterior_all[:, i] = log_posterior_probability

        # assign to the class with the highest log probability
        classification_P = np.argmax(log_posterior_all, axis=1)
        classification_P = classification_P +  1
        classification_P = classification_P.reshape(-1, 1)
        true_P = Y[index].reshape(-1, 1)
        
        # create confusion matrix
        confusion_matrix = np.zeros((num_of_classes_Y, num_of_classes_Y), dtype=np.int32)
        for i in range(len(classification_P)):
            confusion_matrix[true_P[i][0] - 1][classification_P[i][0] - 1] = confusion_matrix[true_P[i][0] - 1][classification_P[i][0] - 1] + 1
        P_confusion_matrix[p] = confusion_matrix[:]

    # save data
    scipy.io.savemat(f't1_mgc_{Kfolds}cv_PMap.mat', mdict={'PMap': PMap})

    for p in PMs.keys():
        Ms = PMs[p]
        Cov = PCov[p]
        CM = P_confusion_matrix[p]

        # save data
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_Ms.mat', mdict={'Ms': Ms})
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_ck{CovKind}_Covs.mat', mdict={'Covs': Cov})
        scipy.io.savemat(f't1_mgc_{Kfolds}cv{p}_ck{CovKind}_CM.mat', mdict={'CM': CM})

    # confusion matrix final calculation
    CM = 0
    for k in P_confusion_matrix.keys():
        CM = CM + (P_confusion_matrix[k] / samples_amt[k - 1])
    CM = CM / Kfolds
    L = Kfolds + 1
    scipy.io.savemat(f't1_mgc_{Kfolds}cv{L}_ck{CovKind}_CM.mat', mdict={'CM': CM})
    return CM


if __name__ == "__main__":

    dset = scipy.io.loadmat('dset.mat')
    Y_species = dset['Y_species'][:]
    X = dset['X'][:]
    
    accuracy1 = np.sum(np.diag(task1_mgc_cv(X, Y_species, CovKind=1, epsilon=0.01, Kfolds=5)))
    print('Accuracy for CovKind=1:')
    print(accuracy1)
    print("\n")
    
    accuracy2 = np.sum(np.diag(task1_mgc_cv(X, Y_species, CovKind=2, epsilon=0.01, Kfolds=5)))
    print('Accuracy for CovKind=2:')
    print(accuracy2)
    print("\n")
    accuracy3 = np.sum(np.diag(task1_mgc_cv(X, Y_species, CovKind=3, epsilon=0.01, Kfolds=5)))
    print('Accuracy for CovKind=3:')
    print(accuracy3)
    print("\n")


    


