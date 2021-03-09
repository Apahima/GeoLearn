#### Reference
#### https://github.com/ClementiGroup/S3D/blob/master/S%5E3D-single_traj.ipynb

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sps
import time
import scipy.sparse.linalg as spl
import sklearn.neighbors as neigh_search
import sys


def kernel_neighbor_search(A, r, epsilon, sparse=False, Mdata = False):
    """
    Calculating the Affinity matrix both for full dataset or missing dataset Affinity matrix
    :param A: Dataset - Full data or Missing Data
    :param r: The neigbour radius
    :param epsilon: the normalization parameter for the neaibours kernel calculation - control the diffusion rate in practice
    :param sparse: Whether or not using sparse data
    :param Mdata: Whether or not using Missing data flag
    :return: Return the Affinity matrix for provided data
    """
    # calling nearest neighbor search class
    kernel = neigh_search.radius_neighbors_graph(A, np.sqrt(r*epsilon), mode='distance')
    # computing the diffusion kernel value at the non zero matrix entries
    kernel.data = np.exp(-(kernel.data ** 2) / (epsilon))

    if Mdata: #Using for missing data section
        np.random.seed(32)
        idx_NaN = np.random.randint(0, 500, [int(0.8 * 500)])
        kernel.toarray()[idx_NaN, :] = 0


    # diagonal needs to be added separately
    kernel = kernel + sps.identity(kernel.shape[0], format='csr')

    if sparse:
        return kernel
    else:
        return kernel.toarray()



def matrix_B(kernel, sparse = False):
    """
    Calculating the B matrix per paper
    :param kernel: the Affinity matrix
    :param sparse: Whether or not using sparse data
    :return: Return the B matrix per paper
    """
    #Equation #13
    if sparse:
        D = np.array(kernel.sum(axis=1)) #Diagonal matrix
        Q = sps.diags(1/np.squeeze(D))
        P_0 = Q @ kernel #P zero at time t matrix
        D_0 = np.squeeze(np.array(1 / P_0.sum(axis=0)))
        B = sps.diags(D_0) @ P_0.T @ P_0 #B Matrix at time t equation 13

    else:
        D = np.sum(kernel, axis = 1)
        P_0 = kernel*(1./D)
        B = (np.dot(P_0, P_0.T)/(np.sum(P_0, axis = 1))).T

    return B


def compute_SpaceTimeDMap(X, r, epsilon, k=20, sparse=False, Mdata = False):
    """
    Compute the space time diffusion matrix from X dataset

    :param X: The dataset which is T x m x 2 array contain the time channel and the trajectory
    :param r: the radius for nearset-neighbour Affinity matrix
    :param epsilon: the normalized parameter for the nearest-neighbour kernel
    :param k: Number of eigenvector decomposition for spectral clustering
    :param sparse: Whether or not using sparse data, the defualt is sparse=True
    :param Mdata: Flag for calculating Missing data section
    :return:
    Return the eigenvalue and eigenfunction for the Q and the normalized laplacian matrix
    """
    m = np.shape(X)[1]
    T = np.shape(X)[0]

    SptDM = np.zeros((m, m))

    for i_t in tqdm(range(T)):
        if (i_t % 10 == 0):
            print('time slice ' + str(i_t))
        distance_kernel = kernel_neighbor_search(X[i_t, :, :], r, epsilon, sparse=sparse, Mdata = Mdata)
        SptDM = SptDM + matrix_B(distance_kernel, sparse=sparse)

    SptDM = SptDM / T #Equation 19

    L_SptDM  = (1/epsilon) * (SptDM - np.identity(m)) #Computing L matrix

    if sparse:
        ll, u = spl.eigs(SptDM, k=k, which='LR')
        ll, u = sort_by_norm(ll, u)
        ll_l, u_l = spl.eigs(L_SptDM, k=k, which='LR')
        ll_l, u_l = sort_by_norm(ll_l, u_l)

    return np.real(ll), np.real(u), np.real(ll_l[::-1]),np.real(u_l[::-1]), SptDM


def sort_by_norm(evals, evecs):
    """
    Reference
    https://github.com/ClementiGroup/S3D/blob/master/S%5E3D-single_traj.ipynb
    Sorts the eigenvalues and eigenvectors by descending norm of the eigenvalues
    Parameters
    ----------
    evals: ndarray(n)
        eigenvalues
    evecs: ndarray(n,n)
        eigenvectors in a column matrix
    Returns
    -------
    (evals, evecs) : ndarray(m), ndarray(n,m)
        the sorted eigenvalues and eigenvectors
    """
    # norms
    evnorms = np.abs(evals)
    # sort
    I = np.argsort(evnorms)[::-1]
    # permute
    evals2 = evals[I]
    evecs2 = evecs[:, I]
    # done
    return (evals2, evecs2)