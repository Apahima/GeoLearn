#### https://github.com/ClementiGroup/S3D/blob/master/S%5E3D-single_traj.ipynb

from tqdm import tqdm
import numpy as np
# import mdtraj as md
import matplotlib.pyplot as plt
# import pyemma
# from pyemma import msm, plots    # pyemma APIs

import scipy
import scipy.sparse as sps

import time
import scipy.sparse.linalg as spl
import sklearn.neighbors as neigh_search
import sklearn.cluster as skl_cl
import sys


def kernel_neighbor_search(A, r, epsilon, sparse=False):
    """
    Analyzes one frame: uses a nearest neighbor algorithm to compute all distances up to cutoff r,
    generates the diffusion kernel sparse matrix

    Parameters:
        A:   nparray (m, 3), m number of heavy atoms
             array of coordinates of heavy atoms
        r:   scalar, cutoff
        epsilon: scalar, localscale

    Return:
        kernel: sparse matrix (m,m)
                diffusion kernel matrix (to be passed on to the SpaceTimeDMap subroutine)
    """

    # calling nearest neighbor search class
    kernel = neigh_search.radius_neighbors_graph(A, r, mode='distance')
    # computing the diffusion kernel value at the non zero matrix entries
    kernel.data = np.exp(-(kernel.data ** 2) / (epsilon))

    # diagonal needs to be added separately
    kernel = kernel + sps.identity(kernel.shape[0], format='csr')

    if sparse:
        return kernel
    else:
        return kernel.toarray()



def matrix_B(kernel, sparse = False):
    #Equation #13
    if sparse:
        m = kernel.shape[0]
        D = sps.csr_matrix.sum(kernel, axis=0) #Diagonal matrix
        Q = 1/D
        P_0 = Q @ kernel #P zero at time t matrix
        B = sps.diags((1/np.squeeze(np.asarray(P_0.sum(axis=0))))) @ P_0.T @ P_0 #B Matrix at time t equation 13

        # m = kernel.shape[0]
        # D = sps.csr_matrix.sum(kernel, axis=0) #Diagonal matrix
        # Q = sps.spdiags(1./D, 0, m, m) #Diagonal matrix inversion
        # P_0 = kernel * Q #P zero at time t matrix
        # B = (P_0*(sps.csr_matrix.transpose(P_0)))/(sps.csr_matrix.sum(P_0, axis=1)) #B matrix at timt t
    else:
        D = np.sum(kernel, axis = 1)
        P_0 = kernel*(1./D)
        B = (np.dot(P_0, P_0.T)/(np.sum(P_0, axis = 1))).T

    return B


def compute_SpaceTimeDMap(X, r, epsilon, sparse=False):
    """
    computes the SpaceTime DIffusion Map matrix out of the dataset available
    Parameters
    -------------------
    X: array T x m x 3
      array of T time slices x 3m features (m=number of atoms, features are xyz-coordinates of the atoms)
    r: scalar
      cutoff radius for diffusion kernel
    epsilon: scalar
      scale parameter

    Returns
    -------------------
    ll: np.darray(m)
      eigenvalues of the SpaceTime DMap
    u: ndarray(m,m)
      eigenvectors of the SpaceTime DMap. u[:,i] is the ith eigenvector corresponding to i-th eigenvalue
    SptDM: ndarray(m,m)
      SpaceTime Diffusion Matrix, eq (3.13) in the paper, time average of all the matrices in the cell list
    """


    m = np.shape(X)[1]
    T = np.shape(X)[0]

    SptDM = np.zeros((m, m))


    for i_t in tqdm(range(T)):
        if (i_t % 10 == 0):
            print('time slice ' + str(i_t))
        distance_kernel = kernel_neighbor_search(X[i_t, :, :], r, epsilon, sparse=sparse)
        SptDM += matrix_B(distance_kernel, sparse=sparse)


    SptDM /= T #Equation 19

    if sparse:
        ll, u = spl.eigs(SptDM, k=50, which='LR')
        ll, u = sort_by_norm(ll, u)
    else:
        ll, u = np.linalg.eig(SptDM)
        ll, u = sort_by_norm(ll, u)

    return ll, u, SptDM


def sort_by_norm(evals, evecs):
    """
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