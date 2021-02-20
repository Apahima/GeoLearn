#### https://github.com/ClementiGroup/S3D/blob/master/S%5E3D-single_traj.ipynb


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

    if sparse:
        m = kernel.shape[0]
        D = sps.csr_matrix.sum(kernel, axis=0)
        Q = sps.spdiags(1./D, 0, m, m)
        S = kernel * Q
        B = (S*(sps.csr_matrix.transpose(S)))/(sps.csr_matrix.sum(S, axis=1))
    else:
        D = np.sum(kernel, axis = 1)
        S = kernel*(1./D)
        B = (np.dot(S, S.T)/(np.sum(S, axis = 1))).T

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

    # initialize the Spacetime Diffusion Map matrix
    # that will be averaged over the different timeslices
    # and the over the different trajectories
    m = np.shape(X)[1]
    T = np.shape(X)[0]

    # SptDM =  sps.csr_matrix((m, m))
    SptDM = np.zeros((m, m))

    # loop over trajectory

    for i_t in range(T):
        if (i_t % 10 == 0):
            print('time slice ' + str(i_t))
        # selecting the heavy atoms coordinates in the timeslice s
        # compute diffusion kernel using data at timeslice s
        distance_kernel = kernel_neighbor_search(X[i_t, :, :], r, epsilon, sparse=sparse)
        SptDM += matrix_B(distance_kernel, sparse=sparse)

    # divide by the total number of timeslices considered
    # this define the Q operator
    SptDM /= T

    # Computing eigenvalues and eigenvectors of the SpaceTime DMap
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