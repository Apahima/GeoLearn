import sklearn.neighbors as Nei
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.spatial import distance_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import identity
from scipy.sparse import csr_matrix
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import KernelPCA

from sklearn import manifold, datasets


def AffiinityMAt(Z,epsilon=1,kernel_method='Gaussian', n_neighbor=None, Normalization=None):
    """
    Create an effinity map for data
    :param Z: Data
    :param epsilon: epsilon control for the gauusian kernel
    :param kernel_method: the method which calculate the distances between neighbours
    :param n_neighbor: number of neighbours for each node
    :param Normalization: If want to normalized the data, In case of normalization required, need to define and implement
    :return: returning the affinity map which reflected graph connectivity
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbor+1).fit(Z)
    distance, neigh_ind = neigh.kneighbors(Z) #Return matrix of neigbor indices for each vertex
    neigh_ind = neigh_ind[:,1:n_neighbor+1] #NEgelect the first neighbor

    M = np.zeros([Z.shape[0],Z.shape[0]])

    for j in range(Z.shape[0]):
        if kernel_method == 'Gaussian':
            M[j,neigh_ind[j]] = np.exp(-np.linalg.norm(Z[j,:]-Z[neigh_ind[j],:],axis=1)**2/epsilon)
        if kernel_method == 'Linear':
            M[j,neigh_ind[j]] = np.linalg.norm(Z[j,:]-Z[neigh_ind[j],:],axis=1)

    if Normalization is not None:
        print('Define normalization and implement in this line')
    return M, neigh_ind

#
# def DiffusionMap(A, n_dim, n_neighbors):
#     alpha = 1
#     epsilon = 1e-5
#     #Calculating the Walk matrix
#     Mg, _ = AffiinityMAt(A, kernel_method='Gaussian', n_neighbor=n_neighbors, Normalization=None)
#
#     Mg = Mg + epsilon #Avoid sigularity
#
#     #Calculating Degree Matrix
#     r = np.sum(Mg, axis=0)
#     Di = np.diag(r ** (-alpha))
#
#     P = Mg @ Di
#     D_right = np.diag((r)**0.5) #Normalization
#     D_left = np.diag((r)**-0.5) #Normalization
#
#     L = D_left @ P @ D_right #Normalization
#     # D = np.diag(r) #Degree Matrix
#     # Wg = np.linalg.inv(D) @ L
#
#     eigenValues, eigenVectors = linalg.eig(L)
#     idx = eigenValues.argsort()[::-1]
#     eigenValues = eigenValues[idx]
#     eigenVectors = eigenVectors[:, idx]
#
#
#     Diff_embbeded = np.real(eigenVectors) * np.real(eigenValues)
#
#     # kernel_pca_ = KernelPCA(n_components=n_dim, kernel="precomputed", eigen_solver='dense')
#     # Diff_embbeded = kernel_pca_.fit_transform(Wg)
#
#     return Diff_embbeded[:,1:n_dim+1]

def LLE(data, n_components=2, n_neighbors=10):
    """
    Dimensionality reduction with FastLLE algorithm
    Using Lagrange: https://www.stat.cmu.edu/~cshalizi/350/lectures/14/lecture-14.pdf
    :param data: input image matrix of shape (n,m)
    :param n_components: number of components for projection
    :param n_neighbors: number of neighbors for the weight extraction
    :return: Projected output of shape (n_components, n)
    """
    # Compute the nearest neighbors
    _, neighbors_idx  = AffiinityMAt(data,n_neighbor=n_neighbors,kernel_method='Linear')

    n = data.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        # Center the neighbors matrix
        k_indexes = neighbors_idx[i, :]
        neighbors = data[k_indexes, :] - data[i, :]

        # Compute the corresponding gram matrix
        gram_inv = np.linalg.pinv(np.dot(neighbors, neighbors.T))

        # Setting the weight values according to the lagrangian
        lambda_par = 2/np.sum(gram_inv)
        w[i, k_indexes] = lambda_par*np.sum(gram_inv, axis=1)/2
    m = np.subtract(np.eye(n), w)
    values, u = np.linalg.eigh(np.dot(np.transpose(m), m))
    return u[:, 1:n_components+1]

def Isomap(A, n_dim, n_neighbors):
    """
    Calculating data Isomap
    :param A: Data array
    :param n_dim: required dim for reduction
    :param n_neighbors: number of neigbour for the Affinity matrix
    :return: Isomap emmbeding to n_dim
    """
    Distmatrix, _ = AffiinityMAt(A, kernel_method='Linear', n_neighbor=n_neighbors) #Using linear kernel for getting distance matrix
    M = graph_shortest_path(Distmatrix,directed=False,method='D')

    H = np.identity(M.shape[0])-(1/M.shape[0])*np.ones_like(M)
    K = -0.5*H*(M**2)*H

    kernel_pca_ = KernelPCA(n_components=n_dim, kernel="precomputed", eigen_solver='dense')
    isomap_embbeded = kernel_pca_.fit_transform(K)

    return isomap_embbeded

def torus(R=10, r=4):
    """
    Creating torus coordinates in 3D
    :param R: Makor radius
    :param r: Minor radius
    :return: S array of 3D coordinates with 2000 random points
    """
    np.random.seed(42)
    # Creating random variables array XY
    N_pairs = 2000
    x = np.random.rand(N_pairs, 1)
    y = np.random.rand(N_pairs, 1)
    #Creating torus
    #Need to construct such that the Nxn wheres N is the number of samples

    S = np.concatenate([(R + r*np.cos(2*np.pi*y))*np.cos(2*np.pi*x),
                        (R + r*np.cos(2*np.pi*y))*np.sin(2*np.pi*x),
                            r*np.sin(2*np.pi*y)], axis=1)

    return S

def digitis(classes):
    #Loading digit
    digits = []
    for i in classes:
        digits.append(load_digits(n_class=i,return_X_y=True))

    return digits

def making_plot(data,pallete = None, neighbors: str = None, method: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=pallete, cmap=plt.cm.Spectral)
    legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    ax.add_artist(legend)
    ax.legend()

    plt.title('{} with {} Neighbours'.format(method,neighbors))


if __name__ == '__main__':
    ### ------ Loading \ Creating data ------###
    S = torus(R = 10,r = 4)
    dig = digitis((3,5,7))

    ### ------ Isomap ------###
    nei = [5,10,20]
    #Ploting Torus Isomapping
    for j in nei:
        Torus_isomap = Isomap(S,2,j)
        making_plot(Torus_isomap, pallete=S[:,0:1], neighbors=j, method='Torus ISOMAP')
        # making_plot(Torus_LLE,pallete=S[:,0:1],neighbors = j, method='Torus LLE')


    #Plotting Digits Isomapping
    for Specificcalss in dig:
        for j in nei:
            Digit_isomap = Isomap(Specificcalss[0],2,j)
            making_plot(Digit_isomap,Specificcalss[1],neighbors = j, method='Digit ISOMAP')

    ### ------ LLE ------###
    nei = [5,10,20]
    # Plotting Torus
    for j in nei:
        Torus_LLE = LLE(S,2,j)
        making_plot(Torus_LLE,pallete=S[:,0:1],neighbors = j, method='Torus LLE')

    #Plotting Digits
    for Specificcalss in dig:
        for j in nei:
            Digit_LLE = LLE(Specificcalss[0],2,j)
            making_plot(Digit_LLE,Specificcalss[1],neighbors = j, method='Digit LLE')


    ### --- Validation with build-in Sikit-learn package --- ###
    # n_points = 1000
    # X, color = datasets.make_s_curve(n_points, random_state=0)
    # n_neighbors = 10
    # n_components = 2
    # Swissroll_isomap = Isomap(X,n_components,n_neighbors)
    # making_isomap_plot(Swissroll_isomap,color, neighbors=n_neighbors)

    # n_points = 1000
    # X, color = datasets.make_s_curve(n_points, random_state=0)
    # n_neighbors = 10
    # n_components = 2
    # Swissroll_isomap = LLE(X,n_components,n_neighbors)
    # making_plot(Swissroll_isomap,color, neighbors=n_neighbors)

    # # ### ------ Diffusion map ------###
    # np.random.seed(42)
    # # Creating random variables array XY
    # N_pairs = 2000
    # x = np.random.rand(N_pairs, 1)
    # # DiffMap
    # DiffMapReduce = DiffusionMap(S,2,1999)
    # plt.figure()
    # plt.scatter(DiffMapReduce[:,0],DiffMapReduce[:,1], c=x)
    # #Plotting Digits
    # nei = [5,10,20]
    # for Specificcalss in dig:
    #     for j in nei:
    #         DiffMapReduce = DiffusionMap(Specificcalss[0],2,j)
    #         plt.figure()
    #         plt.scatter(DiffMapReduce[:, 0], DiffMapReduce[:, 1], c=x)
    plt.show()
    print('Finish')

