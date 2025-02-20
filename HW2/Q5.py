import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


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
    :return: Projected output of shape (n_components, n) exluding the first element since it's a bias
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
    Refrence: https://en.wikipedia.org/wiki/Isomap
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
    x = np.random.uniform(0,1,(N_pairs,1))
    y = np.random.uniform(0,1,(N_pairs,1))
    #Creating torus
    #Need to construct such that the Nxn wheres N is the number of samples

    S = np.concatenate([(R + r*np.cos(2*np.pi*y))*np.cos(2*np.pi*x),
                        (R + r*np.cos(2*np.pi*y))*np.sin(2*np.pi*x),
                            r*np.sin(2*np.pi*y)], axis=1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(-15, 15)
    ax1.scatter(S[:,0],S[:,1],S[:,2], c=np.linalg.norm((S[:,0],S[:,1]),axis=0))
    plt.savefig('Torus R={},r={}'.format(R,r))
    return S

def digitis(classes):
    """
    Create Buld-in Digitis dataset
    :param classes: which digitis dataset to create
    :return: List of digitis dataset
    """
    #Loading digit
    digits = []
    for i in classes:
        digits.append(load_digits(n_class=i,return_X_y=True))

    return digits

def making_plot(data,pallete = None, neighbors: str = None, method: str = None):
    """
    Making the plots
    :param data: data to plot
    :param pallete: color map
    :param neighbors: how many neighbours computes for the Affinity map
    :param method: which method used to embbed the data
    :return: Plot the graph in 2D dimentions
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=pallete, cmap=plt.cm.Spectral)
    legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    ax.add_artist(legend)
    ax.legend()

    plt.title('{} with {} Neighbours'.format(method,neighbors))


def CreateDS_Torus_Digits(TurosR=10,Torusr=4,Classes=[3,5,7]):
    """
    Create Torus and Digits Dataset
    :param TurosR: Major Radius
    :param Torusr: Minor radius
    :param Classes: Number of classes from dataset
    :return: Datasets
    """
    S = torus(R = 10,r = 4)
    Classes = [3,5,7]
    dig = digitis(Classes)

    return S, dig

def ISOMAPEmbbeding(TurosR=10,Torusr=4,Classes=[3,5,7],nei=[5,10,20], DataSet = {'Turos', 'Digits'}):
    """
    Embbeding ISOMAP for Turos and Digits datasets
    :param TurosR: Major Radius
    :param Torusr: Minor radius
    :param Classes: Number of classes from dataset
    :param nei: Number of neighbours for calculating the embbeding space
    :param DataSet: Which Dataset to plot
    :return: Plotting the relevant dataset
    """

    S, dig = CreateDS_Torus_Digits(TurosR=TurosR,Torusr=Torusr,Classes=[3,5,7])
    ### ------ Isomap ------###
    nei = nei

    if 'Turos' in DataSet:
        # Ploting Torus Isomapping
        fig = plt.figure(figsize=(30, 10))
        for i, j in enumerate(nei):
            Torus_isomap = Isomap(S, 2, j)
            neighbors = j
            method = 'Torus ISOMAP'
            ax = fig.add_subplot(1, len(nei), i + 1)
            scatter = ax.scatter(Torus_isomap[:, 0], Torus_isomap[:, 1], c=S[:, 0:1], cmap=plt.cm.Spectral)
            # legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
            # ax.add_artist(legend)
            # ax.legend()
            ax.set_title('{} with {} Neighbours'.format(method, neighbors))
            # making_plot(Torus_isomap, pallete=S[:, 0:1], neighbors=j, method='Torus ISOMAP') #An option to plot single graphs
        plt.savefig('Torus ISOMAP embbeding for {} neighbour'.format(nei))

    if 'Digits' in DataSet:
        # Plotting Digits Isomapping
        for Argclass, Specificcalss in enumerate(dig):
            fig = plt.figure(figsize=(30, 10))
            for i, j in enumerate(nei):
                neighbors = j
                Digit_isomap = Isomap(Specificcalss[0], 2, j)
                method = 'Digit ISOMAP'
                ax = fig.add_subplot(1, len(nei), i + 1)
                scatter = ax.scatter(Digit_isomap[:, 0], Digit_isomap[:, 1], c=Specificcalss[1], cmap=plt.cm.Spectral)
                legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
                ax.add_artist(legend)
                ax.legend()
                ax.set_title('{} with {} Neighbours'.format(method, neighbors))
                # making_plot(Digit_isomap, Specificcalss[1], neighbors=j, method='Digit ISOMAP') #An option to plot single graphs
            plt.savefig('Digits up to {} -  ISOMAP embbeding for {} neighbour'.format(Classes[Argclass], nei))

def LLEEmbedding(TurosR=10,Torusr=4,Classes=[3,5,7],nei=[5,10,20], DataSet = {'Turos', 'Digits'}):
    """
    Embbeding LLE for Turos and Digits datasets
    :param TurosR: Major Radius
    :param Torusr: Minor radius
    :param Classes: Number of classes from dataset
    :param nei: Number of neighbours for calculating the embbeding space
    :param DataSet: Which Dataset to plot
    :return: Plotting the relevant dataset
    """

    S, dig = CreateDS_Torus_Digits(TurosR=TurosR,Torusr=Torusr,Classes=[3,5,7])
    ### ------ LLE ------###
    nei = nei

    if 'Turos' in DataSet:
        # Plotting Torus
        fig = plt.figure(figsize=(30, 10))
        for i,j in enumerate(nei):
            Torus_LLE = LLE(S,2,j)
            neighbors = j
            method = 'Torus LLE'
            ax = fig.add_subplot(1, len(nei), i + 1)
            scatter = ax.scatter(Torus_LLE[:, 0], Torus_LLE[:, 1], c=S[:, 0:1], cmap=plt.cm.Spectral)
            # legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
            # ax.add_artist(legend)
            # ax.legend()
            ax.set_title('{} with {} Neighbours'.format(method, neighbors))
            # making_plot(Torus_LLE, pallete=S[:, 0:1], neighbors=j, method='Torus LLE') #An option to plot single graphs
        plt.savefig('Torus LLE embbeding for {} neighbour'.format(neighbors))

    if 'Digits' in DataSet:
        #Plotting Digits
        for Argclass, Specificcalss in enumerate(dig):
            fig = plt.figure(figsize=(30,10))
            for i,j in enumerate(nei):
                neighbors = j
                Digit_LLE = LLE(Specificcalss[0],2,j)
                method = 'Digit LLE'
                ax = fig.add_subplot(1, len(nei), i + 1)
                scatter = ax.scatter(Digit_LLE[:, 0], Digit_LLE[:, 1], c=Specificcalss[1], cmap=plt.cm.Spectral)
                legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
                ax.add_artist(legend)
                ax.legend()
                ax.set_title('{} with {} Neighbours'.format(method, neighbors))
                # making_plot(Digit_isomap, Specificcalss[1], neighbors=j, method='Digit ISOMAP') #An option to plot single graphs
            plt.savefig('Digits up to {} - LLE embbeding for {} neighbour'.format(Classes[Argclass], nei))

def SwissRollWithConstrain(nei = [5,25,50]):
    """
    Implement Swiss roll with Random noise and implement ISOMMAPING and LLE
    :param nei: Number of neighbours to calculate the manifold learning embbeding
    :return: Ploting the graph in the embbeding space
    """
    n_samples = 4000
    n_neighbor = 60
    noise = 0
    X, _ = make_swiss_roll(n_samples, noise=noise, random_state=42)
    X = X*2 #scaling ths Swiss

    neigh = NearestNeighbors(n_neighbors=n_neighbor).fit(X)
    _, indxes = neigh.kneighbors(X)

    SwissConstrain = np.delete(X,indxes[1500,:], axis=0)
    SwissConstrainNoisy = SwissConstrain + np.random.normal(0,1,[n_samples-n_neighbor,3])

    elevation = 10
    azimoth = 60
    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_zlim(-30, 30)
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.linalg.norm((X[:, 0], X[:, 1]), axis=0))
    ax1.set_title('Swiss Roll')
    ax1.view_init(elev=elevation, azim=azimoth)
    ax1 = fig.add_subplot(132, projection='3d')
    ax1.set_zlim(-30, 30)
    ax1.scatter(SwissConstrain[:, 0], SwissConstrain[:, 1], SwissConstrain[:, 2],
                c=np.linalg.norm((SwissConstrain[:, 0], SwissConstrain[:, 1]), axis=0))
    ax1.set_title('Swiss Roll with constrain')
    ax1.view_init(elev=elevation, azim=azimoth)
    ax1 = fig.add_subplot(133, projection='3d')
    ax1.set_zlim(-30, 30)
    ax1.scatter(SwissConstrainNoisy[:, 0], SwissConstrainNoisy[:, 1], SwissConstrainNoisy[:, 2],
                c=np.linalg.norm((SwissConstrainNoisy[:, 0], SwissConstrainNoisy[:, 1]), axis=0))
    ax1.set_title('Noisy Swiss Roll with constrain')
    ax1.view_init(elev=elevation, azim=azimoth)
    plt.savefig('Swiss Roll with different petubations')

    DataToPlot = [X,SwissConstrain,SwissConstrainNoisy]
    DataName = ['Swiss ISOMAP','Swiss with constrain ISOMAP', 'Swiss with constrain and noise ISOMAP']

    # Ploting Swiss Isomapping
    for neighbors in nei:
        fig = plt.figure(figsize=(30, 10))
        for i, j in enumerate(DataToPlot):
            Swiss_isomap = Isomap(j, 2, neighbors)
            method = DataName[i]
            ax = fig.add_subplot(1, len(DataToPlot), i + 1)
            ax.scatter(Swiss_isomap[:, 0], Swiss_isomap[:, 1],
                       c=np.linalg.norm((Swiss_isomap[:, 0], Swiss_isomap[:, 1]), axis=0), cmap=plt.cm.Spectral)
            ax.set_title('{} with {} Neighbours'.format(method, neighbors))
            # making_plot(Swiss_isomap, pallete=Swiss_isomap[:, 0:1], neighbors=neighbors, method=method) #An option to plot single graphs
        plt.savefig('Swiss ISOMAP embbeding for {} neighbour'.format(neighbors))

    DataName = ['Swiss LLE', 'Swiss with constrain LLE', 'Swiss with constrain and noise LLE']
    # Ploting Swiss LLE
    for neighbors in nei:
        fig = plt.figure(figsize=(30, 10))
        for i, j in enumerate(DataToPlot):
            Swiss_LLE = LLE(j, 2, neighbors)
            method = DataName[i]
            ax = fig.add_subplot(1, len(DataToPlot), i + 1)
            ax.scatter(Swiss_LLE[:, 0], Swiss_LLE[:, 1],
                       c=np.linalg.norm((Swiss_LLE[:, 0], Swiss_LLE[:, 1]), axis=0), cmap=plt.cm.Spectral)
            ax.set_title('{} with {} Neighbours'.format(method, neighbors))
            # making_plot(Swiss_LLE, pallete=Swiss_LLE[:, 0:1], neighbors=neighbors, method=method) #An option to plot single graphs
        plt.savefig('Swiss LLE embbeding for {} neighbour'.format(neighbors))
    return

def ClassificationAccuracy(Class=[7], folds = 20, neighbours = 25):
    """
    Classified the embbeding space and printing the accuracy
    :param Classes: Which digit class to extract
    :param folds: How many folds to make for classification in order to get high confidence
    :param neighbours: Number of neighbours to take into cosideration for manifold learning embbeding
    :return: printing the mean accuracy for each embbeding method
    """
    S, dig = CreateDS_Torus_Digits(Class)
    i = 0

    Digit_isomap_embed = Isomap(dig[i][0], 2, neighbours)
    Digit_LLE_embed = LLE(dig[i][0], 2, neighbours)
    clf = SVC()
    Accuracies_ISOMAP = cross_val_score(clf,Digit_isomap_embed,dig[i][1], cv=folds)
    Accuracies_LLE = cross_val_score(clf,Digit_LLE_embed,dig[i][1], cv=folds)

    print('The mean accuracy is:')
    print('ISOMAPPING {:.3f}'.format(np.mean(Accuracies_ISOMAP)))
    print('LLE {:.3f}'.format(np.mean(Accuracies_LLE)))








if __name__ == '__main__':
    ClassificationAccuracy()
    ISOMAPEmbbeding()
    LLEEmbedding()
    SwissRollWithConstrain()
    ### ------ Loading \ Creating data ------###
    # S = torus(R = 10,r = 4)
    # Classes = [3,5,7]
    # dig = digitis(Classes)
    #
    # ### ------ Isomap ------###
    # nei = [5,10,20]
    # #Ploting Torus Isomapping
    # fig = plt.figure(figsize=(30,10))
    # for i,j in enumerate(nei):
    #     Torus_isomap = Isomap(S,2,j)
    #     neighbors = j
    #     method = 'Torus ISOMAP'
    #     ax = fig.add_subplot(1,len(nei),i+1)
    #     scatter = ax.scatter(Torus_isomap[:, 0], Torus_isomap[:, 1], c=S[:,0:1], cmap=plt.cm.Spectral)
    #     # legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    #     # ax.add_artist(legend)
    #     # ax.legend()
    #     ax.set_title('{} with {} Neighbours'.format(method, neighbors))
    #     # making_plot(Torus_isomap, pallete=S[:, 0:1], neighbors=j, method='Torus ISOMAP') #An option to plot single graphs
    # plt.savefig('Torus ISOMAP embbeding for {} neighbour'.format(nei))
    #
    # #Plotting Digits Isomapping
    # for Argclass, Specificcalss in enumerate(dig):
    #     fig = plt.figure(figsize=(30,10))
    #     for i,j in enumerate(nei):
    #         neighbors = j
    #         Digit_isomap = Isomap(Specificcalss[0],2,j)
    #         method = 'Digit ISOMAP'
    #         ax = fig.add_subplot(1, len(nei), i + 1)
    #         scatter = ax.scatter(Digit_isomap[:, 0], Digit_isomap[:, 1], c=Specificcalss[1], cmap=plt.cm.Spectral)
    #         legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    #         ax.add_artist(legend)
    #         ax.legend()
    #         ax.set_title('{} with {} Neighbours'.format(method, neighbors))
    #         # making_plot(Digit_isomap, Specificcalss[1], neighbors=j, method='Digit ISOMAP') #An option to plot single graphs
    #     plt.savefig('Digits up to {} -  ISOMAP embbeding for {} neighbour'.format(Classes[Argclass], nei))
    #
    #
    # ### ------ LLE ------###
    # nei = [5,10,20]
    # # Plotting Torus
    # fig = plt.figure(figsize=(30, 10))
    # for i,j in enumerate(nei):
    #     Torus_LLE = LLE(S,2,j)
    #     neighbors = j
    #     method = 'Torus LLE'
    #     ax = fig.add_subplot(1, len(nei), i + 1)
    #     scatter = ax.scatter(Torus_LLE[:, 0], Torus_LLE[:, 1], c=S[:, 0:1], cmap=plt.cm.Spectral)
    #     # legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    #     # ax.add_artist(legend)
    #     # ax.legend()
    #     ax.set_title('{} with {} Neighbours'.format(method, neighbors))
    #     # making_plot(Torus_LLE, pallete=S[:, 0:1], neighbors=j, method='Torus LLE') #An option to plot single graphs
    # plt.savefig('Torus LLE embbeding for {} neighbour'.format(neighbors))
    #
    # #Plotting Digits
    # for Argclass, Specificcalss in enumerate(dig):
    #     fig = plt.figure(figsize=(30,10))
    #     for i,j in enumerate(nei):
    #         neighbors = j
    #         Digit_LLE = LLE(Specificcalss[0],2,j)
    #         method = 'Digit LLE'
    #         ax = fig.add_subplot(1, len(nei), i + 1)
    #         scatter = ax.scatter(Digit_LLE[:, 0], Digit_LLE[:, 1], c=Specificcalss[1], cmap=plt.cm.Spectral)
    #         legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    #         ax.add_artist(legend)
    #         ax.legend()
    #         ax.set_title('{} with {} Neighbours'.format(method, neighbors))
    #         # making_plot(Digit_isomap, Specificcalss[1], neighbors=j, method='Digit ISOMAP') #An option to plot single graphs
    #     plt.savefig('Digits up to {} - LLE embbeding for {} neighbour'.format(Classes[Argclass], nei))


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

