import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from scipy.sparse import csr_matrix

def torusGraph(n,N, NeiMinThresh = 8, noisy=False):
    """
    Creating Turos topology with at least number of define neighbours
    :param n: Number of main Ring nodes
    :param N: Number of secondry ring nodes
    :param NeiMinThresh: Minimum neighbourhoods for each nodes
    :param noisy: Whether or not create a noisy topology
    :return: Graph and nodes positions
    """
    np.random.seed(42)
    # Creating random variables array XY
    x = np.random.rand(71, 1)
    y = np.random.rand(31, 1)
    #Creating torus
    #Need to construct such that the Nxn wheres N is the number of samples

    theta = np.linspace(0, 2. * np.pi, n)
    phi = np.linspace(0, 2. * np.pi, N)
    theta, phi = np.meshgrid(theta, phi)
    c, a = 2, 1
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    pos = {i: (np.ravel(x)[i],np.ravel(y)[i],np.ravel(z)[i]) for i in range(n*N)} #Key Dict Positions

    if noisy is not False:
        noise = np.random.normal(0,0.01, size=(np.ravel(x).shape[0],3))
        pos = {i: ((np.ravel(x)[i],np.ravel(y)[i],np.ravel(z)[i])+noise[i]) for i in range(n*N)}

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_zlim(-3, 3)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='k', edgecolors='w')
    ax1.set_title('Turos with Main Ring radius {} and secondary ring radius {}'.format(c,a))
    ax1.view_init(36, 26)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_zlim(-3, 3)
    ax2.scatter(x, y, z)
    ax2.view_init(0, 0)
    ax2.set_xticks([])
    plt.gcf().set_size_inches(15, 8)
    plt.savefig('Turos with Main Ring radius {} and secondary ring radius {}'.format(c,a))
    # plt.show()

    ### --- Optional to get Rnadom Turos Graph with define distance for edge connections --- ###
    i = c * 0.5 #Set the neighbours radius with direct connection to major Torus radius
    ValidEdgesCheckPoint = 1e10
    while ValidEdgesCheckPoint > NeiMinThresh:
        i -= 0.1
        G = nx.random_geometric_graph(n*N, i, pos=pos)
        ValidEdgesCheckPoint = np.min(np.sum(nx.adjacency_matrix(G).toarray(), axis=1))

    return G, pos

def SectionOne():

    n = 201
    DG = nx.Graph()
    DG= nx.path_graph(n)

    Pn = DG.copy() # Classic path graph
    Rn = DG.copy() #Ring path graph
    Rn.add_edge(0,n-1) # Adding edge for closing the ring

    Pn_pos = nx.bipartite_layout(Pn, Pn.nodes)
    Rn_pos = nx.circular_layout(Rn)


    Pn_adj = nx.adjacency_matrix(Pn)
    Rn_adj = nx.adjacency_matrix(Rn)

    row = np.array([*range(n)])
    col = np.array([*range(n)])
    # degrees = [val for (node, val) in Pn.degree()]        ## Method that extract node degree for each node
    Pn_deg = csr_matrix(([val for (node, val) in Pn.degree()], (row,col))) # Creating diagonal sparse matrix for Path degree
    Rn_deg = csr_matrix(([val for (node, val) in Rn.degree()], (row,col))) # Creating diagoal sparse matrix for Ring degree

    # Laplacian matrix defind as L = D - A, so basically we can substract the adjacency matrix from degree matrix Or
    # Using lalacian matrix method with NetworkX
    Pn_lpcn = nx.laplacian_matrix(Pn)
    Rn_lpcn = nx.laplacian_matrix(Rn)


    Pn_eigenValues, Pn_eigenVectors = linalg.eig(Pn_lpcn.toarray())
    idx = (-Pn_eigenValues).argsort()[::-1] #Since argsort is order from largest to smallest I multiply by -1 to order in decreasing order and not decent order
    Pn_eigenValues = Pn_eigenValues[idx]
    Pn_eigenVectors = Pn_eigenVectors[:, idx]

    Rn_eigenValues, Rn_eigenVectors = linalg.eig(Rn_lpcn.toarray())
    idx = (-Rn_eigenValues).argsort()[::-1] #Since argsort is order from largest to smallest I multiply by -1 to order in decreasing order and not decent order
    Rn_eigenValues = Rn_eigenValues[idx]
    Rn_eigenVectors = Rn_eigenVectors[:, idx]

    plt.figure()
    ax = plt.gca()
    nx.draw(Pn,Pn_pos, node_size=50)
    ax.set_title('Path Graph')
    plt.savefig('Path Graph with {} nodes'.format(n-1))
    plt.figure()
    ax = plt.gca()
    nx.draw(Rn, Rn_pos,node_size=50)
    ax.set_title('Ring Graph')
    plt.savefig('Ring Graph with {} nodes'.format(n - 1))
    plt.figure()
    plt.plot(Rn_eigenValues, 'g^')
    plt.title('Ring Eigevalues')
    plt.savefig('Ring Graph Laplacian matrix sorted eigenvalues - {} nodes'.format(n - 1))
    plt.figure()
    plt.plot(Pn_eigenValues, 'r--')
    plt.title('Path Eigevalues')
    plt.savefig('Path Graph Laplacian matrix sorted eigenvalues - {} nodes'.format(n - 1))

    fig1, axs = plt.subplots(2,2,figsize=(10,10))
    nx.draw(Pn, pos=Pn_pos, ax=axs[0,0], node_size=50 ,node_color=Pn_eigenVectors[:,0])
    nx.draw(Pn, pos=Pn_pos, ax=axs[0,1], node_size=50 ,node_color=Pn_eigenVectors[:,1])
    nx.draw(Pn, pos=Pn_pos, ax=axs[1,0], node_size=50 ,node_color=Pn_eigenVectors[:,4])
    nx.draw(Pn, pos=Pn_pos, ax=axs[1,1], node_size=50 ,node_color=Pn_eigenVectors[:,9])
    axs[0, 0].set_title('Ring graph with First eigenvalue')
    axs[0, 1].set_title('Ring graph with Second eigenvalue')
    axs[1, 0].set_title('Ring graph with Fifth eigenvalue')
    axs[1, 1].set_title('Ring graph with Thenth eigenvalue')
    plt.savefig('Path Graph colored by eigenvalues'.format(n - 1))

    fig2, axs = plt.subplots(2, 2,figsize=(10,10))
    nx.draw(Rn,pos=Rn_pos, ax=axs[0, 0], node_size=50 ,node_color=Rn_eigenVectors[:,0])
    nx.draw(Rn,pos=Rn_pos, ax=axs[0, 1], node_size=50 ,node_color=Rn_eigenVectors[:,1])
    nx.draw(Rn,pos=Rn_pos, ax=axs[1, 0], node_size=50 ,node_color=Rn_eigenVectors[:,4])
    nx.draw(Rn,pos=Rn_pos, ax=axs[1, 1], node_size=50 ,node_color=Rn_eigenVectors[:,9])
    axs[0, 0].set_title('Ring graph with First eigenvalue')
    axs[0, 1].set_title('Ring graph with Second eigenvalue')
    axs[1, 0].set_title('Ring graph with Fifth eigenvalue')
    axs[1, 1].set_title('Ring graph with Thenth eigenvalue')
    plt.savefig('Ring Graph colored by eigenvalues'.format(n - 1))
    # plt.show()

def SectionTwo():
    NX = 71
    NY = 31
    Rnx = nx.Graph()
    Rnx = nx.path_graph(NX)
    Rnx.add_edge(0, NX - 1)  # Adding edge for closing the ring
    Rnx_pos = nx.circular_layout(Rnx)

    Rny = nx.Graph()
    Rny = nx.path_graph(NY)
    Rny.add_edge(0, NY - 1)  # Adding edge for closing the ring
    Rny_pos = nx.circular_layout(Rny)

    G = nx.cartesian_product(Rnx, Rny)
    G_adj = nx.adjacency_matrix(G)

    row = np.array([*range(NX * NY)])
    col = np.array([*range(NX * NY)])
    G_deg = csr_matrix(
        ([val for (node, val) in G.degree()], (row, col)))  # Creating diagonal sparse matrix for Path degree

    G_lpcn = nx.laplacian_matrix(G)

    G_eigenValues, G_eigenVectors = linalg.eig(G_lpcn.toarray())
    idx = (-G_eigenValues).argsort()[::-1]  # Since argsort is order from largest to smallest I multiply by -1 to order in decreasing order and not decent order
    G_eigenValues = np.real(G_eigenValues[idx])
    G_eigenVectors = np.real(G_eigenVectors[:, idx])

    _, pos= torusGraph(NX,NY) #I dno't care from neighbourhood in this case
    x = np.array(list(pos.values()), dtype=float)[:,0]
    y = np.array(list(pos.values()), dtype=float)[:,1]
    z = np.array(list(pos.values()), dtype=float)[:,2]

    ThreeDPlot(x,y,z,1)

    plt.figure()
    plt.plot(G_eigenValues, 'g^')
    plt.title('G Laplacian Matrix sorted Eigevalues')
    plt.savefig('G Laplacian Matrix sorted Eigevalues')

    ThreeDPlot(x, y, z, 4, eigenvectors=G_eigenVectors,PresentedEig=[0,1,4,9])
    # plt.show()

def SectionThree():
    NX = 71
    NY = 31
    Nei = 8

    GN , pos = torusGraph(NX,NY,Nei,noisy=True) #Create Turos graph with up to Nei neighbour for each node and noisy

    GN_adj = nx.adjacency_matrix(GN)

    row = np.array([*range(NX * NY)])
    col = np.array([*range(NX * NY)])
    GN_deg = csr_matrix(
        ([val for (node, val) in GN.degree()], (row, col)))  # Creating diagonal sparse matrix for Path degree

    GN_lpcn = nx.laplacian_matrix(GN)

    GN_eigenValues, GN_eigenVectors = linalg.eig(GN_lpcn.toarray())
    idx = (-GN_eigenValues).argsort()[::-1]  # Since argsort is order from largest to smallest I multiply by -1 to order in decreasing order and not decent order
    GN_eigenValues = GN_eigenValues[idx]
    GN_eigenVectors = GN_eigenVectors[:, idx]

    x = np.array(list(pos.values()), dtype=float)[:,0]
    y = np.array(list(pos.values()), dtype=float)[:,1]
    z = np.array(list(pos.values()), dtype=float)[:,2]

    ThreeDPlot(x, y, z, 1,Noisy=True)

    plt.figure()
    plt.plot(GN_eigenValues, 'g^')
    plt.title('G Noisy Laplacian Matrix sorted eigenvalues - {}x{} nodes '.format(NX,NY))
    plt.savefig('G Noisy Laplacian Matrix sorted Eigevalues')

    ThreeDPlot(x, y, z, 4, eigenvectors=GN_eigenVectors,PresentedEig=[0,1,4,9], Noisy=True)
    # plt.show()

def ThreeDPlot(x,y,z,NumPlots=1,eigenvectors=None,PresentedEig=None, Noisy=False):
    """
    Ploting 3D topology
    :param x: X-Coordinate
    :param y: Y-Coordinate
    :param z: Z-Coordinate
    :param NumPlots: Number of plots tp plot
    :param eigenvectors: eigenvectors values
    :param PresentedEig: Specify which eigenvalue to represent
    :param Noisy: Whether or not the graph topology is noisy
    :return: 3D plot of the Torus topology
    """
    fig = plt.figure()
    elev = 36
    azim = 26

    color = np.ones((x.shape[0],2)) #Set ones as deafult colors
    TitleEig = np.array([1]) #Set deafult title
    if eigenvectors is not None:
        color = eigenvectors[:, PresentedEig]
        TitleEig = PresentedEig

    Graphtype = 'G'
    if Noisy is not False:
        Graphtype = 'G Noisy'

    for i in range(NumPlots):
        ax = fig.add_subplot(2, int(np.ceil(NumPlots / 2)), i+1, projection='3d')
        ax.set_zlim(-3, 3)
        ax.scatter(x, y, z, c=color[:,i])
        ax.view_init(elev, azim)
        ax.set_xticks([])
        ax.set_title('{} graph colored by eigenvalue {}'.format(Graphtype, TitleEig[i]))

    plt.gcf().set_size_inches(15, 8)
    plt.savefig('{} Graph {}'.format(Graphtype,NumPlots))
    return

# def SectionTwoBackUp():
#     NX = 71
#     NY = 31
#     G, pos= torusGraph(NX,NY)
#     G_adj = nx.adjacency_matrix(G)
#
#     row = np.array([*range(NX * NY)])
#     col = np.array([*range(NX * NY)])
#     G_deg = csr_matrix(
#         ([val for (node, val) in G.degree()], (row, col)))  # Creating diagonal sparse matrix for Path degree
#
#     G_lpcn = nx.laplacian_matrix(G)
#
#     G_eigenValues, G_eigenVectors = linalg.eig(G_lpcn.toarray())
#     idx = (-G_eigenValues).argsort()[::-1]  # Since argsort is order from largest to smallest I multiply by -1 to order in decreasing order and not decent order
#     G_eigenValues = G_eigenValues[idx]
#     G_eigenVectors = G_eigenVectors[:, idx]
#
#     plt.figure()
#     plt.plot(G_eigenValues, 'g^')
#     plt.title('G Laplacian Matrix sorted Eigevalues')
#
#     x = np.array(list(pos.values()), dtype=float)[:,0]
#     y = np.array(list(pos.values()), dtype=float)[:,1]
#     z = np.array(list(pos.values()), dtype=float)[:,2]
#
#     elev = 36
#     azim = 26
#     fig = plt.figure()
#     ax1 = fig.add_subplot(221, projection='3d')
#     ax1.set_zlim(-3, 3)
#     ax1.view_init(elev, azim)
#     ax1.scatter(x,y,z, c=G_eigenVectors[:,0])
#     ax1.view_init(elev, azim)
#     ax1.set_xticks([])
#     ax2 = fig.add_subplot(222, projection='3d')
#     ax2.set_zlim(-3, 3)
#     ax2.scatter(x,y,z, c=G_eigenVectors[:,1])
#     ax2.view_init(elev, azim)
#     ax2.set_xticks([])
#     ax3 = fig.add_subplot(223, projection='3d')
#     ax3.set_zlim(-3, 3)
#     ax3.scatter(x,y,z, c=G_eigenVectors[:,4])
#     ax3.view_init(elev, azim)
#     ax3.set_xticks([])
#     ax4 = fig.add_subplot(224, projection='3d')
#     ax4.set_zlim(-3, 3)
#     ax4.scatter(x,y,z, c=G_eigenVectors[:,9])
#     ax4.view_init(elev, azim)
#     ax4.set_xticks([])
#     plt.show()
#
#     fig3, axs = plt.subplots(2, 2)
#     nx.draw(G, ax=axs[0, 0], node_size=50, node_color=G_eigenVectors[:, 0])
#     nx.draw(G, ax=axs[0, 1], node_size=50, node_color=G_eigenVectors[:, 1])
#     nx.draw(G, ax=axs[1, 0], node_size=50, node_color=G_eigenVectors[:, 4])
#     nx.draw(G, ax=axs[1, 1], node_size=50, node_color=G_eigenVectors[:, 9])
#
#     print('Finish')

if __name__ == '__main__':
    SectionOne()
    SectionTwo()
    SectionThree()
    # SectionTwoBackUp()
    plt.show()

    print('Finish')