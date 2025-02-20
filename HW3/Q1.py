import numpy as np
import os
import eikonalfm
import matplotlib.pyplot as plt
from os.path import join as pjoin
import scipy.io as sio
import networkx as nx
from PIL import Image
import meshio
import gdist
import pyvista as pv


def shortest_path(FMM_dist, suorce, target, step_size=1):
    """
    Return the shortes path
    :param FMM_dist: distances matrix
    :param suorce: the point you would like to navigate to --> Y coordinate and than X coordinate
    :param target: the point you would like to start to --> Y coordinate and than X coordinate
    :param step_size: the step size (how many pixels)
    :return:
    """
    Fy, Fx = np.gradient(FMM_dist)
    path_maze = []

    indices = target
    while (suorce!=indices):
        dy,dx = -1*Fx[indices] * step_size, -1*Fy[indices] * step_size
        D = np.sqrt(dx**2 + dy**2)
        path_maze.append((target[0]+dx/D, target[1]+dy/D))
        target = (target[0]+dx/D, target[1]+dy/D)
        indices = (round(target[0]),round(target[1]))

    path_maze = [t[::-1] for t in path_maze]

    return path_maze

def plotting_shortest_path(data,shortest_path,title, color=1):
    """
    Plotting the data with the shortest path
    :param data: the array which we would like to plot
    :param shortest_path: the calculated shortest path
    :param title: the method the shortest path calculated with
    :param color:
    :return: plotting the original data with shortest path on top of it
    """
    plt.figure()
    plt.imshow(data, cmap='jet')
    plt.plot(*zip(*shortest_path), color='white')
    plt.title(title)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(title)
    plt.show()


def Graph_adj(data):
    """
    Creatre graph Adjcency matrix
    :param data: provided data for adjacency matrix
    :return: the adjacency matrix as nx (networkxx) Graph
    """
    graph = nx.Graph()
    rows, colums = data.shape
    N_element = rows * colums

    data_as_vec = data.flatten() #Copy data array to new indexing array
    # Build neighbour like array with indexing
    neighbours = np.asarray(
        [(-1, -1),
         (-1, 0),
         (-1, 1),
         (0, -1),
         (0, 1),
         (1, -1),
         (1, 0),
         (1, 1)],
        dtype=np.int)

    for i in range(N_element):
        # Condisder i as vertices, find the j vertices to connet i and create edge
        if data_as_vec[i] == 1:
            position = np.asarray([i // colums, i % colums]) + neighbours
            position = np.minimum((rows - 1,colums - 1), np.maximum(position, (0,0)))
            for q, w in position:
                if data[q, w] == 1:
                    # find the index of j connected neighbour
                    j = int(q * colums + w)
                    # Add graph edge between connected vertices
                    graph.add_edge(i, j)
    return graph

def MDS(data, n_dim):
    """
    Multidimentional scaling - reducing the dimentionality to n_dim
    :param data: data which we would like to reduce the dimentionality
    :param n_dim: what is the dimentsion we would like to embed
    :return: Reduce dimention with MDS algo.
    """
    i, n_feature = data.shape
    H = np.identity(i) - (1 / i) * np.ones_like(data)
    K = -0.5 * (H @ (data**2) @ H)
    Reduce_dim = spectral_decomp(K, n_dim)

    return Reduce_dim

def spherical_mds(data, n_dim):
    """
    Spherical MDS implement as shown in Lecture 7
    :param data: the data we would like to reduce the dimensionality
    :param n_dim: number of dimention to embed
    :return: Spherical embedded data
    """
    Data_phi = np.cos(data)
    Spectral_Spherical = spectral_decomp(Data_phi, n_dim)

    return Spectral_Spherical

def spectral_decomp(data,n_dim):
    """
    Spectral decomposition
    :param data: data
    :param n_dim: SVD for n_dim
    :return: low dimensionality data with SVD decomposition
    """
    G_eigenValues, G_eigenVectors = np.linalg.eigh(data)
    idx = (G_eigenValues).argsort()
    G_eigenValues.sort()   # = np.real(G_eigenValues[idx])
    G_eigenVectors = G_eigenVectors[:, idx]

    G_eigenValues = G_eigenValues[-n_dim:]
    G_eigenVectors = G_eigenVectors[:, -n_dim:]

    return G_eigenVectors @ np.power(np.diag(G_eigenValues), 0.5)


def FarestPointSampling(geodesics_dist, n_dim_compress, init_v):
    """
    Farest point sampling on Triangular meshes. using argmax{min{d(v,s)}}
    :param geodesics_dist: the geodesics distance array extracted from the .ply mesh file
    :param n_dim_compress: number of vertices to extract which better represent the object
    :param init_v: the initial point to start with
    :return: array of n_dim_compress vertices --> the farest vertices
    """
    vertic_set = [init_v]
    for i in range(n_dim_compress - 1):
        reduced = geodesics_dist[:, vertic_set]
        # Using argmax{min{d(v,s)}}
        # Choose the maximal point which is the closet to
        v = np.argmax(np.min(reduced, axis=1), axis=0)
        vertic_set.append(v)

    return vertic_set

def Sec1(path, source, target):
    """
    :param path: path to file
    :param source: the source for computing
    :param target: the target for computing
    """
    im_maze = Image.open(path).convert('L') #Load the binary image
    im_maze = np.asarray(im_maze.point(lambda x: 0 if x < 128 else 255, '1'), dtype=np.int)
    rows, cols = im_maze.shape

    maze_bw = np.zeros_like(im_maze)
    maze_bw[im_maze == 1] = 1000 #Scaling for Fast Narching Method
    maze_bw[im_maze == 0] = 1 #Add 1 to avoid zeros

    dx = (1.0,1.0)
    order = 2

    # Using build-in package FFM
    tau_fm = eikonalfm.fast_marching(maze_bw, source, dx, order)

    ### Creating contur for background
    Contur_Color = eikonalfm.fast_marching(maze_bw, (383, 814), dx, order)
    Contur_Color[im_maze == 0] = -0.1
    Contur_Color[im_maze == 1] = ((Contur_Color[im_maze == 1] - 0) / (np.max(Contur_Color) - 0))
    plt.figure()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.imshow(Contur_Color, cmap='jet')
    plt.title('Distance map as contour color map')
    plt.savefig('Colored Contur')
    plt.show()
    #####

    path_maze = shortest_path(tau_fm,source,target)
    plotting_shortest_path(Contur_Color, path_maze,'Fast Marching Method',tau_fm)

    maze_adj = Graph_adj(im_maze)
    maze_graph_shortest_path = nx.shortest_path(
        maze_adj, source=int(source[0] * cols + source[1]), target=int(target[0] * cols + target[1])
    )

    shortest_path_dijkstra = []
    for node in maze_graph_shortest_path:
        row, col = np.asarray([node // cols, node % cols])
        shortest_path_dijkstra.append((row, col))

    shortest_path_dijkstra = [t[::-1] for t in shortest_path_dijkstra] #Flip the touple to be X,Y oder coordinates
    plotting_shortest_path(Contur_Color, shortest_path_dijkstra, 'Shoertest path using Graph Dijkstra method')

def Sec2(path, source_pool, target_pool):
    """
    :param path: path to file
    :param source_pool: the source for computing
    :param target_pool: the target for computing
    """

    mat_fname = pjoin(path, 'pool.mat')
    mat_contents = sio.loadmat(mat_fname)
    pool = mat_contents['n']

    dx = (1.0,1.0)
    order = 2

    pool_fm = eikonalfm.fast_marching((1/pool)**1.9, source_pool, dx, order)
    path_pool = shortest_path(pool_fm,source_pool,target_pool)
    plotting_shortest_path(pool, path_pool, 'OPL Shortest path')

def Sec4(path, n_dim = 2, embedding='MDS'):
    """
    :param path: The path for the mesh file
    :param n_dim: number of dim.
    :param embedding: which method to use for embbeding MDS or Spherical MDS
    """

    file_path = path.split('.')[0]
    mesh_str = file_path.split('tr_')[1] #Loading and splitting to save the data and figures
    ply = meshio.read(path) #Read the ply file
    vertices = ply.points #Extracting vertices
    faces = ply.cells_dict['triangle'] #Extracting faces
    faces_fix = np.concatenate((np.expand_dims(len(faces[0]) * np.ones((len(faces)), ), 1).astype(np.int), np.array(faces)), 1)

    # # # Compute the geodesic distance by build-in package
    geodesics_dist = gdist.local_gdist_matrix(vertices.astype(np.float64), faces.astype(np.int32))
    # Saving numpy array
    np.save(mesh_str, geodesics_dist.toarray())
    #loading numpy matrix
    geodesics_dist = np.load(r'{}.npy'.format(mesh_str))

    render_pointcloud(vertices,geodesics_dist[:, 250], screenshot=mesh_str)

    if embedding == 'MDS':
        isometric_embedding_mds = MDS(geodesics_dist, n_dim)
        embbeded_n_dim_geodesic_distance = gdist.local_gdist_matrix(isometric_embedding_mds.astype(np.float64),faces.astype(np.int32))
        # Saving numpy array
        np.save('{}_n_dim_{}'.format(embedding,mesh_str), embbeded_n_dim_geodesic_distance.toarray())
        # loading numpy array matrix
        embbeded_n_dim_geodesic_distance = np.load(r'{}_n_dim_{}.npy'.format(embedding,mesh_str))

    elif embedding == 'Spherical MDS':
        isometric_embedding_mds = spherical_mds(geodesics_dist, n_dim)
        embbeded_n_dim_geodesic_distance = gdist.local_gdist_matrix(isometric_embedding_mds.astype(np.float64),faces.astype(np.int32))
        # Saving numpy array
        np.save('{}_n_dim_{}'.format(embedding,mesh_str), embbeded_n_dim_geodesic_distance.toarray())
        # loading numpy array matrix
        embbeded_n_dim_geodesic_distance = np.load(r'{}_n_dim_{}.npy'.format(embedding,mesh_str))

    render_pointcloud(isometric_embedding_mds, embbeded_n_dim_geodesic_distance[:, 250],screenshot=mesh_str +'_'+ embedding)

    return np.linalg.norm((geodesics_dist - embbeded_n_dim_geodesic_distance), ord='fro')


def Sec5(path, compress_to, init_v):
    """
    :param path: The path for the mesh file
    :param compress_to: number of points to compress
    :param init_v: the initial vertex to start with
    """
    file_path = path.split('.')[0]
    mesh_str = file_path.split('tr_')[1] #Loading and splitting to save the data and figures
    ply = meshio.read(path, "ply") #Read the ply file
    faces = ply.cells_dict['triangle']  # Extracting faces
    faces_fix = np.concatenate((np.expand_dims(len(faces[0]) * np.ones((len(faces)), ), 1).astype(np.int), np.array(faces)), 1)
    vertices = ply.points #Extracting vertices
    n_points, _ = ply.points.shape

    # # # Compute the geodesic distance by build-in package
    # geodesics_dist = gdist.local_gdist_matrix(vertices.astype(np.float64), faces.astype(np.int32))
    # # Saving numpy array
    # np.save(mesh_str, geodesics_dist.toarray())
    #loading numpy matrix
    geodesics_dist = np.load(r'{}.npy'.format(mesh_str))

    mesh_plot = pv.Plotter(shape=(1,4))

    pointcloud = pv.PolyData(vertices,faces_fix)
    mesh_plot.subplot(0,0)
    mesh_plot.add_mesh(pointcloud, cmap='hot', scalars=geodesics_dist, show_edges=True)
    mesh_plot.add_text('Original Mesh {} vertices'.format(n_points))

    for i in range(len(compress_to)):
        vertices_set = FarestPointSampling(geodesics_dist, compress_to[i], init_v)

        vertices_reduced = vertices[vertices_set]
        gdists_reduced = geodesics_dist[init_v, vertices_set]

        # pointcloud = pv.PolyData(vertices_reduced)
        mesh_plot.subplot(0, i + 1)
        mesh_plot.add_mesh(pointcloud, cmap='hot', scalars=geodesics_dist, show_edges=True)
        mesh_plot.add_mesh(vertices_reduced, render_points_as_spheres=True, point_size=7)
        mesh_plot.add_text('Geo. colored {} vertices'.format(compress_to[i]))

    mesh_plot.show(full_screen=True,screenshot='{}_ReducedMesh by FarestPointSampling'.format(mesh_str))
    return

def render_pointcloud(v, func, colormap='hot', screenshot='defualt'):
    """
    #Rendering only the vertices of mesh as sphees and asign color to rach sphere
    :param func: Color map weighted function
    :param colormap: Color pallet
    :return: Ploting the graph
    """
    pointcloud = pv.PolyData(v)
    pointcloud['color'] = func
    pointcloud.plot(render_points_as_spheres=True, cmap=colormap, screenshot=screenshot)

if __name__ == '__main__':
    #######################################################
    ##############------ Section 1 ------##################
    # source = (383, 814)
    # target = (233, 8)
    # Sec1(r'Resources\maze.png', source, target)
    #
    #######################################################
    ##############------ Section 2 ------##################
    # source_pool = (0,0)
    # target_pool = (499,399)
    # Sec2(pjoin(os.getcwd(), 'Resources'),source_pool,target_pool)
    # # Flip to verify this is indeed correct # #
    # Sec2(pjoin(os.getcwd(), 'Resources'),target_pool,source_pool)

    #######################################################
    ##############------ Section 4 ------##################
    # err_MDS = Sec4(r'Resources\tr_reg_000.ply', n_dim=3, embedding='MDS')
    # err_SMDS = Sec4(r'Resources\tr_reg_000.ply', n_dim=3, embedding='Spherical MDS')

    # err_MDS = Sec4(r'Resources\tr_reg_001.ply', n_dim=3, embedding='MDS')
    # err_SMDS = Sec4(r'Resources\tr_reg_001.ply', n_dim=3, embedding='Spherical MDS')

    #######################################################
    ##############------ Section 5 ------##################
    # Sec5(r'Resources\tr_reg_000.ply',(1000, 2000, 4000), 120)
    # Sec5(r'Resources\tr_reg_001.ply', (1000, 2000, 4000), 120)

    print('Finish')
