import numpy as np
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import scipy.sparse.linalg as sp
from scipy import sparse
from scipy.spatial import distance

def Conv_sparse_matrix(cols,rows):
    """
    Getting image dimentions and creating convolutional matrix for [[-1 / 8, -1 / 8, -1/8],[-1 / 8, 1, -1 / 8],[-1 / 8, -1 / 8, -1/8]] Kernel
    :param cols: Image cols number
    :param rows: Image rows number
    :return: Return Convulotional matrix (2*cols) X (2*rows)
    """
    cols0_first_block = np.zeros(cols)
    rows0_first_block = np.zeros(rows)
    vals_first_row = np.array([1, -1 / 8])
    vals_first_col = np.array([1, -1 / 8])
    pos_first = np.array([0, 1])
    rows0_first_block[pos_first] = vals_first_row
    cols0_first_block[pos_first] = vals_first_col

    # Create the first Toeplitz block
    First_matrix = toeplitz(cols0_first_block, rows0_first_block)  # The matrix with one in the middle

    cols0_sec_block = np.zeros(cols)
    rows0_sec_block = np.zeros(rows)
    vals_sec_row = np.array([-1 / 8, -1 / 8])
    vals_sec_col = np.array([-1 / 8, -1 / 8])
    pos_sec = np.array([0, 1])
    rows0_sec_block[pos_sec] = vals_sec_row
    cols0_sec_block[pos_sec] = vals_sec_col

    # Create the second Toeplitz block
    Sec_matrix = toeplitz(cols0_sec_block, rows0_sec_block)  # The matrix with 1/8

    cols0_outside_block = np.zeros(cols)
    rows0_outside_block = np.zeros(rows)
    vals_outside_row = np.array([1])
    vals_outside_col = np.array([1])
    pos_outside = np.array([1])
    rows0_outside_block[pos_outside] = vals_outside_row
    cols0_outside_block[pos_outside] = vals_outside_col

    outside_diag = toeplitz(cols0_outside_block,
                            rows0_outside_block)  # The matrix to build the conv matrix besides the diagonal

    skeleton_diag = sparse.eye(rows)
    outside_diag = sparse.kron(outside_diag, Sec_matrix)
    skeleton_diag = sparse.kron(skeleton_diag, First_matrix)

    Conv_matrix = outside_diag + skeleton_diag

    Conv_matrix = Conv_matrix.toarray() - Conv_matrix.toarray().sum(axis=1) * np.eye(rows*cols) # Set each row summation to equal 0

    return np.float32(Conv_matrix)

def Sec1(path, resize_image):
    """
    Calculate the classical reconstruction noisy image with Conv matrix instead of 2D Conv
    And Calculate the reconstruction image with Bilateral filtering using Graph-laplacian
    :param path: Path to image file
    :param resize_image: resize image size to acommodate computational resources
    :return: reconstructed image with classical approach
    """
    im = Image.open(path).convert('L')  #Load image with grayscale
    im = im.resize((resize_image,resize_image))
    imarray = np.array(im) / 255
    noise = np.random.normal(0,0.03,size= imarray.shape)
    noisy_image = imarray + noise
    figs, axs = plt.subplots(ncols=2,figsize=(20, 10))
    axs[0].imshow(imarray, cmap='gray')
    axs[0].title.set_text('Original image')
    axs[1].imshow(noisy_image, cmap='gray')
    axs[1].title.set_text('Noisy image')
    plt.savefig('sourceandnoise')

    cols, rows = imarray.shape
    N = rows*cols
    y = noisy_image.flatten('F') #Transpose the image and row stack

    Conv_matrix = Conv_sparse_matrix(rows,cols)

    # Image reconstruction by Conv matrix using different lamda values
    lamda = [0.2,0.4,0.8,1.5,2,5]
    plt.figure()
    figs, axs = plt.subplots(nrows=2,ncols=3,figsize=(20, 10))
    axs = axs.ravel()

    for i, val in enumerate(lamda):
        x = sp.cg((sparse.eye(N)+ val * Conv_matrix), y)
        x = x[0] # Return only the reconstruc signal without the successful flag
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        rec_image = np.reshape(x, [rows,cols], order='F')
        axs[i].imshow(rec_image, cmap='gray')
        axs[i].title.set_text('Reconstruction with Gamma equal to {}'.format(val))

    plt.savefig('Denoising image with several gamma values by using convolutional kernel')

    #Image reconstruction by Bilateral Filtering
    Weighted_mask = Conv_matrix.copy() # Copy the Conv sparse matrix
    Weighted_mask[Conv_matrix != 0] = 1 # Trasform the matrix to Neighbour matrix
    Weighted_mask = Weighted_mask - sparse.eye(N, dtype=np.float32) # Delete the self connection

    Sigma = 0.1
    Weights = np.float32(np.exp(-distance.cdist(np.expand_dims(y,axis=1),np.expand_dims(y,axis=1), metric='sqeuclidean')/(2*Sigma**2)))
    Weighted_Matrix = np.zeros([N,N], dtype=np.float32)
    Weighted_Matrix[Weighted_mask ==1] = Weights[Weighted_mask == 1]

    Degree = np.diag(np.sum(Weighted_Matrix, axis=1))
    Graph_Laplacian = Degree - Weighted_Matrix

    plt.figure()
    figs, axs = plt.subplots(nrows=2,ncols=3,figsize=(20, 10))
    axs = axs.ravel()

    for i, val in enumerate(lamda):
        x_b = sp.cg((np.eye(N) + val * Graph_Laplacian), y)
        x_b = x_b[0]  # Return only the reconstruc signal without the successful flag
        x_b = (x_b - np.min(x_b)) / (np.max(x_b) - np.min(x_b))
        rec_image = np.reshape(x_b, [rows, cols], order='F')
        axs[i].imshow(rec_image, cmap='gray')
        axs[i].title.set_text('Reconstruction with Gamma equal to {}'.format(val))

    plt.savefig('Denoising image with several gamma values by using bilateral filtering')

    return

def Sec2(path, r, Sigma, s, eigenVec_idx, tau):
    """
    Calculate 3D mesh noisy reconstruction using mesh low pass filtering
    :param path: path to mesh file
    :param r: the neighbour radius
    :param Sigma: the Gaussain variance
    :param s: the noise addetive variance parameter
    :param eigenVec_idx: which eigenvector indices to use for coloroing the mesh
    :param tau: which tau values to run to reconstruct the mesh
    :return: return denoisy meshes with several taus
    """
    # r = 1
    # Sigma = 1
    # s = 2

    v,f = read_off(path)
    N = v[:,0].shape[0]

    render_pointcloud(v,np.ones(N),screenshot='OriginalMesh')
    # render_surface(v,f, screenshot='Original Surface mesh')

    noise = np.random.normal(0, 0.1*s**2, size=[N,3])

    noisy_mesh = v + noise
    render_pointcloud(noisy_mesh,np.ones((N)),screenshot='NoisyMesh')
    # render_surface(noisy_mesh,f, screenshot='Noisy Surface mesh')



    Weights = np.exp(-distance.cdist(noisy_mesh, noisy_mesh, metric='sqeuclidean') / (2 * Sigma ** 2))
    Weights[Weights > r] = 0

    #Construct Normalized Graph Laplacian
    ngl = np.eye(N) - ((np.diag(np.sum(Weights, axis=1)**-0.5)) @ Weights @ (np.diag(np.sum(Weights, axis=1)**-0.5)))
    eigen_Vec, eigen_Val = spectral_decomp(ngl, N)

    # Plot and save the mesh with colored eigenvectors
    # eigenVec_idx = [1,2,4,10]
    plot_results(noisy_mesh, eigen_Vec[:,eigenVec_idx], eigenVec_idx)


    # tau = [0.2,0.5,0.8,1.5,3,5]
    figs, axs = plt.subplots(ncols=len(tau), figsize=(20, 10))
    figs_z, axs_z = plt.subplots(nrows=2, ncols=int(len(tau)/2), figsize=(20, 10))
    axs_z = axs_z.ravel()
    for i, val in enumerate(tau):
        low_pass_graph = np.exp(-(val*eigen_Val)/eigen_Val[-1])
        axs[i].plot(low_pass_graph)
        axs_z[i].plot(low_pass_graph[:1500])
        axs[i].title.set_text('Tau equal to {}'.format(val))
        axs_z[i].title.set_text('Tau equal to {}'.format(val))
        Denois_Mesh = eigen_Vec @ np.diag(low_pass_graph) @ eigen_Vec.T @ noisy_mesh
        render_pointcloud(Denois_Mesh, np.ones((N)), screenshot='Denoise with Tau equal to {0:.0f}'.format(val*10))

    figs.savefig('Tau values')
    figs_z.savefig('ZoomIn Tau values')
    # Denois_Mesh = eigen_Vec @ np.diag(low_pass_graph) @ eigen_Vec.T @ noisy_mesh
    #
    # render_pointcloud(Denois_Mesh, np.ones((N)))

    return

def read_off(files):
    """
    Red off files
    :param files: path to off files
    :return: vertices and faces
    """
    with open(files) as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]

        verts = np.array([np.array(xi) for xi in verts])
        faces = np.array([np.array(xi) for xi in faces])

        return verts, faces

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

def render_surface(v,f, screenshot='defualt' , scalar_func=None, colormap=None):
    """
    :param scalar_func: Scalar funcition
    :param colormap: color map pallet
    :return: Plotting loaded OFF file with define scalar function
    """
    surface = pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))
    # surface['color'] = scalar_func
    surface.plot(cmap=colormap, screenshot=screenshot)

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

    return G_eigenVectors, G_eigenValues

def plot_results(data, eigenvectors, eigenVec_idx):
    for i in range(len(eigenvectors[0,:])):
        screenshot = 'Mesh colored by {} eigenvector'.format(eigenVec_idx[i])
        render_pointcloud(data, eigenvectors[:,i],screenshot=screenshot)



if __name__ == '__main__':
    # path = r'PicSource\cameraman.png'
    # resize_image = 128
    # Sec1(path, resize_image)

    r = 1
    Sigma = 1
    s = 2
    eigenVec_idx = [1,2,4,10]
    tau = [0.2,0.5,0.8,1.5,3,5]

    path_sec2 = r'Meshes\toilet_0003.off'
    # # path_sec2 = r'Meshes\sofa_0003.off'
    Sec2(path_sec2, r, Sigma, s, eigenVec_idx, tau)



    print('Finish')