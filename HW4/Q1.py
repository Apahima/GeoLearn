import numpy as np
import pyvista as pv
from PIL import Image
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import circulant

def Sec1(path):
    im = Image.open(path).convert('L')  #Load image with grayscale
    imarray = np.array(im) / 255
    noise = np.random.normal(0,0.01,size= imarray.shape)
    noisy_image = imarray + noise
    figs, axs = plt.subplots(ncols=2)
    axs[0].imshow(imarray, cmap='gray')
    axs[1].imshow(noisy_image, cmap='gray')

    cols, rows = imarray.shape
    N = rows*cols


    #https://stackoverflow.com/questions/39597649/create-sparse-circulant-matrix-in-python
    h_eight = np.array([[-1/8,-1/8,-1/8,],
                       [-1/8,1,-1/8],
                       [-1/8,-1/8,-1/8]]).flatten()
    offsets = np.array([1,2,rows+1,rows+2,rows+3,2*rows+1,2*rows+2,2*rows+3])
    dupvals = np.concatenate((h_eight, h_eight[::-1]))
    dupoffsets = np.concatenate((np.zeros(1),offsets,-offsets))

    a = sparse.diags(dupvals, dupoffsets, shape=(N, N))

    return a

def read_off(files):
    """

    :param files: OFF raw file
    :return: verts and faces as np.array
    """
    with open(files) as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        # faces = [(int(x), int(y), int(z)) for x, y, z in faces] #Convert faces 'xxx' str to int

        # faces = []
        # for i_face in range(n_faces):
        #     for s in file.readline().strip().split(' '):
        #         faces.append(s)
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

def render_surface(v,f, scalar_func=None, colormap=None):
    """

    :param scalar_func: Scalar funcition
    :param colormap: color map pallet
    :return: Plotting loaded OFF file with define scalar function
    """
    surface = pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))
    # surface['color'] = scalar_func
    surface.plot(cmap=colormap)

if __name__ == '__main__':
    path = r'PicSource\cameraman.png'
    Sec1(path)


    v,f = read_off(r'Meshes\bathtub_0111.off')
    # render_pointcloud(v,np.ones((7940)))
    render_surface(v,f)

    print('Finish')