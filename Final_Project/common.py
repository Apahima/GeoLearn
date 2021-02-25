import imageio
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
def create_gif(path,name):
    """
    :param path: Path where to load the images for GIF
    :param name: Gif name
    :return: saving batch of images into Gif
    """
    image_list = []
    for filename in sorted(glob.glob(path + '\\*'),key=os.path.getmtime): #assuming gif
        im=Image.open(filename)
        image_list.append(im)

    images = []
    for filename in image_list:
        images.append(imageio.imread(filename.filename))
    imageio.mimsave(name, images, fps=2)

    return


def multi_plot(X, k, data='NotGetDataType'):
    plt.figure()
    for idx in range(len(X[0,:])):
        plt.scatter(np.arange(1,k,1), X[1:,idx])

    plt.savefig('{}EigenValues'.format(data))

def SptDM_SC(X, Affinity, Cluster_time, Lamda, DS_Name):
    fix, ax = plt.subplots(3, 1, figsize=(19,10))
    for idx, i in enumerate(Cluster_time):
        Affinity = np.array(Affinity)
        clustering = spectral_clustering(Affinity,n_clusters=Lamda, n_components=Lamda, assign_labels="discretize",
                                        random_state=0)
        ax[idx].scatter(X[Cluster_time[idx],:,0],X[Cluster_time[idx],:,1], c=clustering, cmap='tab10')
        ax[idx].set_title('Clusterig at time {}'.format(i))

    plt.savefig('{}_Clustering for Lamda = {}'.format(DS_Name, Lamda))


def SptDM_EigFunc(X, u, EigFuncIdx ,EigFuncTime,DS_Name):
    fix, ax = plt.subplots(3, 1, figsize=(19,10))
    for idx, i in enumerate(EigFuncIdx):
        ax[idx].scatter(X[EigFuncTime,:,0],X[EigFuncTime,:,1], c=u[:,i], cmap='rainbow')
        ax[idx].set_title('Eigentfunction {} at time {}'.format(i,EigFuncTime))

    plt.savefig('{} Eigentfunctions for Time = {}'.format(DS_Name, EigFuncTime))

