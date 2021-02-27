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

    plt.savefig(r'Figs\{}EigenValues'.format(data))

def SptDM_SC(X, Affinity, Cluster_time, Lamda, DS_Name, eps):
    eps = str(eps).split('.')[1]
    fix, ax = plt.subplots(len(Cluster_time), 1, figsize=(19,10), squeeze=False)
    for idx, i in enumerate(Cluster_time):
        Affinity = np.array(Affinity)
        clustering = spectral_clustering(Affinity,n_clusters=Lamda, n_components=Lamda, assign_labels="discretize",
                                        random_state=0)
        ax[idx,0].scatter(X[Cluster_time[idx],:,0],X[Cluster_time[idx],:,1], c=clustering, cmap='tab10')
        ax[idx,0].set_title('Clusterig at time {}'.format(i))

    plt.savefig(r'Figs\{}_Clustering for Lamda = {} Eps = {} Time = {}'.format(DS_Name, Lamda, eps, Cluster_time))

    return clustering

def SptDM_EigFunc(X, u, EigFuncIdx ,EigFuncTime,DS_Name, eps):
    eps = str(eps).split('.')[1]
    fix, ax = plt.subplots(len(EigFuncIdx), 1, figsize=(19,10), squeeze=False)
    for idx, i in enumerate(EigFuncIdx):
        ax[idx,0].scatter(X[EigFuncTime,:,0],X[EigFuncTime,:,1], c=u[:,i], cmap='rainbow')
        ax[idx,0].set_title('Eigentfunction {} at time {} & Eps {}'.format(i,EigFuncTime,eps))

    plt.savefig(r'Figs\{} Eigentfunctions for Time = {} Eps = {}'.format(DS_Name, EigFuncTime,eps))


def QepsLoading(DS_Name, eps):
    data = np.load('QepsMatices\Q_data_{}_epsilon_{}.npz'.format(DS_Name,eps))
    EigenVal = data['arr_0']
    EigenVec = data['arr_1']
    L_EigenVal= data['arr_2']
    L_EigenVec= data['arr_3']
    Qeps = data['arr_4']

    return EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps

def embedding3D(embed1,embed2,embed3,clustering, DS_Name,Lamda):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embed1, embed2, embed3, marker="s", c=clustering,  cmap='tab10')

    plt.savefig(r'Figs\3D embbeding {} Eigentfunctions {} colored clusters'.format(DS_Name,Lamda))

