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


def multi_plot(X, k, eps,data='NotGetDataType'):
    plt.figure()
    marker = ['+', '<', '^','2','+','x']
    for idx in range(len(X[0,:])):
        plt.scatter(np.arange(1,k,1), X[1:,idx], marker= marker[idx], label='$\epsilon$ = {}'.format(eps[idx]))

    plt.xlabel('n')
    plt.ylabel(r'$\lambda_n$')
    plt.legend(loc=3)
    plt.savefig(r'Figs\{}EigenValues'.format(data))

def SptDM_SC(X, Affinity, Cluster_time, Lamda, DS_Name, eps):
    eps = str(eps).split('.')[1]
    fix, ax = plt.subplots(len(Cluster_time), 1, figsize=(19,10), squeeze=False)
    for idx, i in enumerate(Cluster_time):
        Affinity = np.array(Affinity)
        clustering = spectral_clustering(Affinity,n_clusters=Lamda, n_components=Lamda, assign_labels="discretize", random_state=0)
        ax[idx,0].scatter(X[Cluster_time[idx],:,0],X[Cluster_time[idx],:,1], c=clustering, cmap='jet')
        ax[idx,0].set_title('Clusterig at time {}'.format(i))

    plt.savefig(r'Figs\{}_Clustering for Lamda = {} Eps = {} Time = {}'.format(DS_Name, Lamda, eps, Cluster_time))

    return clustering

def SptDM_EigFunc(X, u, EigFuncIdx ,EigFuncTime,DS_Name, eps):
    eps = str(eps).split('.')[1]
    fix, ax = plt.subplots(len(EigFuncIdx), 1, figsize=(19,10), squeeze=False)
    for idx, i in enumerate(EigFuncIdx):
        ax[idx,0].scatter(X[EigFuncTime,:,0],X[EigFuncTime,:,1], c=u[:,i], cmap='jet')
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

    ax.set_xlabel('$\Xi_2$')
    ax.set_ylabel('$\Xi_4$')
    ax.set_zlabel('$\Xi_5$')

    plt.savefig(r'Figs\3D embbeding {} Eigentfunctions {} colored clusters'.format(DS_Name,Lamda))


# Special for missing data
def SptDM_SC_Mdata(MX, MAffinity, X, Affinity, Cluster_time, Lamda, DS_Name, eps):
    eps = str(eps).split('.')[1]
    fix, ax = plt.subplots(len(Cluster_time), 1, figsize=(19,10), squeeze=False)
    for idx, i in enumerate(Cluster_time):
        Affinity = np.array(Affinity)
        clustering = spectral_clustering(Affinity,n_clusters=Lamda, n_components=Lamda, assign_labels="discretize",
                                        random_state=0)
        MAffinity = np.array(MAffinity)
        Mclustering = spectral_clustering(MAffinity,n_clusters=Lamda, n_components=Lamda, assign_labels="discretize",
                                        random_state=0)
        ax[idx,0].scatter(X[Cluster_time[idx],:,0],X[Cluster_time[idx],:,1], c=clustering)
        cmap = plt.cm.get_cmap('cool', Lamda)
        ax[idx, 0].scatter(MX[Cluster_time[idx], :, 0], MX[Cluster_time[idx], :, 1], c=Mclustering, cmap=cmap)
        ax[idx,0].set_title('Clustering at time {}'.format(i))

    plt.savefig(r'Figs\{}_Clustering for Lamda = {} Eps = {} Time = {} with missing Data layer'.format(DS_Name, Lamda, eps, Cluster_time))

    return clustering

def SptDM_Kmean_SC(X, EigenVec, Cluster_time, Lamda, DS_Name, eps):
    eps = str(eps).split('.')[1]
    fix, ax = plt.subplots(len(Cluster_time), 1, figsize=(19,10), squeeze=False)
    for idx, i in enumerate(Cluster_time):
        kmeans = KMeans(n_clusters=Lamda, random_state=0).fit(EigenVec[:, :Lamda])
        clustering = kmeans.labels_
        ax[idx,0].scatter(X[Cluster_time[idx],:,0],X[Cluster_time[idx],:,1], c=clustering)
        ax[idx,0].set_title('Clustering at time {}'.format(i))

    plt.savefig(r'Figs\Kmeans_{}_Clustering for Lamda = {} Eps = {} Time = {}'.format(DS_Name, Lamda, eps, Cluster_time))

    return clustering

# Special for missing data
def SptDM_Kmeans_SC_Mdata(MX, MEigenVec, X, EigenVec, Cluster_time, Lamda, DS_Name, eps):
    eps = str(eps).split('.')[1]
    fix, ax = plt.subplots(len(Cluster_time), 1, figsize=(19,10), squeeze=False)
    for idx, i in enumerate(Cluster_time):
        kmeans = KMeans(n_clusters=Lamda, random_state=0).fit(EigenVec[:, :Lamda])
        clustering = kmeans.labels_

        kmeans = KMeans(n_clusters=Lamda, random_state=0).fit(MEigenVec[:, :Lamda])
        Mclustering = kmeans.labels_

        ax[idx, 0].scatter(X[Cluster_time[idx], :, 0], X[Cluster_time[idx], :, 1], c=clustering)
        cmap = plt.cm.get_cmap('cool', Lamda)
        ax[idx, 0].scatter(MX[Cluster_time[idx], :, 0], MX[Cluster_time[idx], :, 1], c=Mclustering, cmap=cmap)
        ax[idx, 0].set_title('Clustering at time {}'.format(i))

    plt.savefig(r'Figs\{}_Clustering for Lamda = {} Eps = {} Time = {} with missing Data layer'.format(DS_Name, Lamda, eps, Cluster_time))

    return clustering