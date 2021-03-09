import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
from PIL import Image
import os
## For server compatability
import sys
sys.path.append("..")
##
from Final_Project import DataGeneration
from Final_Project import SpaceTimeDiffMatrix
from Final_Project import common


def BickleyJet(path, load):
    """
    Calculating BickleyJet Section
    :param path: Path to CSV files
    :param load: Whether or not loading the dataset or calculating from scratch
    :return: Returining BickleyJet results
    """
    X = DataGeneration.BickleyJet_DG(path, load = load)
    SaveQeps = True

    ## An option
    ## --- Create GIF to validate the data --- ##
    # for t in np.arange(Diff_Space_time.shape[0]):
    #     plt.figure(figsize=(10,5))
    #     plt.scatter(Diff_Space_time[t, :, 0], Diff_Space_time[t, :, 1], c= np.arange(12000), cmap='hot')
    #     plt.savefig('BickleyJetImg\BickleyJet_{}'.format(t))
    # common.create_gif(r'BickleyJetImg','BickleyJetGif.gif')

    eps = [0.02,0.01,0.05,0.1,0.2,0.5]
    r = 2
    k = 21 #Including the zero order
    m = np.shape(X)[1]
    T = np.shape(X)[0]
    d = np.shape(X)[2]

    EigenVal = np.zeros([k])
    EigenVec = np.zeros([m,k])
    L_EigenVal = np.zeros([k])
    L_EigenVec = np.zeros([m,k])
    L_eps_EigenVal = np.zeros([k,len(eps)]) #Store the eigenvalues for different epsilons

    Cluster_time = np.array([5,20,35])
    EigFuncTime = 20
    Lamda = 9
    DS_Name = 'BickleyJet'

    # Calculate Qeps for different epsilon
    # Save Qeps for each epsilon if needed and plot the eigenvalues for L matrix - for all the epsilons
    for idx, i in enumerate(eps):
        print('Runing eps {}'.format(i))
        data = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(X, r, i,k=k,sparse=True)
        EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps = data
        L_eps_EigenVal[:,idx] = L_EigenVal
        if SaveQeps: #If not specify, all Qeps will be saved, in order to save specific Qeps, needs to run only wanted epsilon
            np.savez(r'QepsMatices\Q_data_{}_epsilon_{}.npz'.format(DS_Name, i), EigenVal, EigenVec, L_EigenVal, L_EigenVec,
                 Qeps)
    common.multi_plot(L_eps_EigenVal, k, eps, 'BicklyJetDS')

    ### --- Figures creation --- ###
    #Fig 10
    eps = 0.02;
    EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps = common.QepsLoading(DS_Name, eps)

    Cluster_time = np.array([5,20,35]); Lamda = 9
    # cluster = common.SptDM_SC(X,Qeps, Cluster_time, Lamda, DS_Name, eps) #Original DS, The Space time diffusion matrix!!! not partial time
    cluster = common.SptDM_Kmean_SC(X, EigenVec, Cluster_time, Lamda, DS_Name, eps) #K means is faster than spectral clustering algo

    EigFuncIdx = [1,2,3]; EigFuncTime = np.array([20])
    common.SptDM_EigFunc(X, EigenVec, EigFuncIdx, EigFuncTime, DS_Name, eps)

    #Fig 9 Right side
    embed_2 = Qeps @ EigenVec[:,1]
    embed_4 = Qeps @ EigenVec[:,3]
    embed_5 = Qeps @ EigenVec[:,4]

    common.embedding3D(embed_2,embed_4,embed_5,cluster, DS_Name,Lamda)
    # print('Finish BickleyJet')


if __name__ == '__main__':
    BickleyJetPath = r'BickleyJetDS\BickleyJet'  # Path to (bickley_x.csv, bickley_y.csv) CSV files
    ### Important note - For the first time running the load flag should be false ###
    BickleyJet(BickleyJetPath, load=True)
