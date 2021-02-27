import matplotlib.pyplot as plt
import numpy as np
## For server compatability
import sys
sys.path.append("..")
##
from Final_Project import DataGeneration
from Final_Project import SpaceTimeDiffMatrix
from Final_Project import common

if __name__ == '__main__':
    Data_name = r'DoubleGyreFlowCorArr'
    X = DataGeneration.nd_ap_gendata(Data_name, load = True) #Loading the data, X is the data with shape of T x m x 2
    # m is the number of trajectories or in other word is the number of "particles" which create the image

    # for t in np.arange(Diff_Space_time.shape[0]):
    #     plt.figure(figsize=(10,5))
    #     plt.scatter(Diff_Space_time[t, :, 0], Diff_Space_time[t, :, 1], c= np.arange(20000), cmap='hot')
    #     plt.savefig('DoubleGyreFlowImg\DoubleGyreFlow_{}'.format(t))
    # common.create_gif(r'DoubleGyreFlowImg', 'DoubleGyreFlowGif.gif')
    eps = [0.0002,0.0005,0.004,0.001,0.002]
    r = 2
    k = 11
    m = np.shape(X)[1]
    T = np.shape(X)[0]
    d = np.shape(X)[2]

    # EigenVal = np.zeros([k,len(eps)])
    # EigenVec = np.zeros([m,k,len(eps)])
    # L_EigenVal = np.zeros([k,len(eps)])
    # L_EigenVec = np.zeros([m,k,len(eps)])

    EigenVal = np.zeros([k])
    EigenVec = np.zeros([m,k])
    L_EigenVal = np.zeros([k])
    L_EigenVec = np.zeros([m,k])
    L_eps_EigenVal = np.zeros([k,len(eps)]) #Store the eigenvalues for different epsilons


    Cluster_time = np.array([0,T-1])
    EigFuncTime = np.array([0])
    Lamda = 3
    DS_Name = 'DoubleGyro'

    # for idx, i in enumerate(eps):
    #     print('Runing eps {}'.format(i))
    #     data = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(X, r, i, Cluster_time, Lamda,EigFuncTime, DS_Name,
    #                                                      k=k, sparse=True)
    #     EigenVal, EigenVec, L_EigenVal, L_EigenVec , Qeps = data
    #     L_eps_EigenVal[:, idx] = L_EigenVal
    #     np.savez(r'QepsMatices\Q_data_{}_epsilon_{}.npz'.format(DS_Name,i), EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps)
    # common.multi_plot(L_EigenVal, k, 'DoubleGyreFlow')


    # ### --- Figures creation --- ###
    # #Fig 5
    # eps = 0.0002;
    # EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps = common.QepsLoading(DS_Name, eps)
    #
    # Cluster_time = np.array([0]); Lamda = 3
    # common.SptDM_SC(X,Qeps, Cluster_time, Lamda, DS_Name, eps) #Original DS, The Space time diffusion matrix!!! not partial time
    #
    # Cluster_time = np.array([196]); Lamda = 3
    # common.SptDM_SC(X,Qeps, Cluster_time, Lamda, DS_Name, eps) #Original DS, The Space time diffusion matrix!!! not partial time
    #
    # #Fig 6
    # eps = 0.0002;
    # EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps = common.QepsLoading(DS_Name, eps)
    #
    # EigFuncIdx = [2]; EigFuncTime = np.array([0])
    # common.SptDM_EigFunc(X, EigenVec, EigFuncIdx, EigFuncTime, DS_Name, eps)
    #
    # Cluster_time = np.array([0]); Lamda = 2
    # common.SptDM_SC(X,Qeps, Cluster_time, Lamda, DS_Name, eps)
    #
    # ######
    # eps = 0.004;
    # EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps = common.QepsLoading(DS_Name, eps)
    #
    # EigFuncIdx = [2]; EigFuncTime = np.array([0])
    # common.SptDM_EigFunc(X, EigenVec, EigFuncIdx, EigFuncTime, DS_Name, eps)
    #
    # Cluster_time = np.array([0]); Lamda = 2
    # common.SptDM_SC(X,Qeps, Cluster_time, Lamda, DS_Name, eps)
    #
    # # Fig 7
    # eps = 0.0005;
    # ##### --- !!!!
    # # In order to get the results, needs to change the number of components (eigenvectors)
    # # which compose the Lamda clusters
    # ##### --- !!!!
    # EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps = common.QepsLoading(DS_Name, eps)
    # Cluster_time = np.array([0]); Lamda = 4
    # common.SptDM_SC(X, Qeps, Cluster_time, Lamda, DS_Name, eps)
    #
    # eps = 0.004;
    # EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps = common.QepsLoading(DS_Name, eps)
    #
    # Cluster_time = np.array([0]); Lamda = 4
    # common.SptDM_SC(X, Qeps, Cluster_time, Lamda, DS_Name, eps)

    print('Finish DoubleGyreFlow')
    print('Starting incomplete data case')

    np.random.seed(32)
    eps = 0.01
    # DS_Name = 'DoubleGyro_ReducedSamples'
    m_reduced = 500
    idx = np.random.randint(0, m, m_reduced)
    X_rand = X[:,idx,:] #Reduce sample data

    for i in np.arange(T):
        idx_NaN = np.random.randint(0, m_reduced, [int(0.8 * m_reduced)])
        X_rand[i,idx_NaN,:] = np.nan
    print('Creating desampling data')

    data = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(X_rand, r, eps, Cluster_time, Lamda,EigFuncTime, DS_Name,
                                                     k=k, sparse=True)
    EigenVal, EigenVec, L_EigenVal, L_EigenVec , Qeps = data
    # np.savez(r'QepsMatices\Q_data_{}_epsilon_{}.npz'.format(DS_Name,i), EigenVal, EigenVec, L_EigenVal, L_EigenVec, Qeps)
    Cluster_time = np.array([0]); Lamda = 4
    common.SptDM_SC(X_rand, Qeps, Cluster_time, Lamda, DS_Name, eps)



    print('Finish')