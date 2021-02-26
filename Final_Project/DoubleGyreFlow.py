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
    Diff_Space_time = DataGeneration.nd_ap_gendata(Data_name,load = True)

    # for t in np.arange(Diff_Space_time.shape[0]):
    #     plt.figure(figsize=(10,5))
    #     plt.scatter(Diff_Space_time[t, :, 0], Diff_Space_time[t, :, 1], c= np.arange(20000), cmap='hot')
    #     plt.savefig('DoubleGyreFlowImg\DoubleGyreFlow_{}'.format(t))
    # common.create_gif(r'DoubleGyreFlowImg', 'DoubleGyreFlowGif.gif')
    eps = [0.0002,0.0005,0.001,0.002,0.004]
    r = 2
    k = 11
    m = np.shape(Diff_Space_time)[1]
    T = np.shape(Diff_Space_time)[0]
    d = np.shape(Diff_Space_time)[2]

    EigenVal = np.zeros([k,len(eps)])
    EigenVec = np.zeros([m,k,len(eps)])
    L_EigenVal = np.zeros([k,len(eps)])
    L_EigenVec = np.zeros([m,k,len(eps)])

    Cluster_time = np.array([0,T-1])
    EigFuncTime = np.array([0])
    Lamda = 3
    DS_Name = 'DoubleGyro'

    for idx, i in enumerate(eps):
        print('Runing eps {}'.format(i))
        data = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(Diff_Space_time, r, i, Cluster_time, Lamda,EigFuncTime, DS_Name,
                                                         k=k, sparse=True)
        EigenVal[:, idx], EigenVec[:, :, idx], L_EigenVal[:, idx], L_EigenVec[:, :, idx] = data
    np.savez('data_DoubleGyreFlow.npz', EigenVal, EigenVec, L_EigenVal, L_EigenVec)
    common.multi_plot(L_EigenVal, k, 'DoubleGyreFlow')
    print('Finish DoubleGyreFlow')


    # for i in eps:
    #     npzfile = np.load('EigVector_Function_SptDM_epsilon{}.npz'.format(i))
    #     ll = npzfile['arr_0']
    #     u = npzfile['arr_1']
    #     SptDM = npzfile['arr_2']
    #     print('Loading Epsilons')

    print('Finish')