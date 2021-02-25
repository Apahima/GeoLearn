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

# def Psi(t,x,y):
#     U0 = 5.414
#     L = 1.77
#     re = 6.371
#
#     c = np.array([0.1446, 0.2053, 0.4561]) * U0
#     A = np.array([0.075, 0.4, 0.3])
#
#     xx, yy = np.reshape(x,[200,60]), np.reshape(y,[200,60])
#     for i in np.arange(c.shape[0]):
#         k_n = (2*i) / re
#         Sig = A[i]* np.cos(k_n * (xx -c[i]*t))
#
#     P = - U0*L*np.tanh(yy/L) + U0*L* (1/np.cosh(yy/L))**2 * Sig
#
#     plt.imshow(P.T)
#     plt.savefig('BickleyJet_Img\BickleyJet_{}'.format(t))
#     return P



if __name__ == '__main__':
    path = r'BickleyJet\BickleyJet' #Path to CSV files
    Diff_Space_time = DataGeneration.BickleyJet_DG(path, load = True)

    ## --- Create GIF to validate the data --- ##
    # for t in np.arange(Diff_Space_time.shape[0]):
    #     plt.figure(figsize=(10,5))
    #     plt.scatter(Diff_Space_time[t, :, 0], Diff_Space_time[t, :, 1], c= np.arange(12000), cmap='hot')
    #     plt.savefig('BickleyJetImg\BickleyJet_{}'.format(t))
    # common.create_gif(r'BickleyJetImg','BickleyJetGif.gif')
    eps = [0.02,0.01,0.05,0.1,0.2,0.5]
    r = 2
    k = 21 #Including the zero order
    m = np.shape(Diff_Space_time)[1]
    T = np.shape(Diff_Space_time)[0]
    d = np.shape(Diff_Space_time)[2]

    EigenVal = np.zeros([k,len(eps)])
    EigenVec = np.zeros([m,k,len(eps)])
    L_EigenVal = np.zeros([k,len(eps)])
    L_EigenVec = np.zeros([m,k,len(eps)])

    Cluster_time = np.array([5,20,35])
    EigFuncTime = 20
    Lamda = 9
    DS_Name = 'BickleyJet'

    for idx, i in enumerate(eps):
        print('Runing eps {}'.format(i))
        data = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(Diff_Space_time, r, i, Cluster_time, Lamda,EigFuncTime, DS_Name
        ,k=k,sparse=True)
        EigenVal[:,idx], EigenVec[:,:,idx], L_EigenVal[:,idx], L_EigenVec[:,:,idx] = data
    np.savez('data_BickleyJet.npz', EigenVal, EigenVec, L_EigenVal, L_EigenVec)
    common.multi_plot(L_EigenVal, k, 'BicklyJetDS')
    print('Finish BickleyJet')