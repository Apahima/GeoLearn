#Reference https://github.com/jollybao/LCS

import os
import pylab as plt
import numpy as np
import matplotlib.animation as animation
from scipy.integrate import odeint
from itertools import product
from tqdm import tqdm
import csv
from numpy import genfromtxt

def nd_ap_gendata(file_name,load = False):
    """
    Data generation for doubly Gyre following the paper
    :param file_name: The file name for save the matrix
    :param load: Whether or not reload the saved matrix, If True no need to recomputed the matrix and can be loaded
    :return: Return the raw data for specific dataset
    """
    # constants
    p = np.pi
    A = 0.25
    Alpha = 0.25  # Alpha
    w = 2 * p
    partition = 100
    T = 201

    delta = 0.0001
    dt = 0.1

    # time points
    t = np.linspace(0, 20, T)

    # X, Y = plt.meshgrid(np.arange(0, 2, 1 / partition), np.arange(0, 1, 1 / partition))
    X = np.arange(0, 2, 1 / partition)
    Y = np.arange(0, 1, 1 / partition)
    Comb_cor = product(X, Y) #Creating all combination of X and Y

    def model(z,t):
        x,y = z
        dfx = (2 * Alpha * np.sin(w * t) * (x - 1) + 1)  # df/dx

        dxdt = -p * A * np.sin(p * f(x, t)) * np.cos(p * y)
        dydt = p * A * np.cos(p * f(x,t)) * np.sin(p * y) * dfx
        return [dxdt, dydt]

    def f(x, t):
        temp = Alpha * np.sin(w * t) * x ** 2 + (1 - 2 * Alpha * np.sin(w * t)) * x
        return temp

    # y = 0.3
    # z0 = (0.2,0.3)
    if load:
        Diff_Space_time = np.load('{}.npy'.format(file_name))
        print('Double Gyre loaded from saved file array')
    else:
        Diff_Space_time = np.zeros([T,X.shape[0] * Y.shape[0],2])
        for idx, i in tqdm(enumerate(Comb_cor)):
            Diff_Space_time[:,idx,:] = odeint(model,i,t)

        np.save(file_name, Diff_Space_time.astype('float32'))
        print('Data Generation Done, Array saved')


    return Diff_Space_time

def BickleyJet_DG(path, load = False):
    """
    Loading Bickley dataset and \ or save it
    :param path: Path frpm where to load the data
    :param load: Whether or not to load the data
    :return: Return Bickley Jet data
    """
    if load:
        BickleyJet = np.load('BickleyJet.npy')
        print('BickleyJet loaded from saved file array')
    else:
        BickleyJet_x = np.expand_dims(genfromtxt(os.path.join(path, 'bickley_x.csv'), delimiter=',').T, 2)
        BickleyJet_y = np.expand_dims(genfromtxt(os.path.join(path, 'bickley_y.csv'), delimiter=',').T, 2)

        BickleyJet = np.concatenate((BickleyJet_x, BickleyJet_y), 2)
        np.save('BickleyJet', BickleyJet)
        print('Data Generation Done, Array saved')

    return BickleyJet

if __name__ == '__main__':
    # nd_ap_gendata()

    BickleyJet_DG(r'BickleyJet\BickleyJet')
    print('Finish')