import matplotlib.pyplot as plt
import numpy as np
## For server compatability
import sys
sys.path.append("..")
##
from Final_Project import DataGeneration
from Final_Project import SpaceTimeDiffMatrix


if __name__ == '__main__':
    path = r'BickleyJet\BickleyJet' #Path to CSV files
    Diff_Space_time = DataGeneration.BickleyJet_DG(path, load = False)

    eps = [0.01,0.02,0.05,0.1,0.2,0.5]
    r = 2

    for i in eps:
        print('Runing eps {}'.format(i))
        EigenVal, EigenVec, SptDM, L_EigenVal, L_EigenVec, L_SptDM = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(Diff_Space_time, r, i, sparse=True)
        np.savez('EigVector_Function_SptDM_epsilon{}.npz'.format(i), EigenVal, EigenVec, SptDM, L_EigenVal, L_EigenVec, L_SptDM)
