import matplotlib.pyplot as plt
import numpy as np
## For server compatability
import sys
sys.path.append("..")
##
from Final_Project import DataGeneration
from Final_Project import SpaceTimeDiffMatrix


if __name__ == '__main__':
    Data_name = r'DoubleGyreFlowCorArr'
    Diff_Space_time = DataGeneration.nd_ap_gendata(Data_name,load = True)

    eps = [0.0002,0.005,0.001,0.002,0.004]
    r = 1

    for i in eps:
        print('Runing eps {}'.format(i))
        ll, u, SptDM = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(Diff_Space_time, r, i, sparse=True)
        np.savez('EigVector_Function_SptDM_epsilon{}.npz'.format(eps), [ll,u,SptDM], Diff_Space_time.astype('float32'))

    print('Finish')