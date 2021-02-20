import matplotlib.pyplot as plt
import numpy as np
from Final_Project import DataGeneration
from Final_Project import SpaceTimeDiffMatrix


if __name__ == '__main__':
    Data_name = r'DoubleGyreFlowCorArr_Part50'
    Diff_Space_time = DataGeneration.nd_ap_gendata(Data_name,load = True)

    # Temp
    Diff_Space_time = Diff_Space_time.astype('float32')
    assert Diff_Space_time.dtype == 'float32'
    ##

    eps = 0.002
    r = 1
    ll, u, SptDM = SpaceTimeDiffMatrix.compute_SpaceTimeDMap(Diff_Space_time, r, eps, sparse=True);

    print('Finish')