import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import os
from HW3 import Q1

if __name__ == '__main__':
    # ######################################################
    # #############------ Section 1 ------##################
    # source = (383, 814)
    # target = (233, 8)
    # Q1.section1(r'Resources\maze.png', source, target)
    #
    # ######################################################
    # #############------ Section 2 ------##################
    # source_pool = (0,0)
    # target_pool = (499,399)
    # Q1.section2(pjoin(os.getcwd(), 'Resources'),source_pool,target_pool)
    # # Flip to verify this is indeed correct # #
    # Q1.section2(pjoin(os.getcwd(), 'Resources'),target_pool,source_pool)

    # ######################################################
    # #############------ Section 4 ------##################
    # err_MDS = Q1.section4(r'Resources\tr_reg_000.ply', n_dim=3, embedding='MDS')
    # err_SMDS = Q1.section4(r'Resources\tr_reg_000.ply', n_dim=3, embedding='Spherical MDS')
    #
    # err_MDS = Q1.section4(r'Resources\tr_reg_001.ply', n_dim=3, embedding='MDS')
    # err_SMDS = Q1.section4(r'Resources\tr_reg_001.ply', n_dim=3, embedding='Spherical MDS')
    #
    # ######################################################
    # #############------ Section 5 ------##################
    # Q1.section5(r'Resources\tr_reg_000.ply',(1000, 2000, 4000), 500)
    # Q1.section5(r'Resources\tr_reg_001.ply', (1000, 2000, 4000), 500)

    plt.show()
    print('Finish')
