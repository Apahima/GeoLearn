from HW3 import Q1
from os.path import join as pjoin
import os

if __name__ == '__main__':
    # ######################################################
    # #############------ Section 1 ------##################
    source = (383, 814)
    target = (233, 8)
    # # Source to Target
    Q1.Sec1(r'Resources\maze.png', target, source)
    # # Target to Source
    Q1.Sec1(r'Resources\maze.png', source, target)

    # ######################################################
    # #############------ Section 2 ------##################
    # source_pool = (0,0)
    # target_pool = (499,399)
    # Q1.Sec2(pjoin(os.getcwd(), 'Resources'),source_pool,target_pool)
    # Flip to verify this is indeed correct method # #
    # Q1.Sec2(pjoin(os.getcwd(), 'Resources'),target_pool,source_pool)

    # ######################################################
    # #############------ Section 4 ------##################
    # err_MDS_000 = Q1.Sec4(r'Resources\tr_reg_000.ply', n_dim=3, embedding='MDS')
    # err_SMDS_000 = Q1.Sec4(r'Resources\tr_reg_000.ply', n_dim=3, embedding='Spherical MDS')
    # #
    # err_MDS_001 = Q1.Sec4(r'Resources\tr_reg_001.ply', n_dim=3, embedding='MDS')
    # err_SMDS_001 = Q1.Sec4(r'Resources\tr_reg_001.ply', n_dim=3, embedding='Spherical MDS')

    # ######################################################
    # #############------ Section 5 ------##################
    # Q1.Sec5(r'Resources\tr_reg_000.ply',(1000, 2000, 3000), 234)
    # Q1.Sec5(r'Resources\tr_reg_001.ply', (1000, 2000, 3000), 234)

    # print('err_MDS_000 = {}'.format(err_MDS_000))
    # print('err_SMDS_000 = {}'.format(err_SMDS_000))
    # print('err_MDS_001 = {}'.format(err_MDS_001))
    # print('err_SMDS_001 = {}'.format(err_SMDS_001))
    print('Finish')
