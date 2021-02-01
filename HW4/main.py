from HW4 import Q1
from os.path import join as pjoin
import os

if __name__ == '__main__':
    #### ----- Section 1 ----- ####
    # path = r'PicSource\cameraman.png'
    # resize_image = 128 #Resizing the image depend on computational resources
    # Q1.Sec1(path, resize_image)

    #### ----- Section 2 ----- ####
    r = 1
    Sigma = 1
    s = 2
    eigenVec_idx = [1,2,4,10]
    tau = [0.2,0.5,0.8,1.5,3,5]

    path_sec2 = r'Meshes\toilet_0003.off'
    # # path_sec2 = r'Meshes\sofa_0003.off'
    Q1.Sec2(path_sec2, r, Sigma, s, eigenVec_idx, tau)