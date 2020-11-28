import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
from PIL import Image
import imageio
import glob
import math

### Functions
def time_step_update(imag, j, k, dt=.1):
    """
    :param imag: Getting the next image
    :param j: Row index
    :param k: Column index
    :param dt: Step lplcn
    :return: weighted pixel per lplcn
    """
    lplcn = imag[j, k - 1] + imag[j - 1, k] + imag[j, k + 1] + imag[j + 1, k] - 4 * imag[j, k]
    return imag[j, k] + dt * lplcn



def create_gif(path,name):
    """
    :param path: Path where to load the images for GIF
    :param name: Gif name
    :return: saving batch of images into Gif
    """
    image_list = []
    for filename in glob.glob(path): #assuming gif
        im=Image.open(filename)
        image_list.append(im)

    images = []
    for filename in image_list:
        images.append(imageio.imread(filename.filename))
    imageio.mimsave(name, images)


im = Image.open('cameraman.png').convert('L') #Load image with grayscale
imarray = np.array(im)
imarray.shape



# Resizing the image to relevant format
cameraman = im.resize((256,256))
cameraman.show() #option to show the original image
width, height = cameraman.size

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:width, 0:height]
cameraman = np.array(cameraman)

cur_cameraman = cameraman.copy()

## Smooth the image using gradient desecnt method (Laplace operator)
# for L in range(0, 8):
#     next_img = np.zeros(cameraman.shape)
#     for k in range(1, cameraman.shape[0] - 1):
#         for j in range(1, cameraman.shape[1] - 1):
#             next_img[j, k] = time_step_update(cur_cameraman, j, k,0.08)
#     cur_cameraman = next_img
#     plt.figure(figsize=(5,5))
#     plt.title('Original Image Smooth by Gradient', fontsize=18)
#     plt.imshow(next_img, cmap=plt.cm.gray)
#     plt.savefig('Pics\Gradient\Step\Step_cameraman_{}'.format(L))
#
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.title('Original Surface Smooth by Gradient', fontsize=18)
#     surf = ax.plot_surface(xx, yy, np.abs(1 - next_img / 255), rstride=1,
#                            cstride=1, cmap=plt.cm.Greys,
#                            linewidth=0, antialiased=False)
#     ax.view_init(elev=80, azim=0)
#     plt.savefig('Pics\Gradient\Surface\Surface_cameraman_{}'.format(L))
# create_gif('Pics/Gradient/Step/*.png', 'PicSmoothByGradient.gif')
# create_gif('Pics/Gradient/Surface/*.png', 'SurfaceSmoothByGradient.gif')


#Smooth the picture using Fourier transform and gauusian kernel
#Multiply the image on the spectral domain with the gaussian kernel


def fourier_transform(image):
    """
    Computes the Fourier transform on the image and shifts it to the "typical"
    representation that is shown.
    """
    temp = np.fft.fft2(image)
    temp = np.fft.fftshift(temp)  # Shift!
    return temp

def inverse_fourier_transform(f_input_imag):
    """

    :param f_input_imag: Fourier image
    :return: Invers Fourier image (time domain)
    """
    temp = np.fft.ifftshift(f_input_imag)
    imag = np.fft.ifft2(temp)
    return np.abs(imag)**2 # Remove those imaginary values

def gaussian(image, kappa=1, t=.0001):
    # Gaussian Kernel for the Heat Equation in Fourier Space

    n, k = image.shape
    center_i = int(n / 2)
    center_j = int(k / 2)

    # This image is symmetric so we do one quadrant and fill in the others
    gaussian = np.zeros((n, k))
    for i in range(0, n - center_i):
        for j in range(0, k - center_j):
            temp = np.exp(-(i ** 2 + j ** 2) * kappa * t)
            gaussian[center_i + i, center_j + j] = temp
            gaussian[center_i - i, center_j + j] = temp
            gaussian[center_i + i, center_j - j] = temp
            gaussian[center_i - i, center_j - j] = temp

    return 1/(2*math.sqrt(math.pi*((t+1)*t))) * gaussian


cur_blurcameraman = cameraman.copy()

#
#Smooth the image using gradient desecnt method (Laplace operator)
for L in range(0, 8):

    Fourier_cameraman = fourier_transform(cur_blurcameraman) #Transfer the image to spectral domain
    gauss = gaussian(Fourier_cameraman) #create gaussian in spectral domain control the t step
    combination = np.multiply(gauss, Fourier_cameraman)  # Element wise in spectral domain
    BluredCameraman = inverse_fourier_transform(combination) #Transfer the spectral domain blured image to Real domain
    cur_blurcameraman = np.log10(BluredCameraman)

    plt.figure(figsize=(5, 5))
    plt.title('Original Image smooth by Gaussian', fontsize=18)
    plt.imshow(cur_blurcameraman, cmap=plt.cm.gray)
    plt.savefig('Pics\Gaussian\Step\Step_cameraman_{}'.format(L))
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Original Image', fontsize=18)
    surf = ax.plot_surface(xx, yy, cur_blurcameraman, rstride=1,
                           cstride=1, cmap=plt.cm.gray,
                           linewidth=0, antialiased=False)
    ax.view_init(elev=80, azim=0)
    plt.savefig('Pics\Gaussian\Surface\Surface_cameraman_{}'.format(L))

create_gif('Pics/Gaussian/Step/*.png', 'PicSmoothByGaussian.gif')
create_gif('Pics/Gaussian/Surface/*.png', 'SurfaceSmoothByGauusian.gif')
#
#
# plt.figure(figsize=(10, 20))
# plt.subplot(121)
# plt.title("Original (Grayscale)")
# plt.imshow(cameraman, cmap=plt.cm.gray)
# plt.subplot(122)
# plt.title("Gaussian Blurred")
# plt.imshow(BluredCameraman, cmap=plt.cm.gray)
#
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection='3d')
# plt.title('Original Image', fontsize=18)
# surf = ax.plot_surface(xx, yy, BluredCameraman, rstride=1,
#                        cstride=1, cmap=plt.cm.Greys,
#                        linewidth=0, antialiased=False)
# ax.view_init(elev=80, azim=0)


# show it
# plt.show()
print('Plotting and saving all images')

if __name__ == '__main__':
    print('Finish to compute')