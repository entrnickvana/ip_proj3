import code
import os
import skimage
from skimage import io
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
#from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float, img_as_ubyte, util
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema
from skimage.exposure import histogram
from cantor import *
#from sample_signals import *
from canvas_funcs import *


### Section 1
# 1) FFTs
# Use the built-in FFT functions in python (i.e., numpy) to compute the Fourier
# transform of some images and show their power spectrum.  You should organize
# the frequencies so that zero is in the middle of the FFT result/images.  You
# might need to use a log to see the power clearly.   Show results on several
# images and comment on what you see in both domains.  Implement a lower pass
# filter in the Fourier domain and do the inverse FFT to show the effects on several images.

# Experimenting in Fourier domain with low pass filters
def sect1_tests():

    # color images
    img0 = color2grey(io.imread('f18_super_hornet.png'))
    img1 = color2grey(io.imread('iceland.png'))
    img7 = color2grey(io.imread('houndog1.png'))    
    
    # Gray Images
    clouds = io.imread('clouds.png')
    img2 = io.imread('cell_images/cell_images/0001.002.png')
    img4 = io.imread('cell_images/cell_images/0001.004.png')
    img5 = io.imread('cell_images/cell_images/0001.005.png')
    img6 = io.imread('cell_images/cell_images/0001.006.png')


    # Low Pass Gaussian, Sigma = .5
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(img7, 0, .5)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.5')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.5)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.5)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()

    

    # Low Pass Gaussian, Sigma = .05
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(img7, 0, .1)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.1')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.1)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.1)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()

    # Low Pass Gaussian, Sigma = .05
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(img7, 0, .05)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.05')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.05)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.05)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()

    
    # Low Pass Gaussian, Sigma = .5
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(clouds, 0, .5)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.5')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.5)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.5)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()

    

    # Low Pass Gaussian, Sigma = .05
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(clouds, 0, .1)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.1')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.1)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.1)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()

    # Low Pass Gaussian, Sigma = .05
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(clouds, 0, .05)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.05')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.05)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.05)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()
    
    # Low Pass Gaussian, Sigma = .5
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(img1, 0, .5)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.5')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.5)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.5)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()

    # Low Pass Gaussian, Sigma = .5
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(img1, 0, .1)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.1')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.1)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.1)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()
    

    # Low Pass Gaussian, Sigma = .05
    img_filt, FILT, I, IMG_FILT = lo_pass_gauss(img1, 0, .05)
    plt.subplot(2,2,1)
    plt.title('Filtered Image, sigma = 0.05')            
    plt.imshow(img_filt, cmap='gray')
    plt.subplot(2,2,2)
    plt.title('FFT image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,2,3)
    plt.title('Fourier Domain Gaussian Filter (Magnitude Logscale, sigma = 0.05)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,2,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale, sigma = 0.05)')            
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.show()
    


# Section 4 Test cases
def sect4_tests():

    # Import images for section 4 testing
    clouds = io.imread('clouds.png')
    i0 = io.imread('cell_images/cell_images/0001.000.png')
    i1 = io.imread('cell_images/cell_images/0001.001.png')
    i2 = io.imread('cell_images/cell_images/0001.002.png')
    i4 = io.imread('cell_images/cell_images/0001.004.png')
    i5 = io.imread('cell_images/cell_images/0001.005.png')
    i6 = io.imread('cell_images/cell_images/0001.006.png')
    
    plt.imshow(clouds, cmap='gray')
    plt.show()
    
    # Partition Teste images from clouds.png
    s1,s2,s3, s4 = partition_imag4(clouds, 128)

    print(s1.shape)
    print(s2.shape)
    print(s3.shape)
    print(s4.shape)
    
    # Correct asymetries in shape
    s1 = s1[0:628, 0:628]
    s2 = s2[0:628, 0:628]
    s3 = s3[0:628, 0:628]
    s4 = s4[0:628, 0:628]
    
    # Put in array to pass to mosaic function
    cloud_imgs = [s1, s2, s3, s4]
    
    partition_imag4_print(clouds, cloud_imgs)
    s1.shape
    s2.shape
    s3.shape
    s4.shape
    
    
    clouds_reconstructed = mosaic_core(cloud_imgs)
    plt.imshow(clouds_reconstructed, cmap='gray')
    plt.savefig('clouds_mosiac.png')
    
    imgs1 = [i0, i1, i2, i4, i5, i6]
    test1 = test_img((256, 256))
    
    canvas = mosaic_core(imgs1)
    plt.savefig('cells_mosiac.png')
    plt.imshow(canvas, cmap='gray')
    plt.show()



# Run code here
sect4_tests()
sect1_tests()


    
