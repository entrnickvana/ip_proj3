import code
import os
import skimage
from skimage import io


# Here is the link to the article I read, also see pwr_exp.py for what is done in the article
# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/

import matplotlib.pyplot as plt
import numpy as np
#from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float, img_as_ubyte, util
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema
from skimage.exposure import histogram
from cantor import *
from sample_signals import *
#from skimage.filter import edges

clouds = io.imread('clouds.png')
i0 = io.imread('cell_images/cell_images/0001.000.png')
i1 = io.imread('cell_images/cell_images/0001.001.png')
i2 = io.imread('cell_images/cell_images/0001.002.png')
i4 = io.imread('cell_images/cell_images/0001.004.png')
i5 = io.imread('cell_images/cell_images/0001.005.png')
i6 = io.imread('cell_images/cell_images/0001.006.png')

gs_kern = 

F_i0_orig = np.fft.fft2(i0)
F_i0 = abs(np.fft.fft2(i0))
F_i0_pwr = np.log(F_i0**2)
x_len1 = np.shape(i0)[0]
y_len1 = np.shape(i0)[1]
lo_filt = lo_p(x_len1, 128)
F_i0_filt = np.multiply(F_i0_orig, lo_filt)
F_i0_pwr_filt = np.log(abs(F_i0_filt)**2)
i0_filt = np.fft.ifft(F_i0_filt)


plt.subplot(3,2,1)
plt.imshow(i0, cmap='gray')
plt.subplot(3,2,2)
plt.imshow(F_i0, cmap='gray')
plt.subplot(3,2,3)
plt.imshow(lo_filt, cmap='gray')
plt.subplot(3,2,4)
plt.imshow(F_i0_pwr, cmap='gray')
plt.subplot(3,2,4)
plt.imshow(F_i0_pwr, cmap='gray')
plt.subplot(3,2,5)
plt.imshow(F_i0_pwr_filt, cmap='gray')
plt.subplot(3,2,6)
plt.imshow(abs(i0_filt), cmap='gray')
plt.show()


exit()

d1 = delta1(128)
#d1 = lo_p(128, 3)
f1 = np.abs(np.fft.fft2(d1))




circk = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0],          
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0]
                  ])


plt.subplot(3,1,1)
plt.imshow(d1, cmap='gray')
plt.subplot(3,1,2)
plt.imshow(f1, cmap='gray')
plt.subplot(3,1,3)
plt.imshow(circ_filt, cmap='gray')
plt.show()
