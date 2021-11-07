import code
import os
import skimage
from skimage import io

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

i0 = io.imread('cell_images/cell_images/0001.000.png')
i1 = io.imread('cell_images/cell_images/0001.001.png')
i2 = io.imread('cell_images/cell_images/0001.002.png')
i4 = io.imread('cell_images/cell_images/0001.004.png')
i5 = io.imread('cell_images/cell_images/0001.005.png')
i6 = io.imread('cell_images/cell_images/0001.006.png')

F_i0 = np.log(abs(np.fft.fft2(i0))**2)
plt.subplot(2,1,1)
plt.imshow(i0, cmap='gray')
plt.subplot(2,1,2)
plt.imshow(F_i0, cmap='gray')
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
