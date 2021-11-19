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

clouds = io.imread('clouds.png')
i0 = io.imread('cell_images/cell_images/0001.000.png')
i1 = io.imread('cell_images/cell_images/0001.001.png')
i2 = io.imread('cell_images/cell_images/0001.002.png')
i4 = io.imread('cell_images/cell_images/0001.004.png')
i5 = io.imread('cell_images/cell_images/0001.005.png')
i6 = io.imread('cell_images/cell_images/0001.006.png')

plt.imshow(clouds, cmap='gray')
plt.show()

s1,s2,s3, s4 = partition_imag4(clouds, 256)
s1 = s1[0:744, 0:744]
s2 = s2[0:744, 0:744]
s3 = s3[0:744, 0:744]
s4 = s4[0:744, 0:744]

cloud_imgs = [s1, s2, s3, s4]
partition_imag4_print(clouds, cloud_imgs)

clouds_reconstructed = mosaic_core(cloud_imgs)
plt.imshow(clouds_reconstructed, cmap='gray')
plt.show()
code.interact(local=locals())

imgs1 = [i0, i1, i2, i4, i5, i6]
test1 = test_img((256, 256))

canvas = mosaic_core(imgs1)
plt.imshow(canvas, cmap='gray')
plt.show()
code.interact(local=locals())
