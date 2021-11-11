import code
import os
import skimage
from skimage import io
from scipy import fftpack

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
#from numpy import unravel_index
#from skimage.filter import edges

clouds = io.imread('clouds.png')
i0 = io.imread('cell_images/cell_images/0001.000.png')
i1 = io.imread('cell_images/cell_images/0001.001.png')
i2 = io.imread('cell_images/cell_images/0001.002.png')
i4 = io.imread('cell_images/cell_images/0001.004.png')
i5 = io.imread('cell_images/cell_images/0001.005.png')
i6 = io.imread('cell_images/cell_images/0001.006.png')

img_arr=[i0, i1, i2, i4, i5, i6]

## plot ideal
#img_filt, FILT, I, IMG_FILT = lo_pass_gauss(i0, 0, .2)
#p_filt, ORIG_FILT, P, F, G = gen_corr_filt(i0, i1, FILT)
#print_gen_corr(i0, i1, p_filt, FILT, P, F, G )
#max_index = np.unravel_index(p_filt.argmax(), p_filt.shape)
#min_index = np.unravel_index(p_filt.argmin(), p_filt.shape)
#
#
#code.interact(local=locals())  

for ii in range(0, len(img_arr)):  
  # plot ideal
  img_filt, FILT, I, IMG_FILT = lo_pass_gauss(i0, 0, .2)
  p_filt, ORIG_FILT, P, F, G = gen_corr_filt(i0, img_arr[ii], FILT)
  print_gen_corr(i0, img_arr[ii], p_filt, FILT, P, F, G )
  print('index: ', ii)
  max_index = np.unravel_index(p_filt.argmax(), p_filt.shape)
  print('max_index: ', max_index)
  min_index = np.unravel_index(p_filt.argmin(), p_filt.shape)
  print('min_index: ', min_index)
  print('Max val: ', p_filt.argmax(), '     Min val: ', p_filt.argmin())
  print('Delta: ', p_filt.argmax()-p_filt.argmin())
  hst, edges = np.histogram(p_filt, bins=256)
  plt.plot(edges, hst)
  
  

  

  
