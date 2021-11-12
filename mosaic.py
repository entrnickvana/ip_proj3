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

imgs = [i0, i1, i2, i4, i5, i6]

for ii in range(0, len(imgs)):
    for jj in range(0, len(imgs)):

      # Generate gaussian filter 'FILT'
      img_filt, FILT, I, IMG_FILT = lo_pass_gauss(i0, 0, .2)

      # Calculate Phase Correlation and Filter Phase correlation in Freq Dom.
      p_filt, ORIG_FILT, P, F, G = gen_corr_filt(imgs[ii], imgs[jj], FILT)

      # Print debug info like max, min, difference
      print('IMG 1 index: ', ii)
      print('IMG 2 index: ', jj)

      # Get max/min indices
      max_index = np.unravel_index(p_filt.argmax(), p_filt.shape)
      min_index = np.unravel_index(p_filt.argmin(), p_filt.shape)
      print('max_index: ', max_index)
      print('min_index: ', min_index)

      # Get max/min values
      pmax = p_filt[max_index[0], max_index[1]]
      pmin = p_filt[min_index[0], min_index[1]]
      delta = pmax-pmin
      print('Max val: ', pmax, '     Min val: ', pmin)
      print('Delta: ', delta)

      # Get statistics for potential thresholding
      pmean = np.mean(p_filt)
      pvar = np.var(p_filt)
      print('Mean: ', pmean)
      print('Var: ', pvar)

      # Create a histogram for thresholding if necassry
      flat = np.ndarray.flatten(p_filt)
      hist, edges = np.histogram(img_as_float(flat), bins=512)
      print_gen_corr_hist(imgs[ii], imgs[jj], p_filt, FILT, P, F, G , hist)

      # Determine if correlated

      # Create 3 x 3 canvas with image 1 in center
      canvas = np.zeros((3*imgs[ii].shape[0], 3*imgs[ii].shape[1]))

      # dim of pfilt
      y1_len = imgs[ii].shape[0]
      x1_len = imgs[ii].shape[1]
      
      y2_len = imgs[jj].shape[0]
      x2_len = imgs[jj].shape[1]

      # Center of canvas and top left corner of canvas
      cy_ctr = np.int(canvas.shape[0]/2)
      cx_ctr = np.int(canvas.shape[1]/2)
      
      # Image 1 corner/origin
      y1_orig = cy_ctr - np.int(y1_len/2)
      x1_orig = cx_ctr - np.int(x1_len/2)

      y2_ctr = max_index[0] + y1_orig
      x2_ctr = max_index[1] + x1_orig
      
      delta_y = y2_ctr + cy_ctr 
      delta_x = x2_ctr + cx_ctr
      #code.interact(local=locals())      

      y2_orig = y1_orig + delta_y
      x2_orig = x1_orig + delta_x

      # Insert image1 in canvas
      canvas[y1_len:2*y1_len, x1_len:2*x1_len] = imgs[ii]
      c1 = np.array(canvas)
      c1[y2_orig:y2_orig + 510, x2_orig:x2_orig + 510] = imgs[jj]
      plt.subplot(1,2,1)
      plt.imshow(canvas, cmap='gray')
      plt.subplot(1,2,2)
      plt.imshow(c1, cmap='gray')
      plt.show()
      code.interact(local=locals())      


               

      

      

  
  

  

  
