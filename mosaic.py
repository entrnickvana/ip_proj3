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

#i00 = np.random.random((3*i0.shape[0], 3*i0.shape[1]))*(100)
#i00 = i00.astype(int)
#i11 = np.random.random((3*i1.shape[0], 3*i1.shape[1]))*(100)
#i11 = i11.astype(int)
#
#i00[i0.shape[0]:2*i0.shape[0], i0.shape[1]:2*i0.shape[1]] = i0
#i11[i1.shape[0]:2*i1.shape[0], i1.shape[1]:2*i1.shape[1]] = i1
#img_filt, FILT, I, IMG_FILT = lo_pass_gauss(i00, 0, .2)
#p_filt, ORIG_FILT, P, F, G = gen_corr_filt(i00, i11, FILT)
##print_gen_corr_hist(imgs[ii], imgs[jj], p_filt, FILT, P, F, G , hist)
#
#plt.imshow(p_filt, cmap='gray')
#plt.show()
#exit()

for ii in range(0, len(imgs)):
    for jj in range(0, len(imgs)):

      ii_ylen = imgs[ii].shape[0]
      ii_xlen = imgs[ii].shape[1]      
      jj_ylen = imgs[jj].shape[0]
      jj_xlen = imgs[jj].shape[1]      
        
      #Pad with random noise
      f = np.random.random((3*imgs[ii].shape[0], 3*imgs[jj].shape[1]))*(100)
      g = np.random.random((3*imgs[ii].shape[0], 3*imgs[jj].shape[1]))*(100)
      f[ii_ylen:2*ii_ylen, ii_xlen:2*ii_xlen] = imgs[ii]
      g[jj_ylen:2*jj_ylen, jj_xlen:2*jj_xlen] = imgs[jj]      

      #code.interact(local=locals())

      # Generate gaussian filter 'FILT'
      img_filt, FILT, I, IMG_FILT = lo_pass_gauss(f, 0, .2)

      # Calculate Phase Correlation and Filter Phase correlation in Freq Dom.
      p_filt, ORIG_FILT, P, F, G = gen_corr_filt(f, g, FILT)

      # Print debug info like max, min, difference
      print('IMG 1 index: ', ii)
      print('IMG 2 index: ', jj)

      # Get max/min indices
      max_index = np.unravel_index(p_filt.argmax(), p_filt.shape)
      print('Orig Max index: ', max_index)      
      min_index = np.unravel_index(p_filt.argmin(), p_filt.shape)
      max_index = (np.int(max_index[0]/3),np.int(max_index[1]/3))
      print('Modified Max index: ', max_index)

      if(max_index[0]> 510 or max_index[1] > 510):
          print("Continuing to next loop ii: ", ii, "  jj: ", jj)
          continue
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


      # Loop over each quadrant
      q1_y_orig = ii_ylen + max_index[0] - imgs[ii].shape[0]
      q1_x_orig = ii_xlen + max_index[1] - imgs[ii].shape[1]

      q2_y_orig = ii_ylen + max_index[0] 
      q2_x_orig = ii_xlen + max_index[1] - imgs[ii].shape[1]
      
      q3_y_orig = ii_ylen + max_index[0]
      q3_x_orig = ii_xlen + max_index[1]
      
      q4_y_orig = ii_ylen + max_index[0] - imgs[ii].shape[0]
      q4_x_orig = ii_xlen + max_index[1]

      # Place image 2 in 4 quadrants in canvas using correlation point
      q1 = np.array(f)
      q1[q1_y_orig:q1_y_orig + ii_ylen, q1_x_orig:q1_x_orig + ii_xlen] = imgs[jj]

      q2 = np.array(f)
      q2[q2_y_orig:q2_y_orig + ii_ylen, q2_x_orig:q2_x_orig + ii_xlen] = imgs[jj]

      q3 = np.array(f)
      q3[q3_y_orig:q3_y_orig + ii_ylen, q3_x_orig:q3_x_orig + ii_xlen] = imgs[jj]

      q4 = np.array(f)
      q4[q4_y_orig:q4_y_orig + ii_ylen, q4_x_orig:q4_x_orig + ii_xlen] = imgs[jj]

      plt.figure(1)
      plt.subplot(2,2,1)
      plt.title('q1')
      plt.imshow(q1, cmap='gray')
      plt.subplot(2,2,2)
      plt.title('q4')      
      plt.imshow(q4, cmap='gray')      
      plt.subplot(2,2,3)
      plt.title('q2')            
      plt.imshow(q2, cmap='gray')            
      plt.subplot(2,2,4)
      plt.title('q3')                  
      plt.imshow(q3, cmap='gray')
      
      plt.figure(2)
      plt.subplot(1,2,1)
      plt.imshow(imgs[ii], cmap='gray')
      plt.subplot(1,2,2)
      plt.imshow(imgs[jj], cmap='gray')
      plt.show()

      code.interact(local=locals())      





      

      
      


      
      




               

      

      

  
  

  

  
