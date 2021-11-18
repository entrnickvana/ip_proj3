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


# TODO
# 1) Thresholding on correlation values
# 2) Variable image size correction
# 3) 
#
#
#


c1 = lo_pass_gauss(np.zeros((256, 256)), 0, .1)
c1 = np.real(fftpack.fftshift(c1[1]))
c1[c1 >= 0.5] = .8
c1[c1 < 0.8] = .2
c1 = c1-1
c1 = -1*c1

plt.imshow(c1, cmap='gray')
plt.show()

vect = np.arange(0, 256**2)
vect = (1/np.max(vect))*vect
steps1 = vect.reshape((256, 256))
steps1[0:16, 0:32] = 1
steps1[128-16: 128 + 16, 0:128+64] = .5
steps1[32:256-32, 256-16:256-8] = .25

steps1 = np.array(c1)

s1 = steps1[0:128+64, 0:128+64]
s2 = steps1[64:256, 0:128+64]
s3 = steps1[64:256, 64:564]
s4 = steps1[0:128+64, 64:256]

fig = plt.figure()

gs = fig.add_gridspec(2,4)
ax0 = fig.add_subplot(gs[:, 0:2])
plt.imshow(steps1, cmap='gray')
ax1 = fig.add_subplot(gs[0, 2])
plt.imshow(s1, cmap='gray')
ax2 = fig.add_subplot(gs[1, 2])
plt.imshow(s2, cmap='gray')
ax4 = fig.add_subplot(gs[0, 3])
plt.imshow(s4, cmap='gray')
ax5 = fig.add_subplot(gs[1, 3])
plt.imshow(s3, cmap='gray')
plt.show()

code.interact(local=locals())

pad_s1 = pad_ave(s1)
pad_s4 = pad_ave(s4)
img_filt, FILT, I, IMG_FILT = lo_pass_gauss(pad_s1, 0, .8)
p_filt, ORIG_FILT, P, F, G = gen_corr_filt(pad_s1, pad_s4, FILT)
print_gen_corr(pad_s1, pad_s4, p_filt, ORIG_FILT, P, F, G)
#code.interact(local=locals())

clouds = io.imread('clouds.png')
i0 = io.imread('cell_images/cell_images/0001.000.png')
i1 = io.imread('cell_images/cell_images/0001.001.png')
i2 = io.imread('cell_images/cell_images/0001.002.png')
i4 = io.imread('cell_images/cell_images/0001.004.png')
i5 = io.imread('cell_images/cell_images/0001.005.png')
i6 = io.imread('cell_images/cell_images/0001.006.png')

#imgs = [i0, i1, i2, i4, i5, i6]

imgs = [s1, s2, s3, s4]

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
code.interact(local=locals())

##mosiac = np.random.random((3*imgs[ii].shape[0], 3*imgs[jj].shape[1]))*(100)
for ii in range(0, len(imgs)):
    for jj in range(0, len(imgs)):

      ii_ylen = imgs[ii].shape[0]
      ii_xlen = imgs[ii].shape[1]      
      jj_ylen = imgs[jj].shape[0]
      jj_xlen = imgs[jj].shape[1]      
        
      #Pad with random noise
      #f = np.random.random((3*imgs[ii].shape[0], 3*imgs[jj].shape[1]))*(100)
      #g = np.random.random((3*imgs[ii].shape[0], 3*imgs[jj].shape[1]))*(100)
      f = pad_ave(imgs[ii])
      g = pad_ave(imgs[jj])
      f[ii_ylen:2*ii_ylen, ii_xlen:2*ii_xlen] = imgs[ii]
      g[jj_ylen:2*jj_ylen, jj_xlen:2*jj_xlen] = imgs[jj]      

      #code.interact(local=locals())

      # Generate gaussian filter 'FILT'
      img_filt, FILT, I, IMG_FILT = lo_pass_gauss(f, 0, .8)

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

      if(max_index[0] > 510 or max_index[1] > 510):
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
      #flat[flat == nan] = 0
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
      s1_f = imgs[ii][0:max_index[0], 0:max_index[1]]
      s1_g = imgs[jj][jj_ylen-max_index[0]:jj_ylen, jj_xlen-max_index[1]:jj_xlen]
      #q1_corr = gen_corr_no_filt(s1_f, s1_g)
      #q1_corr = q1_corr[0]
      #q1_corr_idx = np.unravel_index(q1_corr.argmax(), q1_corr.shape)
      ##code.interact(local=locals())
      #q1_corr_val = q1_corr[q1_corr_idx[0], q1_corr_idx[1]]
      #print('q2 corr: ', q1_corr_idx)      

      q2 = np.array(f)
      q2[q2_y_orig:q2_y_orig + ii_ylen, q2_x_orig:q2_x_orig + ii_xlen] = imgs[jj]
      s2_f = imgs[ii][max_index[0]:ii_ylen, 0:max_index[1]]
      s2_g = imgs[jj][0:jj_ylen-max_index[0], jj_xlen-max_index[1]:jj_xlen ]
      #q2_corr = gen_corr_no_filt(s2_f, s2_g)
      #q2_corr = q2_corr[0]      
      #q2_corr_idx = np.unravel_index(q2_corr.argmax(), q2_corr.shape)
      #q2_corr_val = q2_corr[q2_corr_idx[0], q2_corr_idx[1]]      
      #print('q2 corr: ', q2_corr_idx)      
      
      q3 = np.array(f)
      q3[q3_y_orig:q3_y_orig + ii_ylen, q3_x_orig:q3_x_orig + ii_xlen] = imgs[jj]
      s3_f = imgs[ii][max_index[0]:ii_ylen, max_index[1]:ii_xlen]
      s3_g = imgs[jj][0:jj_ylen-max_index[0] , 0:jj_xlen-max_index[1]]
      #q3_corr = gen_corr_no_filt(s3_f, s3_g)
      #q3_corr = q3_corr[0 ]     
      #q3_corr_idx = np.unravel_index(q3_corr.argmax(), q3_corr.shape)
      #q3_corr_val = q3_corr[q3_corr_idx[0], q3_corr_idx[1]]            
      #print('q3 corr: ', q3_corr_idx)            
 
      q4 = np.array(f)
      q4[q4_y_orig:q4_y_orig + ii_ylen, q4_x_orig:q4_x_orig + ii_xlen] = imgs[jj]
      s4_f = imgs[ii][0:max_index[0], max_index[1]:ii_xlen]
      s4_g = imgs[jj][jj_ylen-max_index[0]:jj_ylen, 0: max_index[1]-jj_xlen]
      #q4_corr = gen_corr_no_filt(s4_f, s4_g)
      #q4_corr = q4_corr[0]      
      #q4_corr_idx = np.unravel_index(q4_corr.argmax(), q4_corr.shape)
      #q4_corr_val = q4_corr[q4_corr_idx[0], q4_corr_idx[1]]                  
      #print('q4 corr: ', q4_corr_idx)

      #corr_arr = [q1_corr_val, q2_corr_val, q3_corr_val, q4_corr_val]
      #max_corr = max(corr_arr)
      #max_corr_idx = corr_arr.index(max_corr)
      #
      #print('corr idx: ', max_corr_idx, '  max val: ', max_corr)
      
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

      plt.figure(3)
      plt.subplot(1, 3, 1)
      plt.imshow(q1, cmap='gray')
      plt.subplot(1, 3, 2)
      plt.imshow(s1_f, cmap='gray')
      plt.subplot(1, 3, 3)      
      plt.imshow(s1_g, cmap='gray')

      plt.figure(4)
      plt.subplot(1, 3, 1)
      plt.imshow(q2, cmap='gray')
      plt.subplot(1, 3, 2)
      plt.imshow(s2_f, cmap='gray')
      plt.subplot(1, 3, 3)      
      plt.imshow(s2_g, cmap='gray')

      plt.figure(5)
      plt.subplot(1, 3, 1)
      plt.imshow(q3, cmap='gray')
      plt.subplot(1, 3, 2)
      plt.imshow(s3_f, cmap='gray')
      plt.subplot(1, 3, 3)      
      plt.imshow(s3_g, cmap='gray')
      
      plt.figure(6)
      plt.subplot(1, 3, 1)
      plt.imshow(q4, cmap='gray')
      plt.subplot(1, 3, 2)
      plt.imshow(s4_f, cmap='gray')
      plt.subplot(1, 3, 3)      
      plt.imshow(s4_g, cmap='gray')

      plt.show()
      
      code.interact(local=locals())      





      

      
      


      
      




               

      

      

  
  

  

  
