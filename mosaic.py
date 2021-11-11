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
#from skimage.filter import edges

clouds = io.imread('clouds.png')
i0 = io.imread('cell_images/cell_images/0001.000.png')
i1 = io.imread('cell_images/cell_images/0001.001.png')
i2 = io.imread('cell_images/cell_images/0001.002.png')
i4 = io.imread('cell_images/cell_images/0001.004.png')
i5 = io.imread('cell_images/cell_images/0001.005.png')
i6 = io.imread('cell_images/cell_images/0001.006.png')

img_arr=[i0, i1, i2, i4, i5, i6]


for ii in range(0, len(img_arr)):  
  # plot ideal
  img_filt, FILT, I, IMG_FILT = lo_pass_gauss(i0, 0, .2)
  p_filt, ORIG_FILT, P, F, G = gen_corr_filt(i0, img_arr[ii], FILT)
  print_gen_corr(i0, img_arr[ii], p_filt, FILT, P, F, G )

code.interact(local=locals())

freq1 = np.log(np.abs(np.fft.fft2(delta1(510))))
plt.imshow(freq1, cmap='gray')
plt.show()

exit()


# filter ideal
# 


#np.shape(i0)
#np.shape(i1)
#max_len_y = 2*max([np.shape(i0)[0], np.shape(i1)[0]])
#max_len_x = 2*max([np.shape(i0)[1], np.shape(i1)[1]])
#canvas = np.zeros((max_len_y, max_len_x))
#canvas[0:np.int(max_len_x/2), 0:np.int(max_len_y/2)] = i1
#canvas[375:375+np.int(max_len_y/2), 0:np.int(max_len_x/2)] = i0
##canvas[0:np.int(max_len_x/2), 0:np.int(max_len_y/2)] = i0
##canvas[np.int(max_len_x/2):2*np.int(max_len_x/2), np.int(max_len_y/2):2*np.int(max_len_y/2)] = i0
#
#code.interact(local=locals())
#exit()


#lo_pass_btrw_print(clouds, .2, 3)
F = np.fft.fft2(i0)
F_conj = F.conj()
G = np.fft.fft2(i1)
numer = F_conj*G
denom = np.abs(numer)
P = numer/denom
#img_filt, FILT, I, IMG_FILT = lo_pass_btrw(P, 1, 1)
img_filt, FILT, I, IMG_FILT = lo_pass_gauss(P, 0, 1, np.shape(P)[0], np.shape(P)[0])
#code.interact(local=locals())
#f_corr = np.abs(np.fft.ifft2(P))
f_corr = np.abs(np.fft.ifft2(IMG_FILT))
#plt.subplot(2,2,1)
#plt.imshow(i0, cmap='gray')
#plt.subplot(2,2,2)
#plt.imshow(i1, cmap='gray')
#plt.subplot(2,2,3)
plt.imshow(f_corr, cmap='gray')
plt.savefig('f_corr.png')
plt.show()

exit()

i0_filt = lo_pass_gauss(i0, 0, 0.1, np.shape(i0)[0], np.shape(i0)[1])

plt.subplot(2,1,1)
plt.imshow(i0, cmap='gray')
plt.subplot(2,1,2)
plt.imshow(i0_filt, cmap='gray')
plt.show()

#F1 = np.real(np.fft.fft2(i0))

#g_img = fftpack.ifftshift(gs2(0,1, np.shape(i0)[0], np.shape(i0)[0]))
#F1_filt = np.multiply(g_img,F1)
#i0_orig = np.real(np.fft.ifft2(F1))
#i0_filt = np.real(fftpack.ifftshift(np.fft.ifft2(F1_filt)))
#g_filt = gs2(0,.02, np.shape(i0)[0], np.shape(i0)[0])
#g_img = fftpack.ifftshift(gs2(0,.2, np.shape(i0)[0], np.shape(i0)[0]))
#F1 = np.fft.fft2(i0)
#F1_log = np.real(np.log(fftpack.ifftshift(np.fft.fft2(i0))))
#F1_filt = F1*g_img
#i0_filt = np.real(np.fft.ifft2(F1_filt))
#
#plt.subplot(3,2,1)
#plt.title('i0')
#plt.imshow(i0, cmap='gray')
#plt.subplot(3,2,2)
#plt.title('F1_log')
#plt.imshow(F1_log, cmap='gray')
#plt.subplot(3,2,3)
#plt.title('g_img')
#plt.imshow(g_filt, cmap='gray')
#plt.subplot(3,2,4)
#plt.title('i0_filt')
#plt.imshow(i0_filt, cmap='gray')
#plt.show()

exit()

filt = lo_p(np.shape(i0)[0], 300)

F1 = np.fft.fft2(i0)
F1_filt = np.multiply(F1, filt)
F1_mag = np.abs(F1)
F1_mag_filt = np.abs(F1_filt)
iF1 = np.real(np.fft.ifft2(F1))
iF1_filt = np.real(np.fft.ifft2(F1_filt))
#iF1 = 0.5*iF1
#iF1[iF1 > 255] = 255

plt.subplot(3,2,1)

plt.title('i0')
plt.imshow(i0, cmap='gray')
plt.subplot(3,2,2)
plt.title('F1_mag')
plt.imshow(np.log(F1_mag), cmap='gray')
plt.subplot(3,2,3)
plt.title('F1_mag_filt')
plt.imshow(np.log(F1_mag_filt), cmap='gray')
plt.subplot(3,2,4)
plt.title('iF1')
plt.imshow(iF1, cmap='gray')
plt.subplot(3,2,5)
plt.title('iF1_filt')                      
plt.imshow(iF1_filt, cmap='gray')
plt.show()
exit()

filt_len1 = np.shape(i0)[0]
filt_ones = np.ones((filt_len1,filt_len1))
F_i0_orig = np.fft.fft2(i0)
F_i0 = abs(np.fft.fft2(i0))
F_i0_pwr = np.log(F_i0**2)
x_len1 = np.shape(i0)[0]
y_len1 = np.shape(i0)[1]
lo_filt = .02*filt_ones
#lo_filt = lo_p(x_len1, 128)
F_i0_filt = np.multiply(F_i0_orig, lo_filt)
#F_i0_pwr_filt = np.log(abs(F_i0_filt)**2)
F_i0_pwr_filt = np.log(abs(F_i0_filt)**2)
i0_filt = np.fft.ifft(F_i0_filt)



#F_i0_orig = np.fft.fft2(i0)
#F_i0 = abs(np.fft.fft2(i0))
#F_i0_pwr = np.log(F_i0**2)
#x_len1 = np.shape(i0)[0]
#y_len1 = np.shape(i0)[1]
#lo_filt = lo_p(x_len1, 128)
#F_i0_filt = np.multiply(F_i0_orig, lo_filt)
#F_i0_pwr_filt = np.log(abs(F_i0_filt)**2)
#i0_filt = np.fft.ifft(F_i0_filt)




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
