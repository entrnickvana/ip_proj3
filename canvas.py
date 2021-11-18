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
#from sample_signals import *
from canvas_funcs import *

clouds = io.imread('clouds.png')
i0 = io.imread('cell_images/cell_images/0001.000.png')
i1 = io.imread('cell_images/cell_images/0001.001.png')
i2 = io.imread('cell_images/cell_images/0001.002.png')
i4 = io.imread('cell_images/cell_images/0001.004.png')
i5 = io.imread('cell_images/cell_images/0001.005.png')
i6 = io.imread('cell_images/cell_images/0001.006.png')

#imgs = [i0, i1, i2, i4, i5, i6]


test1 = test_img((256, 256))

# Create test partitions
part_imgs = partition_imag4(test1, 64)
#partition_imag4_print(test1, partition_imag4(test1, 64))

# Use generic array for loop
#imgs = part_imgs
imgs = [i0, i1, i2, i4, i5, i6]

# Pad images in all directions with ave value
pad_imgs = pad_ave_arr(imgs)
filt = lo_pass_gauss(pad_imgs[0], 0, .8)[1]

for ff in range(0, len(imgs)):
    for gg in range(0, len(imgs)):
        
        #phase correlation
        corr = phase_corr(pad_imgs[ff], pad_imgs[gg], filt)
        quad = quad_img(draw_bound(imgs[ff]))
        quad_comp = np.array(quad)
        y_off = np.unravel_index(corr.argmax(), corr.shape)[0]
        x_off = np.unravel_index(corr.argmax(), corr.shape)[1]
        bound = draw_bound(np.array(imgs[gg]))
        y_len = np.int(pad_imgs[ff].shape[0])
        x_len = np.int(pad_imgs[ff].shape[1])                
        y_ctr = np.int(pad_imgs[ff].shape[0]/2)        
        x_ctr = np.int(pad_imgs[ff].shape[1]/2)
        dy = y_ctr-y_off
        if(dy < 0):
            dy = imgs[ff].shape[0] + dy
        dx = x_ctr-x_off
        if(dx < 0):
            dx = imgs[ff].shape[1] + dx
        
        print('y offset: ', y_off)
        print('x offset: ', x_off)
        print('dy: ', dy)
        print('dx: ', dx)        
        
        #quad_comp[dy:dy + imgs[ff].shape[0], dx:dx + imgs[ff].shape[1]] = quad_comp[dy:dy + imgs[ff].shape[0], dx:dx + imgs[ff].shape[1]] + bound
        quad_comp[dy:dy + imgs[ff].shape[0], dx:dx + imgs[ff].shape[1]] = bound        
        plt.imshow(quad_comp, cmap='gray')
        plt.show()        
        
        code.interact(local=locals())


        

        



  

  
