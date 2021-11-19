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
rows, cols = (len(imgs), len(imgs))
pos_arr = [[0 for i in range(cols)] for j in range(rows)]

canvas_size_y = 2*len(imgs)*imgs[0].shape[0] + imgs[0].shape[0]
canvas_size_x = 2*len(imgs)*imgs[0].shape[1] + imgs[0].shape[1]
canvas = np.zeros((canvas_size_y, canvas_size_x))

# Pad images in all directions with ave value
pad_imgs = pad_ave_arr(imgs)
filt = lo_pass_gauss(pad_imgs[0], 0, .8)[1]
corr_threshold = 100000


for ff in range(0, len(imgs)):
    for gg in range(0, len(imgs)):

        temp_y = len(imgs)*imgs[ff].shape[0]
        temp_x = len(imgs)*imgs[ff].shape[1]        
        
        if(ff == 0 and gg == 0):
            canvas[temp_y:temp_y + imgs[0].shape[0], temp_x:temp_x + imgs[0].shape[1]] = imgs[0]
            pos_arr[0][0] = [temp_y, temp_x]

        plt.imshow(canvas, cmap='gray')
        plt.show()
        
        
        #phase correlation
        corr = phase_corr(pad_imgs[ff], pad_imgs[gg], filt)
        quad = quad_img(draw_bound(imgs[ff]))
        quad_comp = np.array(quad)
        y_off = np.unravel_index(corr.argmax(), corr.shape)[0]
        x_off = np.unravel_index(corr.argmax(), corr.shape)[1]
        max_val = corr[y_off, x_off]
        mean_val = np.mean(corr)
        
        if(mean_val != 0):
          max_mean_ratio = max_val/mean_val
        else:
          max_mean_ratio = 0
          
        bound = draw_bound(np.array(imgs[gg]))
        
        y_len = np.int(pad_imgs[ff].shape[0])
        x_len = np.int(pad_imgs[ff].shape[1])                
        y_ctr = np.int(pad_imgs[ff].shape[0]/2)        
        x_ctr = np.int(pad_imgs[ff].shape[1]/2)

        # Calculate offset in y, wrap around if negative
        dy = y_ctr-y_off
        dy_orig = dy
        if(dy < 0):
            dy = imgs[ff].shape[0] + dy
            
        # Calculate offset in x, wrap around if negative            
        dx = x_ctr-x_off
        dx_orig = dx
        if(dx < 0):
            dx = imgs[ff].shape[1] + dx

        print("Image F index: ", ff)
        print("Image G index: ", gg)        
        print('y offset: ', y_off)
        print('x offset: ', x_off)
        print('dy: ', dy)
        print('dx: ', dx)
        print('Max Corr Val: ', max_val)
        print('Mean Corr Val: ', mean_val)
        print('Max/Mean Ratio', max_mean_ratio)
        
        if( 0.008 > max_val):
            print('Uncorrelated')
            continue

        #quad_comp[dy:dy + imgs[ff].shape[0], dx:dx + imgs[ff].shape[1]] = quad_comp[dy:dy + imgs[ff].shape[0], dx:dx + imgs[ff].shape[1]] + bound
        quad_comp[dy:dy + imgs[ff].shape[0], dx:dx + imgs[ff].shape[1]] = bound

        if(dy == 0 and dx == 0):
          print('SAME IMAGE\n\n\n')
          continue

        q_idx = quad_corr_max_index(imgs[ff], imgs[gg], dy, dx)

        if(len(pos_arr[ff][ff]) == 2 and pos_arr[ff][gg] == 0):
            fy_pos = pos_arr[ff][ff][0]            
            fx_pos = pos_arr[ff][ff][1]

            if( q_idx == 1):
                pos_arr[gg][gg] = [fy_pos + dy, fx_pos + dx]
                pos_arr[ff][gg] = [dy, dx]                
            elif( q_idx == 2):
                pos_arr[gg][gg] = [fy_pos + dy_orig, fx_pos + dx]
                pos_arr[ff][gg] = [dy, dx]                                
            elif( q_idx == 3):
                pos_arr[gg][gg] = [fy_pos + dy_orig, fx_pos + dx_orig]
                pos_arr[ff][gg] = [dy_orig, dx_orig]                                                
            elif( q_idx == 4):
                pos_arr[gg][gg] = [fy_pos + dy_orig, fx_pos + dx_orig]
                pos_arr[ff][gg] = [dy, dx_orig]

            canvas[pos_arr[gg][gg][0]:pos_arr[gg][gg][0] + imgs[gg].shape[0], pos_arr[gg][gg][1]:pos_arr[gg][gg][1] + imgs[gg].shape[1]] = imgs[gg]
            code.interact(local=locals())



        
        # Index 1 dy+, dx+ -> dy,      dx
        # Index 2 dy-, dx+ -> dy_orig, dx
        # Index 3 dy-, dx- -> dy_orig, dx_orig
        # Index 4 dy+, dx- -> dy,      dx_orig

        code.interact(local=locals())            
        
        #for ii in range(1, 5):

            
            #f_quad, g_quad = quad_corr(imgs[ff], imgs[gg], dy, dx, ii)
            #print('f', ii, ' shape: ', f_quad.shape)
            #print('g', ii, ' shape: ', g_quad.shape)            
            #plt.subplot(1,3,1)
            #plt.imshow(quad_comp, cmap='gray')
            #plt.subplot(1,3,2)
            #plt.imshow(g_quad, cmap='gray')
            #plt.subplot(1,3,3)
            #plt.imshow(f_quad, cmap='gray')
            
            #plt.show()
            #code.interact(local=locals())            
            

        
        ## Quadrant 1
        #q1_f = imgs[ff][dy:dy+imgs[ff].shape[0], dx:dx+imgs[ff].shape[1]]
        #q1_g = imgs[gg][0:imgs[gg].shape[0]-dy, 0:imgs[gg].shape[1]-dx]
        #
        #
        #q2_f = imgs[ff][0:dy, dx:imgs[ff].shape[1]]
        #q2_g = imgs[gg][imgs[gg].shape[0]-dy:imgs[gg].shape[0], 0:imgs[gg].shape[1]-dx]
        #
        #q3_f = imgs[ff][0:dy,0:dx]
        #q3_g = imgs[gg][imgs[gg].shape[0]-dy:imgs[gg].shape[0], imgs[gg].shape[1]-dx:imgs[gg].shape[1]]
        #
        #q4_f = imgs[ff][dy:imgs[ff].shape[0], 0:dx]
        #q4_g = imgs[gg][0:imgs[gg].shape[0]-dy, imgs[gg].shape[1]-dx:imgs[gg].shape[1]]

        plt.imshow(quad_comp, cmap='gray')
        plt.show()        
        
        code.interact(local=locals())

        

        



  

  
