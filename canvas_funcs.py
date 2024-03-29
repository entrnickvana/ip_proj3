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
from numpy.fft import fft2, ifft2
from scipy import fftpack

def mosaic_core(imgs):
    print('Beginning Mosiac, this may take a minute...')

    debug = 0
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
    
            if(debug == 1):
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
                
            if(debug == 1):
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
    
            quad_comp[dy:dy + imgs[ff].shape[0], dx:dx + imgs[ff].shape[1]] = bound
    
            if(dy == 0 and dx == 0):
              if(debug == 1):
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
    
    plt.figure()
    plt.imshow(canvas, cmap='gray')
    plt.savefig('canvas.png')

    return canvas


def quad_corr(ff, gg, dy, dx, quad_num):

    if(quad_num == 1):
        q1_f = ff[dy:dy+ff.shape[0], dx:dx+ff.shape[1]]
        q1_g = gg[0:gg.shape[0]-dy, 0:gg.shape[1]-dx]
        return q1_f, q1_g
        
    if(quad_num == 2):
        q2_f = ff[0:dy, dx:ff.shape[1]]
        q2_g = gg[gg.shape[0]-dy:gg.shape[0], 0:gg.shape[1]-dx]
        return q2_f, q2_g
    
    if(quad_num == 3):
        q3_f = ff[0:dy,0:dx]
        q3_g = gg[gg.shape[0]-dy:gg.shape[0], gg.shape[1]-dx:gg.shape[1]]
        return q3_f, q3_g
        
    if(quad_num == 4):
        q4_f = ff[dy:ff.shape[0], 0:dx]
        q4_g = gg[0:gg.shape[0]-dy, gg.shape[1]-dx:gg.shape[1]]
        return q4_f, q4_g

    return

def quad_corr_max_index(ff, gg, dy, dx):

    corr_max_arr = []
    
    q1_f = ff[dy:dy+ff.shape[0], dx:dx+ff.shape[1]]
    q1_g = gg[0:gg.shape[0]-dy, 0:gg.shape[1]-dx]
    corr_max_arr.append(norm_corr(q1_f, q1_g))
    
    q2_f = ff[0:dy, dx:ff.shape[1]]
    q2_g = gg[gg.shape[0]-dy:gg.shape[0], 0:gg.shape[1]-dx]
    corr_max_arr.append(norm_corr(q2_f, q2_g))    
    
    q3_f = ff[0:dy,0:dx]
    q3_g = gg[gg.shape[0]-dy:gg.shape[0], gg.shape[1]-dx:gg.shape[1]]
    corr_max_arr.append(norm_corr(q3_f, q3_g))            

    q4_f = ff[dy:ff.shape[0], 0:dx]
    q4_g = gg[0:gg.shape[0]-dy, gg.shape[1]-dx:gg.shape[1]]
    corr_max_arr.append(norm_corr(q4_f, q4_g))
    
    #print('Quadrant corr values: ', corr_max_arr)
    #code.interact(local=locals())            
    max_corr_idx = corr_max_arr.index(max(corr_max_arr))
    #print('Index selected: ', max_corr_idx)
    return max_corr_idx + 1

def norm_corr(a, b):
    
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_hat = a-(a_mean*np.ones(a.shape))
    b_hat = b-(b_mean*np.ones(b.shape))
    a_hat_mag = np.sqrt(np.sum(a_hat*a_hat))
    b_hat_mag = np.sqrt(np.sum(b_hat*b_hat))    
    n_corr = (np.sum(a_hat*b_hat))/(a_hat_mag*b_hat_mag + .0001)
    
    return n_corr

def btrw(w, h, cutoff, order):
    w_half = np.int(w/2)
    h_half = np.int(h/2)
    x, y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
    z = np.sqrt(x*x + y*y)
    butter = 1/(1 + (z/cutoff)**(2*order))
    return butter

def lo_pass_btrw(img, cutoff, order):
    FILT = fftpack.ifftshift(btrw(np.shape(img)[0], np.shape(img)[1], cutoff, order))
    I = np.fft.fft2(img)
    IMG_FILT = I*FILT
    img_filt = np.real(np.fft.ifft2(IMG_FILT))
    return img_filt, FILT, I, IMG_FILT
    #return img_filt, FILT, I, IMG_FILT


def draw_bound(img):
    new_img = np.array(img)
    new_img[0:1, ::] = 0
    new_img[::, 0:1] = 0
    new_img[img.shape[0]-1:img.shape[0], ::] = 0
    new_img[::, img.shape[1]-1:img.shape[1]] = 0            
    return new_img

def pad_ave_arr(imgs):
    pad_arr = []
    for ii in range(len(imgs)):
        pad_arr.append(pad_average(imgs[ii]))
    return pad_arr

def quad_img(img):
    q_img = np.zeros((2*img.shape[0], 2*img.shape[1]))
    q_img[0:img.shape[0], 0:img.shape[1]] = img
    q_img[img.shape[0]:2*img.shape[0], 0:img.shape[1]] = img
    q_img[img.shape[0]:2*img.shape[0], img.shape[1]:2*img.shape[1]] = img
    q_img[0:img.shape[0], img.shape[1]:2*img.shape[1]] = img
    return q_img

def pad_average(img):
    av = np.mean(img)
    pad_img = av*np.ones((3*img.shape[0], 3*img.shape[1]))
    pad_img[img.shape[0]:2*img.shape[0], img.shape[1]:2*img.shape[1]] = img
    return pad_img

def phase_corr(f, g, filt):
    eps = .00001
    fft_prod = np.fft.fft2(f).conj()*np.fft.fft2(g)
    fft_prod_phase = fft_prod/(np.abs(fft_prod) + eps)
    filt_phase = filt*fft_prod_phase
    
    return np.real(np.fft.fftshift(np.fft.ifft2(filt_phase)))

def gs2(muu, sg, w, h):
    w_half = np.int(w/2)
    h_half = np.int(h/2)
    #x, y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
    x, y = np.meshgrid(np.linspace(-1,1,h), np.linspace(-1,1,w))    
    z = np.sqrt(x*x + y*y)
    sigma = 1
    gauss1 = np.exp(-1*((z-muu)**2 / (2.0 * sg**2)))
    return gauss1

def lo_pass_gauss(img, muu, sg):
    #Shift the gaussian filter as it is constructed to be in the center of X,Y, this doesn't
    #match fourier tranforms of images
    FILT = fftpack.fftshift(gs2(muu, sg, np.shape(img)[0], np.shape(img)[1]))
    I = np.fft.fft2(img)
    IMG_FILT = I*FILT
    img_filt = np.abs(np.fft.ifft2(IMG_FILT))
    return img_filt, FILT, I, IMG_FILT

def test_img(test_shape):
    tst = gs2(0, .2, test_shape[0], test_shape[1])
    tst[tst >= 0.5] = .5
    tst[tst < 0.5] = .1
    tst = tst-1
    tst = -1*tst
    return tst

def color2grey(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)


def plt_flt_plot(img, FILT, I, IMG_FILT, img_filt):
    plt.subplot(2,3,1)
    plt.title('Input Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(2,3,2)
    plt.title('FFT image (Magnitude Logscale)')    
    plt.imshow(np.log(np.abs(fftpack.ifftshift(I))), cmap='gray')
    plt.subplot(2,3,3)
    plt.title('Centered Filter (Magnitude Logscale)')    
    plt.imshow(np.log(np.abs(fftpack.ifftshift(FILT))), cmap='gray')
    plt.subplot(2,3,4)
    plt.title('Fourier Domain Filtered Image (Magnitude Logscale)')        
    plt.imshow(np.log(np.abs(fftpack.ifftshift(IMG_FILT))), cmap='gray')
    plt.subplot(2,3,5)
    plt.title('Filtered Output Image')            
    plt.imshow(img_filt, cmap='gray')
    plt.show()


def partition_imag4(img, overlap):
    half_len_y = np.int(img.shape[0]/2)
    half_len_x = np.int(img.shape[1]/2)
    
    s1 = img[0:half_len_x+overlap, 0:half_len_x+overlap]
    s2 = img[overlap:2*half_len_y, 0:half_len_x+overlap]
    s3 = img[overlap:2*half_len_y, overlap:2*half_len_x]
    s4 = img[0:half_len_y+overlap, overlap:2*half_len_x]

    return s1, s2, s3, s4

def partition_imag4_print(img, imgs):

    s1 = imgs[0]
    s2 = imgs[1]
    s3 = imgs[2]
    s4 = imgs[3]
    
    fig = plt.figure()

    gs = fig.add_gridspec(2,4)
    ax0 = fig.add_subplot(gs[:, 0:2])
    plt.imshow(img, cmap='gray')
    ax1 = fig.add_subplot(gs[0, 2])
    plt.imshow(s1, cmap='gray')
    ax2 = fig.add_subplot(gs[1, 2])
    plt.imshow(s2, cmap='gray')
    ax4 = fig.add_subplot(gs[0, 3])
    plt.imshow(s4, cmap='gray')
    ax5 = fig.add_subplot(gs[1, 3])
    plt.imshow(s3, cmap='gray')
    plt.show()



    


    
    
    
    

    
    
