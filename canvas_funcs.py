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

def quad_corr(imgs, ff, gg, quad_num):

    if(quad_num == 1):
        q1_f = imgs[ff][dy:dy+imgs[ff].shape[0], dx:dx+imgs[ff].shape[1]]
        q1_g = imgs[gg][0:imgs[gg].shape[0]-dy, 0:imgs[gg]-dx]
    if(quad_num == 2):
        q2_f = imgs[ff][0:dy, dx:imgs[ff].shape[1]]
        q2_g = imgs[gg][imgs[gg].shape[0]-dy:imgs[gg].shape[0], 0:imgs[gg].shape[1]-dx]
    if(quad_num == 3):
        q3_f = imgs[ff][0:dy,0:dx]
        q3_g = imgs[gg][imgs[gg].shape[0]-dy:imgs[gg].shape[0], imgs[gg].shape[1]-dx:imgs[gg].shape[1]]
        
    if(quad_num == 1):
        
        
    

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
    x, y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
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



    


    
    
    
    

    
    
