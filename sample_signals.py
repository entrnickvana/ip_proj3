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

def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(y)))
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, -m/2+1,axis=0)
    cc = np.roll(cc, -n/2+1,axis=1)
    return cc

def nrm_test(dim, mu, sgm):
    mean = 0; std = 1; variance = np.square(std);
    x = np.arange(-5,4,.01)
    f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

    plt.plot(x,f)
    plt.ylabel('gaussian')
    plt.show()
    
def gs2_proto(muu, sg):
    x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    z = np.sqrt(x*x+y*y)
    sigma = 1
    muu = 0.00
    gauss1 = np.exp(-1*((z-muu)**2 / (2.0 * sg**2)))
    return gauss1

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

def lo_pass_btrw_print(img, cutoff, order):
    FILT = fftpack.ifftshift(btrw(np.shape(img)[0], np.shape(img)[1], cutoff, order))
    I = np.fft.fft2(img)
    IMG_FILT = I*FILT
    img_filt = np.real(np.fft.ifft2(IMG_FILT))
    plt_flt(img, FILT, I, IMG_FILT, img_filt)

def plt_flt(img, FILT, I, IMG_FILT, img_filt):
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
    
def gs2(muu, sg, w, h):
    w_half = np.int(w/2)
    h_half = np.int(h/2)
    x, y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
    z = np.sqrt(x*x + y*y)
    sigma = 1
    gauss1 = np.exp(-1*((z-muu)**2 / (2.0 * sg**2)))
    return gauss1

def trim(img):
    x_len, y_len = image.shape
    
    if (x_len % 2) == 0:
        img = img[:-1, :]
    if (y_len % 2) == 0:
        img = img[:, :-1]
        
    return img

#def lo_pass_gauss(img, muu, sg, w, h):
#    #Shift the gaussian filter as it is constructed to be in the center of X,Y, this doesn't
#    #match fourier tranforms of images
#    FILT = fftpack.ifftshift(gs2(muu, sg, np.shape(img)[0], np.shape(img)[1]))
#    I = np.fft.fft2(img)
#    IMG_FILT = I*FILT
#    return np.real(np.fft.ifft2(IMG_FILT))

def lo_pass_gauss(img, muu, sg):
    #Shift the gaussian filter as it is constructed to be in the center of X,Y, this doesn't
    #match fourier tranforms of images
    FILT = fftpack.fftshift(gs2(muu, sg, np.shape(img)[0], np.shape(img)[1]))
    I = np.fft.fft2(img)
    IMG_FILT = I*FILT
    img_filt = np.abs(np.fft.ifft2(IMG_FILT))
    return img_filt, FILT, I, IMG_FILT

def gen_corr(a, b):
    F = np.fft.fft2(a)
    F_conj = F.conj()
    G = np.fft.fft2(b)
    numer = F_conj*G
    denom = np.abs(numer)
    P = numer/denom
    return P, F, G

def gen_corr_filt(a, b, filt):
    P, F, G = gen_corr(a, b)
    #p_filt = np.fft.ifft2(P*filt)
    p_filt = fftpack.fftshift(np.abs(np.fft.ifft2(P*filt)))    
    return p_filt, filt, P, F, G

def print_gen_corr(img1, img2, p_filt, filt, P, F, G):
    plt.subplot(2,4,1)
    plt.title('img1')
    plt.imshow(img1, cmap='gray')
    
    plt.subplot(2,4,2)
    plt.title('img2')
    plt.imshow(img2, cmap='gray')
    
    plt.subplot(2,4,3)
    plt.title('filtered_corr')
    plt.imshow(p_filt, cmap='gray')
    
    plt.subplot(2,4,4)
    plt.title('filter (shifted)')
    plt.imshow(fftpack.ifftshift(filt), cmap='gray')
    
    plt.subplot(2,4,5)
    plt.title('Phase Correlation')
    plt.imshow(np.real(P), cmap='gray')
    
    plt.subplot(2,4,6)
    plt.title('F spectrum (Log)')
    plt.imshow(np.log(np.abs(F)), cmap='gray')    
    
    plt.subplot(2,4,7)
    plt.title('G spectrum (Log)')
    plt.imshow(np.log(np.abs(G)), cmap='gray')
    plt.show()

def print_gen_corr_hist(img1, img2, p_filt, filt, P, F, G, hist):
    plt.subplot(2,4,1)
    plt.title('img1')
    plt.imshow(img1, cmap='gray')
    
    plt.subplot(2,4,2)
    plt.title('img2')
    plt.imshow(img2, cmap='gray')
    
    plt.subplot(2,4,3)
    plt.title('filtered_corr')
    plt.imshow(p_filt, cmap='gray')
    
    plt.subplot(2,4,4)
    plt.title('filter (shifted)')
    plt.imshow(fftpack.ifftshift(filt), cmap='gray')
    
    plt.subplot(2,4,5)
    plt.title('Phase Correlation')
    plt.imshow(np.real(P), cmap='gray')
    
    plt.subplot(2,4,6)
    plt.title('F spectrum (Log)')
    plt.imshow(np.log(np.abs(fftpack.ifftshift(F))), cmap='gray')    
    
    plt.subplot(2,4,7)
    plt.title('G spectrum (Log)')
    plt.imshow(np.log(np.abs(fftpack.ifftshift(G))), cmap='gray')

    plt.subplot(2,4,8)
    plt.title('Histogram')
    plt.plot(hist)
    
    plt.show()
    
    

def lo_pass_(img, muu, sg, w, h):
    #Shift the gaussian filter as it is constructed to be in the center of X,Y, this doesn't
    #match fourier tranforms of images
    FILT = fftpack.ifftshift(gs2(muu, sg, np.shape(img)[0], np.shape(img)[1]))
    I = np.fft.fft2(img)
    IMG_FILT = I*FILT
    return np.real(np.fft.ifft2(IMG_FILT))

def three_d(img):
    #ax = plt.axes(projection='3d')
    #zline = np.linspace(0, 15, 1000)
    #xline = np.sin(zline)
    #yline = np.cos(zline)
    #ax.plot3D(xline, yline, zline, 'gray')
    #plt.show()

    ax = plt.axes(projection='3d')
    zline = linspace(0,255)

#def centroid(x, y, kern_w):
#    x_st = x-kern_w
    
    
def delta1(dim):
    #len1 = np.shape(img)[0]
    len1 = dim
    cntr = np.int(len1/2)
    img = np.zeros((dim, dim))
    img[cntr, cntr] = 1.0
    return img



def lo_p(dim, wdth):
    #len1 = np.shape(img)[0]
    img = np.zeros((dim,dim))
    len1 = dim
    cntr = np.int(len1/2)
    w=np.int(wdth/2)
    img[cntr-w:cntr+w, cntr-w:cntr+w] = 1.0
    return img

def mfft(img):
    return np.fft.fft2(img)

def mag_fft(img):
    return np.abs(mfft(img))

def phs_fft(img):
    return np.angle(mfft(img))

def mfft(img):
    res = np.fft.fft2(img)
    mag = np.abs(res)
    phs = np.angle(res)
    return mag, phs

def fft_pwr(img):
    return mfft(img)**2


def helly():
    print("helly")

    

    
    
