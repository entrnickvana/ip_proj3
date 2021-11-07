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


def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(y)))
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, -m/2+1,axis=0)
    cc = np.roll(cc, -n/2+1,axis=1)
    return cc

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

    

    
    
