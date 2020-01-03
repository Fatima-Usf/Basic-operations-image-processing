from PIL import Image, ImageDraw, ImageFont
from numpy import *
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import filters, measurements, morphology


def histeq(im,nbr_bins=256):
  # get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
  # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

im = array(Image.open('health2.jpeg').convert('L'))
im2,cdf = histeq(im)

save_file = open('histogram-equalization','ab')
pickle.dump(out, save_file)