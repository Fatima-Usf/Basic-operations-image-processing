from PIL import Image, ImageDraw, ImageFont
from numpy import *
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import filters, measurements, morphology

image = Image.open('health2.jpeg')
image = np.array(image.convert('L'))

im2 = filters.gaussian_filter(image, 2)
img2= Image.fromarray(im2)
img2.show()
save_file = open('gaussian-blur','ab')

pickle.dump(im2, save_file)

#Averaging 
def compute_average(imlist):
    averageim = np.array(Image.open(imlist[0]), 'f')
    
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)
    
    return np.array(averageim, 'uint8')

img_shape = np.array(image.convert('L')).shape

Image.open('health2.jpeg').resize((img_shape[1], img_shape[0])).save('health3.jpeg')
Image.open('health21.jpeg').resize((img_shape[1], img_shape[0])).save('health213.jpeg')

imlist = np.array(['health3.jpeg', 'health213.jpeg'])

out_array = compute_average(imlist)

out = Image.fromarray(out_array)

save_file = open('averaging', 'ab')

pickle.dump(out, save_file)

# """ Histogram equalization of a grayscale image. """
def histeq(im,nbr_bins=256):

# get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
#use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

im_array = np.array(image.convert('L'))
out, cdf = histeq(im_array)

save_file = open('histogram-equalization','ab')
pickle.dump(out, save_file)