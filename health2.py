from PIL import Image
from numpy import *
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
# Basic operations 
""" --------- part 1 -------------- """
# A- Load an image
im = Image.open('health2.jpeg')
#im.show()
# 1- Resizing an image
image= im.resize((400,300))
#image.show()
#2 - Rotate an image
imrotated = im.rotate(45)
#imrotated.show()
#3- Crop an image
### Create a box to defin a region, the region is defined by a 4-tuple, where coordinates are (left, upper, right, lower).
box = (40,30,90,70)
### cropping a region from an image is done using the crop() method.
region = im.crop(box)
#region.show()

# 4-color conversion to grayscale
im = Image.open('health2.jpeg').convert("L")
#im.show()
# Convert using adaptive palette of color depth 8
im = Image.open('health2.jpeg').convert("P", palette=Image.ADAPTIVE, colors=8)
#im.show()

# thumbnail: Réduit ou aggrandit l'image
im.thumbnail((128,128))
#im.show()


""" --------- part 2 -------------- """

#Convert images to another format
# 1- from jpeg to jpg
if not im.mode == 'RGB':
  im = im.convert('RGB')
print(im.mode)
im.save('ConvertedImg.jpg', quality=95)

# 2- from jpg to jpeg
imgJpg = Image.open('ConvertedImg.jpg')
im.save('NewConvertedImg.jpeg', quality=95)


"""--------- part 3 -------------- """

# read image to array
im = array(Image.open("health2.jpeg").convert("L"))
# 1- contour, 
# create a new figure
figure()
# to not use colors in the contour
gray()
#show contours with origin upper left corner
contour(im, origin="image")
axis("equal")
axis("off")
figure()
# 2 - Histograme
hist(im.flatten(),128)
#show()


"""--------- part 4 -------------- """
#Anotation
#im = array(Image.open("health21.jpeg"))
#imshow(im)
#print("Please click 3 points")
#x = ginput(3)
#print("you clicked:",x )
#show() 

""" --------- part 5 -------------- """
#Array representation and gray transformation, resizing, averaging and equalizatio
im = array(Image.open("health2.jpeg"))
print(im.shape, im.dtype)

# Reverse - Way 1 :
im = np.flipud(plt.imread('health2.jpeg'))
plt.subplot(2, 1, 1)
plt.imshow(im)
plt.subplot(2, 1, 2)
plt.imshow(np.fliplr(im))
plt.show()
""" Reverse - Way 2 :
im = array(Image.open("health21.jpeg"))
imgOut = Image.fromarray(im)
imgOut.show()"""

# Reziding 
def imresize(im,sz):
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

im = np.flipud(plt.imread('health2.jpeg'))
imresize(im, (128,128))
show()

# """ Histogram equalization of a grayscale image. """
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