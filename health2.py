from PIL import Image
from numpy import *
from pylab import *
# Basic operations 
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
region.show()

# 4-Converssion de couleurs au niveau de gris 
im = Image.open('health2.jpeg').convert('L')
#im.show()