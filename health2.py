from PIL import Image
from numpy import *
from pylab import *
# Basic operations 
# A- Load an image
im = Image.open('health2.jpeg')
im.show()
# 1- Resizing an image
image= im.resize((400,300))
image.show()
#2 - Rotate an image
imrotated = im.rotate(45)
imrotated.show()


# 4-Converssion de couleurs au niveau de gris 
im = Image.open('health2.jpeg').convert('L')
im.show()