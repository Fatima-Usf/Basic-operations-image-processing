from PIL import Image
from numpy import *
from pylab import *
# Basic operations 
# 1- Resizing an image
# A- Chargement de l'image 
im = Image.open('health2.jpeg')
im.show()
image= im.resize((400,300))
image.show()



# 4-Converssion de couleurs au niveau de gris 
im = Image.open('health2.jpeg').convert('L')
im.show()""""" 