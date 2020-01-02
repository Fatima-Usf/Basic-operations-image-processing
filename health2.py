from PIL import Image
from numpy import *
from pylab import *
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


""" --------- part 3 -------------- """


# read image to array
im = array(Image.open("health2.jpeg").convert("L"))
# create a new figure
figure()
# don’t use colors
gray()
# 1- contour, show contours with origin upper left corner
contour(im, origin="image")
axis("equal")
axis("off")
figure()
# 2 - Histograme
hist(im.flatten(),128)
show()