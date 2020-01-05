from PIL import Image, ImageDraw, ImageFont
from numpy import *
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import filters, measurements, morphology
image = Image.open('health2.jpeg')
"""
def histeq(im,nbr_bins=256):

# get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
#use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

im_array = np.array(image.convert('L'))
im2, cdf = histeq(im_array)

out = Image.fromarray(im2)
out.show()

hist(im2.flatten(),128)
show()
save_file = open('histogram-equalization','ab')
pickle.dump(im2, save_file)
"""
"""
im = Image.open('health2.jpeg')
im.show()"""
"""
box = (40,30,90,70)
### cropping a region from an image is done using the crop() method.
region = im.crop(box)
region.show()"""

"""# 4-color conversion to grayscale
im = Image.open('health2.jpeg').convert("L")
im.show()
# Convert using adaptive palette of color depth 8
im = Image.open('health2.jpeg').convert("P", palette=Image.ADAPTIVE, colors=8)
im.show()"""

"""im = Image.open('health2.jpeg').convert("L")
im.show()
# Convert using adaptive palette of color depth 8
im = Image.open('health2.jpeg').convert("P", palette=Image.ADAPTIVE, colors=8)
im.show()"""
"""
im.thumbnail((128,128))
im.show()"""

"""
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
show()"""

"""im = np.flipud(plt.imread('health2.jpeg'))
plt.subplot(2, 1, 1)
plt.imshow(im)
plt.subplot(2, 1, 2)
plt.imshow(np.fliplr(im))
plt.show()"""

"""im = array(Image.open("health21.jpeg"))
imshow(im)
print("Please click 3 points")
x = ginput(3)
print("you clicked:",x )
show() """
"""
image = Image.open('health2.jpeg')

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
out.show()"""


# PCA
"""def pca(X):
        
    num_data, dim = X.shape
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim > num_data:
        M = np.dot(X, X.T)
        e, EV = np.linalg.eigh(M)
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        S = np.sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U,S,V = np.linalg.svd(X)
        V = V[:num_data]
        
    return V, S, mean_X

img_shape = np.array(image.convert('L')).shape


Image.open('health2.jpeg').resize((img_shape[1], img_shape[0])).save('health3.jpeg')
Image.open('health21.jpeg').resize((img_shape[1], img_shape[0])).save('health213.jpeg')
#Image.open('ConvertedImg.jpg').resize((img_shape[1], img_shape[0])).save('covrt.jpg')


imlist = np.array(['health2.jpeg','health3.jpeg', 'health213.jpeg'])

m, n = img_shape[0:2]

imnbr = len(imlist)

immatrix = np.array([np.array(Image.open(im).convert('L')).flatten() for im in imlist], 'f')

V, S, immean = pca(immatrix)

out = plt.figure()
plt.gray()
plt.subplot(2, 4, 1)
plt.imshow(immean.reshape(m,n))
for i in range(3):
    plt.subplot(2, 4, i+2)
    plt.imshow(V[i].reshape(m, n))
    
with open('pca','wb') as f:
    pickle.dump(out, f)"""


"""from scipy.ndimage import measurements,morphology
# load image and threshold to make sure it is binary
im = array(Image.open("health2.jpeg").convert("L"))
im = 1*(im<128)
labels, nbr_objects = measurements.label(im)
print("Number of objects:", nbr_objects)
save_file = open('nbr_objects', 'wb')
pickle.dump(nbr_objects, save_file)"""

"""#Gaussian blur
image = Image.open('health2.jpeg')
image = np.array(image.convert('L'))

im2 = filters.gaussian_filter(image, 2)
img2= Image.fromarray(im2)
img2.show()
save_file = open('gaussian-blur','ab')

pickle.dump(im2, save_file)"""

"""image = Image.open('health2.jpeg')

image = np.array(image.convert('L'))

image_x = np.zeros(image.shape)
filters.sobel(image, 1, image_x)

image_y = np.zeros(image.shape)
filters.sobel(image, 0, image_y)

out = np.sqrt(image_x**2 + image_y**2)

Image.fromarray(out).show()

save_file = open('derivative-image','ab')

pickle.dump(out, save_file)"""

"""image = Image.open('health2.jpeg')

image = np.array(image.convert('L'))

image_x = np.zeros(image.shape)
filters.sobel(image, 1, image_x)

image_y = np.zeros(image.shape)
filters.sobel(image, 0, image_y)

out = np.sqrt(image_x**2 + image_y**2)

Image.fromarray(out).show()

save_file = open('derivative-image','ab')

pickle.dump(out, save_file)"""

"""#Count Object
from scipy.ndimage import measurements,morphology
# load image and threshold to make sure it is binary
im = array(Image.open("health2.jpeg").convert("L"))
im = 1*(im<128)
labels, nbr_objects = measurements.label(im)
print("Number of objects:", nbr_objects)

save_file = open('nbr_objects', 'wb')
pickle.dump(nbr_objects, save_file)"""

def pca(X):
        
    num_data, dim = X.shape
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
 #Calcule des valeurs er vecteurs propores   
    if dim > num_data:
        M = np.dot(X, X.T) #covariance matrice
        e, EV = np.linalg.eigh(M)
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        S = np.sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U,S,V = np.linalg.svd(X)
        V = V[:num_data]
        
    return V, S, mean_X

img_shape = np.array(image.convert('L')).shape


Image.open('health2.jpeg').resize((img_shape[1], img_shape[0])).save('health3.jpeg')
Image.open('health21.jpeg').resize((img_shape[1], img_shape[0])).save('health213.jpeg')
#Image.open('ConvertedImg.jpg').resize((img_shape[1], img_shape[0])).save('covrt.jpg')


imlist = np.array(['health2.jpeg','health3.jpeg', 'health213.jpeg'])

m, n = img_shape[0:2]

imnbr = len(imlist)

immatrix = np.array([np.array(Image.open(im).convert('L')).flatten() for im in imlist], 'f')

V, S, immean = pca(immatrix)

out = plt.figure()
plt.gray()
plt.subplot(2, 4, 1)
plt.imshow(immean.reshape(m,n))
for i in range(3):
    plt.subplot(2, 4, i+2)
    plt.imshow(V[i].reshape(m, n))
    
with open('pca','wb') as f:
    pickle.dump(out, f)



"""
def pca(X):
#get dimensions
    num_data,dim = X.shape
# center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    if dim>num_data:
# PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # Valeurs propres et vecteurs propres
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # inverser puisque les valeurs propres sont en ordre croissant
        for i in range(V.shape[1]):  #shape dimension du tableau 
            V[:,i] /= S
    else:
# PCA - SVD used
                U,S,V = linalg.svd(X)
                V = V[:num_data] # only makes sense to return the first num_data
# return the projection matrix, the variance and the mean
    return V,S,mean_X
imPca = array(Image.open('health2.jpeg').convert('L'))
print(pca(imPca))"""