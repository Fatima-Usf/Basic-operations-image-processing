from PIL import Image, ImageDraw, ImageFont
from numpy import *
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import filters, measurements, morphology
image = Image.open('health2.jpeg')

