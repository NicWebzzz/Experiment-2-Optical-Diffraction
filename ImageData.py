#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:09:46 2023

@author: nic
"""

import numpy as np
from scipy.ndimage import gaussian_filter as gf
import matplotlib.pyplot as plt
import cv2


im = cv2.imread("imagename.format", 1)

# Converting to grayscale
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Uncomment for vertical slice
# im = np.transpose(im)


# Viewing image
# cv2.imshow('image',im)

irow=int(np.shape(im)[0]/2) # Location of middle row

inten = im[irow] # Taking just the middle row (slicing out the centre)

print('Max intensity is',max(inten))

if max(inten) == 255:
    print('Image is oversaturated!') # Brightness takes on an 8bit value 0-255


xsize = np.shape(inten)[0] # size of x-axis for plot


# Applying gaussian filter
# Useful for planewave analysis (smoothes noise)
fltr=20
gfinten=gf(inten, fltr)

x = range(xsize) # x-axis of plot

plt.figure('Beam intensity')
plt.plot(x, inten, color='orange',label='Image Data')
plt.plot(x, gfinten, color='blue',label='Gaussian Filter = {}'.format(fltr))
plt.ylim(0,255) # Gives a better representation of brightness range
plt.title('Beam Intensity', fontsize = 40)
plt.legend(fontsize=30)
plt.show()

plt.savefig('plotname.format')

# 3D representation of beam intensity
# Beware of uncommenting! Very slow and you may freeze your computer :(
# x = np.linspace(-1,1,np.shape(im)[1])
# y = np.linspace(-1,1,np.shape(im)[0])
# X, Y = np.meshgrid(x, y)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, gf(im, fltr), rstride=1, cstride=1, cmap='hot', edgecolor='none')
