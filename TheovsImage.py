#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:53:29 2023

@author: nic
"""

import numpy as np
from numpy import transpose as tp
from numpy import sin,cos,sqrt,exp,pi,arctan
from scipy.special import jv
import matplotlib.pyplot as plt
import cv2
# install with 'pip install opencv'
# or 'conda install opencv' if you use anaconda

### Theoretical Plots ###

N,M = 4,3 # Dimensions of plot in mm
          # (roughly the size of the CCD sensor)
r = 400 # Resolution of plot
        # (change this at your discretion)
x=np.linspace(-N/2,N/2,N*r)
y=np.linspace(-M/2,M/2,M*r)

X,Y = np.meshgrid(x,y)
R = sqrt(X**2 + Y**2)
# You may find these useful

# Now define the functions!


### Capturing Image Data ###

im = cv2.imread("path/to/image.ext")

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# Convert to greyscale as we only want intensity


### Plotting ###
fig, ax = plt.subplots()
# ax.matshow(circtheo, cmap='turbo')
# ax.matshow(squaretheo, cmap='turbo')
ax.matshow(im, cmap='turbo')
plt.axis('off')
plt.show()
