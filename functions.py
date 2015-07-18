import numpy as np
import math, cv2
from scipy import misc
import matplotlib.pyplot as plt
from scipy.sparse import *
import scipy.ndimage as ndimage
import scipy.sparse.linalg as spla
import numpy.matlib
import contextlib
import glob


numpy.set_printoptions(threshold=numpy.nan, precision=4)

def readIm(im=None, ext=None, *args, **kwargs):
    return cv2.imread(im, 1)

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

def getA(I, J,numPixels):

    brightestJ = np.zeros((numPixels,3))
    x_dim, y_dim = J.shape
    min_index = np.argmin(brightestJ[:,2])
    min_element = np.amin(brightestJ[:,2])

    for i in xrange(0,x_dim):
        for j in xrange(0,y_dim):

            min_index = np.argmin(brightestJ[:,2])
            min_element = np.amin(brightestJ[:,2])
            if J[i,j] > min_element:
                # print 'np.array([i,j,J[i,J]])', np.array([i+1,j+1,J[i,j]])
                brightestJ[min_index,:] = np.array([i+1,j+1,J[i,j]])

    highestIntensity = np.zeros((1,1,3))

    for i in xrange(0,numPixels):
        x = brightestJ[i,0]
        y = brightestJ[i,1]

        intensity = np.sum(I[x,y,:])
        if intensity >  np.sum(highestIntensity):
            highestIntensity[:,:,:] = I[x,y,:]

    __, __, dimI = I.shape

    if dimI == 3:
        A = np.zeros((x_dim, y_dim,3))
        for a in xrange(0, 3):
            A[:, :, a]  = A[:, :, a] + highestIntensity[:,:,a]
    else:
        A = np.zeros((x_dim,y_dim))
        A[:,:] = A[:,:] + highestIntensity[:,:]

    tmp = np.array(A[:,:,0])
    A[:,:,0] = A[:,:,2]
    A[:,:,2] = tmp
    return A

# def darkChannel(im2=None, *args, **kwargs):
#     height, width, __ = im2.shape
#     patchSize = 3
#     padSize = math.floor(patchSize / 2.0)
#     JDark = np.zeros((height,width))
#     imJ = np.pad(im2, (padSize, padSize), 'constant', constant_values=(0, 0))

#     for j in xrange(0, height):
#       for i in xrange(0, width):
#          patch = imJ[j:(j + patchSize - 1), i:(i + patchSize - 1), :]
#          JDark[j, i] = np.amin(patch[:])

#     return JDark


def makeDarkChannel(I, patchsize):
    height, width, channels = I.shape
    J = np.zeros((height, width))
    padsize = math.floor(patchsize/2) 
    I_R = np.pad(I[:,:,2], (padsize, padsize),  'symmetric')
    I_G = np.pad(I[:,:,1], (padsize, padsize),  'symmetric')
    I_B = np.pad(I[:,:,0], (padsize, padsize),  'symmetric')
    I = np.zeros((height+padsize*2,width+padsize*2, 3))

    I[:,:,0] = I_R 
    I[:,:,1] = I_G
    I[:,:,2] = I_B 

    tmpPatch = np.zeros((padsize, padsize, channels))

    for i in xrange(0, height):
        minX = i
        maxX = (i+padsize*2)

        for j in xrange(0, width):
            minY = j
            maxY = (j+ padsize*2)

            tmpPatch = I[minX:maxX,minY:maxY,:]
            # print 'tmpPatch', tmpPatch
            J[i,j] = np.amin(tmpPatch[:])

    return J


def preprocessImage(I):
    max_of_I = np.amax(I)

    if max_of_I > 768:
        scale = 768 / max_of_I
        I = imresize(I, scale)
    I = I / 255.0

    # make gray scales to color 
    height, width, channels = I.shape
    if channels == 2:
        # print 'now changing grayscale image into color image version'
        tmpI = np.zeros((height,width,3))
        for c in xrange(0,3):
            tmpI[:,:,c] = I
        I = tmpI

    return I



