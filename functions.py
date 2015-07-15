import numpy as np
import math, cv2
from scipy import misc
import matplotlib.pyplot as plt
from scipy.sparse import *
import scipy.ndimage as ndimage
import scipy.sparse.linalg as spla
import numpy.matlib
import contextlib

numpy.set_printoptions(threshold=numpy.nan, precision=4)

def readIm(im=None, ext=None, *args, **kwargs):
    return cv2.imread(im, 1)

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)
