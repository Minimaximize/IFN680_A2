

'''

A short script to illustrate the warping functions of 'assign2_utils'

'''

#import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib import keras
#from tensorflow.contrib.keras import backend as K


import assign2_utils



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#ims = x_train[0]
#plt.imshow(ims,cmap='gray')
#ims2 = x_test

for i in range(10):
    plt.figure()
#    print("Train "+str(i))
    plt.imshow(x_train[i], cmap='gray')
    plt.figure()
    g = assign2_utils.random_deform(x_train[i],20,0.1)
#    print("Test "+str(i))
    plt.imshow(g, cmap='gray')
    
#
#
#im2 = assign2_utils.random_deform(im1,45,0.3)
#
#plt.figure()
#
#plt.imshow(im2,cmap='gray')
#
#plt.show()