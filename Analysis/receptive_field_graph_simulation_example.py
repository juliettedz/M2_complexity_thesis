import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

## Files
images_folder = './data/'
kernels_folder = './kernels/'
quality = '_3000_21_'
stimuli_type_sparse = 'SparseNoise'
stimuli_type_dense = 'DenseNoise'
image_format = '.pickle'
kernel_format = '.npy'

cell_number = 1
delay = 1
collapse_to = 5

#### Load the dense part

filename_images_dense = images_folder + 'images' + quality + stimuli_type_dense + image_format
filename_h1kernel_dense = kernels_folder + 'h1' + str(cell_number) + stimuli_type_dense + kernel_format
filename_h2kernel_dense = kernels_folder + 'h2' + str(cell_number) + stimuli_type_dense + kernel_format

h1_dense = np.load(filename_h1kernel_dense)
h2_dense = np.load(filename_h2kernel_dense)


#### Load the sparse part
filename_images_sparse = images_folder + 'images' + quality + stimuli_type_sparse + image_format
filename_h1kernel_sparse = kernels_folder + 'h1' + str(cell_number) + stimuli_type_sparse + kernel_format
filename_h2kernel_sparse = kernels_folder + 'h2' + str(cell_number) + stimuli_type_sparse + kernel_format

h1_sparse = np.load(filename_h1kernel_sparse)
h2_sparse = np.load(filename_h2kernel_sparse)



#### Plot the sparse part
cdict1 = {'red':   ((0.0, 0.0, 0.0),
               (0.5, 0.0, 0.1),
               (1.0, 1.0, 1.0)),

     'green': ((0.0, 0.0, 0.0),
               (1.0, 0.0, 0.0)),

     'blue':  ((0.0, 0.0, 1.0),
               (0.5, 0.1, 0.0),
               (1.0, 0.0, 0.0))
    }

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

aux1 = np.min(h1_sparse[delay,...])
aux2 = np.min(h2_sparse[delay,...])
vmin = np.min([aux1, aux2])

aux1 = np.max(h1_sparse[delay,...])
aux2 = np.max(h2_sparse[delay,...])
vmax = np.max([aux1, aux2])

plt.subplot(1, 2, 1)
plt.imshow(h1_sparse[delay, ...], interpolation='bilinear', cmap=blue_red1,  vmin=vmin, vmax=vmax)


plt.subplot(1, 2, 2)
plt.imshow(h2_sparse[delay, ...], interpolation='bilinear', cmap=blue_red1,  vmin=vmin, vmax=vmax)


plt.show()





