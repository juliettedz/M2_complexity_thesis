import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

## Files
images_folder = './data/'
kernels_folder = './kernels/'
real = ''
stimuli_type_sparse = 'SparseNoise'
stimuli_type_dense = 'DenseNoise'
image_format = '.pickle'
kernel_format = '.npy'

remove_axis = True
save_figures = True


delay = 14
collapse_to = 5

#### Load the dense part
filename_h1kernel_dense = kernels_folder + stimuli_type_dense + 'real_regresion_h1' + kernel_format
filename_h2kernel_dense = kernels_folder + stimuli_type_dense + 'real_regresion_h2' + kernel_format 

h1_dense = np.load(filename_h1kernel_dense)
h2_dense = np.load(filename_h2kernel_dense)


#### Plot the dense part 
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

aux1 = np.min(h1_dense[delay,...])
aux2 = np.min(h2_dense[delay,...])
vmin = np.min([aux1, aux2])

aux1 = np.max(h1_dense[delay,...])
aux2 = np.max(h2_dense[delay,...])
vmax = np.max([aux1, aux2])

vmin = None
vmax = None


figure = plt.gcf()
ax = plt.gca()
plt.imshow(h1_dense[delay, ...], interpolation='bilinear', cmap=blue_red1,  vmin=vmin, vmax=vmax)


if remove_axis:
    figure.get_axes()[0].get_xaxis().set_visible(False)
    figure.get_axes()[0].get_yaxis().set_visible(False)


folder = './figures/'
format = '.pdf'
title = 'example_h1'
save_filename = folder + title + format 
figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi=100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

plt.show()

figure = plt.gcf()
ax = plt.gca()
im = ax.imshow(h2_dense[delay, ...], interpolation='bilinear', cmap=blue_red1,  vmin=vmin, vmax=vmax)

if remove_axis:
    figure.get_axes()[0].get_xaxis().set_visible(False)
    figure.get_axes()[0].get_yaxis().set_visible(False)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="10%", pad=0.15)
plt.colorbar(im, cax=cax)

title = 'example_h2'
save_filename = folder + title + format 
figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

plt.show()