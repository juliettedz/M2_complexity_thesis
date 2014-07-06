import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_functions import plot_mutliplot_bilinear

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
show_plot = True


delay = 14
collapse_to = 5

#### Load the dense part
filename_h1kernel_dense = kernels_folder + stimuli_type_dense + 'real_regresion_h1' + kernel_format
filename_h2kernel_dense = kernels_folder + stimuli_type_dense + 'real_regresion_h2' + kernel_format

h1_dense = np.load(filename_h1kernel_dense)
h2_dense = np.load(filename_h2kernel_dense)

kernel_size = h1_dense.shape[0]

# Plotting
############
symmetric = 1
colorbar = True
closest_square_to_kernel = int(np.sqrt(kernel_size)) ** 2

# Plot dense

directory = './figures/'
formating='.pdf'
title = 'real_regresion_h1' + stimuli_type_dense
save_filename = directory + title + formating

plot_mutliplot_bilinear(closest_square_to_kernel, h1_dense, colorbar=colorbar, symmetric=symmetric)
figure = plt.gcf()  # get current figure

if remove_axis:
    #Remove axis
    for i in xrange(closest_square_to_kernel):
        figure.get_axes()[i].get_xaxis().set_visible(False)
        figure.get_axes()[i].get_yaxis().set_visible(False)

figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

if show_plot:
    plt.show()


plot_mutliplot_bilinear(closest_square_to_kernel, h2_dense, colorbar=colorbar, symmetric=symmetric)
title = 'real_regresion_h2' + stimuli_type_dense
save_filename = directory + title + formating
figure = plt.gcf() # get current figure

if remove_axis:
# Remove axis
    for i in xrange(closest_square_to_kernel):
        figure.get_axes()[i].get_xaxis().set_visible(False)
        figure.get_axes()[i].get_yaxis().set_visible(False)

figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

if show_plot:
    plt.show()


